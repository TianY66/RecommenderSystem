"""Tool-calling orchestration with traceable evidence checks."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .providers import LLMProvider, ProviderTurn, ToolCall
from .tools import BookTools, ToolValidationError


_BOOK_REFERENCE = re.compile(r"\[(I[A-Za-z0-9_-]+)\]")

_INSTRUCTIONS = """你是一个图书检索与推荐助手。回答必须基于本次工具返回的数据。
工作流程：先用 search_catalog 找候选；若用户提出个性化要求或给出 user_id，必须再用
rank_candidates_for_user 对这些候选排序；最后用 get_book_details 核验要推荐的图书。
如果排序结果表明没有可用的用户历史或画像，不要声称结果已按该用户偏好个性化。
只能推荐详情工具返回的图书，只能陈述详情中存在的字段。目录没有星级评分，不能声称书籍有评分。
若没有候选、用户身份不明或约束无法满足，要明确说明并在必要时询问澄清。不要编造图书、属性或推荐依据。
涉及具体图书时，请在书名后用方括号标注 item_id，例如 [I0001]。"""


@dataclass(frozen=True)
class AgentResult:
    success: bool
    answer: str
    error: str | None
    trace: dict[str, Any]


class BookAgent:
    def __init__(
        self,
        tools: BookTools,
        provider: LLMProvider,
        *,
        model_name: str,
        max_tool_rounds: int = 6,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        self.tools = tools
        self.provider = provider
        self.model_name = model_name
        self.max_tool_rounds = max_tool_rounds

    def run(self, user_message: str) -> AgentResult:
        started = time.monotonic()
        trace: dict[str, Any] = {
            "model": self.model_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "tool_calls": [],
            "detail_item_ids": [],
            "detail_evidence": {},
            "response_ids": [],
            "response_statuses": [],
            "usage": {},
        }
        if not isinstance(user_message, str) or not user_message.strip():
            return self._result(False, "请输入要查找或推荐的图书需求。", "empty_user_message", trace, started)

        session = self.tools.new_session()
        input_items: list[dict[str, Any]] = [
            {"role": "user", "content": user_message.strip()}
        ]
        previous_response_id: str | None = None
        tool_rounds = 0

        while True:
            try:
                turn = self.provider.create_response(
                    input_items=input_items,
                    tools=self.tools.tool_definitions(),
                    previous_response_id=previous_response_id,
                    instructions=_INSTRUCTIONS,
                )
            except Exception as exc:  # Provider failures are returned without leaking credentials.
                trace["provider_error_type"] = type(exc).__name__
                return self._result(
                    False,
                    "模型服务暂时无法完成请求，请稍后重试。",
                    "provider_error",
                    trace,
                    started,
                )

            self._record_turn(trace, turn)
            previous_response_id = turn.response_id
            if turn.tool_calls:
                if tool_rounds >= self.max_tool_rounds:
                    return self._result(
                        False,
                        "工具调用次数已达上限，未能完成可核验的回答。",
                        "maximum_tool_rounds_reached",
                        trace,
                        started,
                    )
                tool_rounds += 1
                input_items = [
                    self._execute_tool_call(call, session, trace)
                    for call in turn.tool_calls
                ]
                continue

            answer = (turn.output_text or "").strip()
            if not answer:
                return self._result(
                    False,
                    "模型没有返回可用回答。",
                    "empty_model_response",
                    trace,
                    started,
                )

            unverified_ids = sorted(
                {
                    item_id
                    for item_id in _BOOK_REFERENCE.findall(answer)
                    if item_id not in set(trace["detail_item_ids"])
                }
            )
            if unverified_ids:
                trace["unverified_item_ids"] = unverified_ids
                return self._result(
                    False,
                    "回答引用了未核验的书目，已拦截。",
                    "answer_contains_unverified_item_id",
                    trace,
                    started,
                )
            title_mismatches = sorted(
                item_id
                for item_id in _BOOK_REFERENCE.findall(answer)
                if item_id in trace["detail_evidence"]
                and trace["detail_evidence"][item_id].get("name")
                and trace["detail_evidence"][item_id]["name"] not in answer
            )
            if title_mismatches:
                trace["title_mismatch_item_ids"] = title_mismatches
                return self._result(
                    False,
                    "回答中的书名与详情核验结果不一致，已拦截。",
                    "answer_book_title_mismatch",
                    trace,
                    started,
                )
            return self._result(True, answer, None, trace, started)

    @staticmethod
    def _record_turn(trace: dict[str, Any], turn: ProviderTurn) -> None:
        if turn.response_id:
            trace["response_ids"].append(turn.response_id)
        trace["response_statuses"].append(turn.status)
        if turn.usage:
            for key, value in turn.usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    trace["usage"][key] = trace["usage"].get(key, 0) + value
                else:
                    trace["usage"][key] = value

    @staticmethod
    def _execute_tool_call(
        call: ToolCall,
        session: Any,
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "call_id": call.call_id,
            "name": call.name,
            "arguments": None,
            "result_item_ids": [],
            "error": None,
        }
        try:
            arguments = json.loads(call.arguments)
            record["arguments"] = arguments
        except (json.JSONDecodeError, TypeError):
            result: dict[str, Any] = {
                "error": "invalid_json_arguments",
                "message": "工具参数不是有效 JSON。",
            }
            record["error"] = "invalid_json_arguments"
        else:
            try:
                result = session.call(call.name, arguments)
                if call.name == "get_book_details":
                    detail_ids = [
                        row["item_id"]
                        for row in result.get("items", [])
                        if isinstance(row, dict) and row.get("item_id") is not None
                    ]
                    trace["detail_item_ids"] = list(
                        dict.fromkeys(trace["detail_item_ids"] + detail_ids)
                    )
                    trace["detail_evidence"].update(
                        {
                            row["item_id"]: row
                            for row in result.get("items", [])
                            if isinstance(row, dict) and row.get("item_id") is not None
                        }
                    )
                    record["result_item_ids"] = detail_ids
                elif call.name == "search_catalog":
                    record["result_item_ids"] = [
                        row["item_id"] for row in result.get("candidates", [])
                    ]
                elif call.name == "rank_candidates_for_user":
                    record["result_item_ids"] = [
                        row["item_id"] for row in result.get("ranked_candidates", [])
                    ]
                    record["user_context"] = result.get("user_context", {})
            except ToolValidationError as exc:
                code = "unknown_tool" if str(exc).startswith("Unknown tool:") else "tool_validation_error"
                result = {"error": code, "message": str(exc)}
                record["error"] = code
            except Exception as exc:
                result = {
                    "error": "tool_execution_error",
                    "message": "工具执行失败。",
                }
                record["error"] = "tool_execution_error"
                record["exception_type"] = type(exc).__name__

        trace["tool_calls"].append(record)
        return {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
        }

    def _result(
        self,
        success: bool,
        answer: str,
        error: str | None,
        trace: dict[str, Any],
        started: float,
    ) -> AgentResult:
        trace["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
        trace["success"] = success
        trace["error"] = error
        return AgentResult(success=success, answer=answer, error=error, trace=trace)
