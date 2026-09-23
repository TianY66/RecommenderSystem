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
_USER_ID = re.compile(r"(?<![A-Za-z0-9])[Uu]\d+(?![A-Za-z0-9])")
_PERSONALIZATION_HINT = re.compile(
    r"阅读历史|阅读记录|历史偏好|个性化|按.{0,8}偏好|结合.{0,8}偏好|"
    r"based on.{0,12}(history|preferences)|reading history|personaliz",
    re.IGNORECASE,
)
_UNSUPPORTED_RATING_HINT = re.compile(
    r"评分|星级|几星|高于.{0,4}星|star rating|ratings?",
    re.IGNORECASE,
)
_TOOL_BY_STAGE = {
    "search": "search_catalog",
    "rank": "rank_candidates_for_user",
    "details": "get_book_details",
}
_KNOWN_TOOL_NAMES = frozenset(_TOOL_BY_STAGE.values())

_INSTRUCTIONS = """你是一个图书检索与推荐助手。回答必须基于本次工具返回的数据。
工作流程：先用 search_catalog 找候选；若用户提出个性化要求或给出 user_id，必须再用
rank_candidates_for_user 对这些候选排序；最后用 get_book_details 核验要推荐的图书。
每轮只能调用当前阶段提供的工具，并按检索、排序、详情的顺序执行。缺少个性化所需的用户 ID，
或用户要求目录未提供的星级评分时，先澄清，不要调用工具。
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
        tool_definitions = {
            definition["name"]: definition for definition in self.tools.tool_definitions()
        }
        personalization_requested = self._personalization_requested(user_message)
        workflow_stage = self._initial_stage(user_message, personalization_requested)
        trace["personalization_requested"] = personalization_requested
        trace["workflow_stages"] = []
        input_items: list[dict[str, Any]] = [
            {"role": "user", "content": user_message.strip()}
        ]
        previous_response_id: str | None = None
        tool_rounds = 0

        while True:
            trace["workflow_stages"].append(workflow_stage or "complete")
            available_tool = _TOOL_BY_STAGE.get(workflow_stage or "")
            available_tools = [tool_definitions[available_tool]] if available_tool else []
            try:
                turn = self.provider.create_response(
                    input_items=input_items,
                    tools=available_tools,
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
                input_items = []
                first_record_index = len(trace["tool_calls"])
                for index, call in enumerate(turn.tool_calls):
                    allowed_names = {available_tool} if available_tool and index == 0 else set()
                    forced_error = "parallel_tool_call_not_supported" if index else None
                    input_items.append(
                        self._execute_tool_call(
                            call,
                            session,
                            trace,
                            allowed_tool_names=allowed_names,
                            forced_error=forced_error,
                        )
                    )
                first_record = trace["tool_calls"][first_record_index]
                if first_record["error"] is None:
                    workflow_stage = self._next_stage(
                        workflow_stage,
                        first_record["result_item_ids"],
                        personalization_requested,
                    )
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
        *,
        allowed_tool_names: set[str],
        forced_error: str | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "call_id": call.call_id,
            "name": call.name,
            "arguments": None,
            "result_item_ids": [],
            "error": None,
        }
        if forced_error:
            result = {
                "error": forced_error,
                "message": "每轮只处理一个工具调用，请基于当前阶段重试。",
            }
            record["error"] = forced_error
        elif call.name not in _KNOWN_TOOL_NAMES:
            result = {"error": "unknown_tool", "message": f"Unknown tool: {call.name}"}
            record["error"] = "unknown_tool"
        elif call.name not in allowed_tool_names:
            result = {
                "error": "tool_not_available_in_stage",
                "message": "该工具不属于当前工作流阶段。请使用本轮提供的工具。",
            }
            record["error"] = "tool_not_available_in_stage"
        else:
            try:
                arguments = json.loads(call.arguments)
                record["arguments"] = arguments
            except (json.JSONDecodeError, TypeError):
                result = {
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
                    record["error"] = "tool_validation_error"
                    result = {"error": "tool_validation_error", "message": str(exc)}
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

    @staticmethod
    def _personalization_requested(user_message: str) -> bool:
        return bool(_USER_ID.search(user_message) or _PERSONALIZATION_HINT.search(user_message))

    @classmethod
    def _initial_stage(cls, user_message: str, personalization_requested: bool) -> str:
        if _UNSUPPORTED_RATING_HINT.search(user_message):
            return "clarify"
        if personalization_requested and not _USER_ID.search(user_message):
            return "clarify"
        return "search"

    @staticmethod
    def _next_stage(
        current_stage: str | None,
        result_item_ids: list[str],
        personalization_requested: bool,
    ) -> str | None:
        if not result_item_ids:
            return None
        if current_stage == "search":
            return "rank" if personalization_requested else "details"
        if current_stage == "rank":
            return "details"
        return None

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
