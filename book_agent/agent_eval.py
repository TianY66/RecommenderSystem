"""Repeatable, trace-based evaluation for conversational book tasks."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import AgentResult
from .catalog import BookCatalog
from .evidence import answer_facts_supported

_ITEM_REFERENCE = re.compile(r"\[(I[A-Za-z0-9_-]+)\]")
_CLARIFICATION = re.compile(r"请.*(?:提供|告诉|补充|确认)|请问|需要.*(?:用户|编号)|\?|？")
_NO_RESULTS = re.compile(r"没有|未找到|暂无|无符合|不存在|找不到")


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            required = {"case_id", "prompt", "expected_tools", "expected_outcome", "constraints"}
            if not isinstance(case, dict) or required - case.keys():
                raise ValueError(f"Case on line {line_number} is missing required fields")
            case_id = case["case_id"]
            if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
                raise ValueError(f"Case IDs must be non-empty and unique (line {line_number})")
            if case["expected_outcome"] not in {"recommend", "clarify", "no_results"}:
                raise ValueError(f"Unsupported expected_outcome for case {case_id}")
            if not isinstance(case["expected_tools"], list) or not isinstance(case["constraints"], dict):
                raise ValueError(f"Invalid tools or constraints for case {case_id}")
            if not isinstance(case["prompt"], str) or not case["prompt"].strip():
                raise ValueError(f"Prompt must be non-empty for case {case_id}")
            seen_ids.add(case_id)
            cases.append(case)
    return cases


def score_case(
    case: Mapping[str, Any],
    result: AgentResult,
    catalog: BookCatalog,
) -> dict[str, Any]:
    trace = result.trace
    tool_calls = trace.get("tool_calls", [])
    actual_tools = [call.get("name") for call in tool_calls]
    expected_tools = list(case["expected_tools"])
    sequence_correct = actual_tools == expected_tools
    outcome = case["expected_outcome"]
    detail_ids = list(dict.fromkeys(trace.get("detail_item_ids", [])))
    cited_ids = list(dict.fromkeys(_ITEM_REFERENCE.findall(result.answer)))

    trace_complete = bool(trace.get("response_ids")) and all(
        isinstance(call, Mapping)
        and isinstance(call.get("call_id"), str)
        and bool(call.get("call_id"))
        and isinstance(call.get("name"), str)
        and bool(call.get("name"))
        for call in tool_calls
    )
    if outcome == "recommend":
        constraints_satisfied = _recommendation_satisfies_constraints(
            cited_ids, detail_ids, case.get("constraints", {}), catalog
        )
        constraints_satisfied = constraints_satisfied and len(cited_ids) >= int(
            case.get("min_recommendations", 1)
        )
        grounded = _answer_is_grounded(result.answer, cited_ids, detail_ids, catalog)
        facts_consistent = _stated_facts_are_supported(result.answer, cited_ids, catalog)
        personalization_claim_accurate = _personalization_claim_is_accurate(case, result.answer, tool_calls)
        failure_handled: bool | None = None
        end_to_end = bool(
            result.success
            and sequence_correct
            and constraints_satisfied
            and grounded
            and facts_consistent
            and personalization_claim_accurate
            and trace_complete
        )
    elif outcome == "clarify":
        constraints_satisfied = True
        grounded = not cited_ids and not detail_ids
        facts_consistent = grounded
        personalization_claim_accurate = True
        failure_handled = bool(
            result.success
            and not tool_calls
            and not detail_ids
            and _CLARIFICATION.search(result.answer)
        )
        end_to_end = bool(failure_handled and sequence_correct and trace_complete)
    else:
        search_calls = [call for call in tool_calls if call.get("name") == "search_catalog"]
        no_candidates = bool(search_calls) and all(
            not call.get("result_item_ids") for call in search_calls
        )
        constraints_satisfied = no_candidates and not detail_ids
        grounded = not cited_ids and not detail_ids
        facts_consistent = grounded
        personalization_claim_accurate = True
        failure_handled = bool(
            result.success
            and no_candidates
            and not detail_ids
            and _NO_RESULTS.search(result.answer)
        )
        end_to_end = bool(failure_handled and sequence_correct and trace_complete)

    return {
        "case_id": case["case_id"],
        "expected_outcome": outcome,
        "tool_sequence_correct": sequence_correct,
        "constraints_satisfied": bool(constraints_satisfied),
        "grounded_answer": bool(grounded),
        "facts_consistent": bool(facts_consistent),
        "personalization_claim_accurate": bool(personalization_claim_accurate),
        "failure_handling_applicable": failure_handled is not None,
        "failure_handled": failure_handled,
        "trace_complete": trace_complete,
        "end_to_end_success": end_to_end,
        "cited_item_ids": cited_ids,
        "detail_item_ids": detail_ids,
        "answer_fallback_used": bool(trace.get("answer_fallback_used", False)),
        "answer": result.answer,
        "error": result.error,
        "trace": trace,
    }


def summarize_results(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    results = list(rows)

    def rate(key: str, *, applicable_only: bool = False) -> float | None:
        values = [row.get(key) for row in results]
        if applicable_only:
            values = [value for value, row in zip(values, results) if row.get("failure_handling_applicable")]
        values = [bool(value) for value in values if value is not None]
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    return {
        "case_count": len(results),
        "tool_sequence_accuracy": rate("tool_sequence_correct"),
        "constraint_satisfaction_rate": rate("constraints_satisfied"),
        "grounded_answer_rate": rate("grounded_answer"),
        "fact_consistency_rate": rate("facts_consistent"),
        "personalization_claim_accuracy": rate("personalization_claim_accurate"),
        "failure_handling_rate": rate("failure_handled", applicable_only=True),
        "answer_fallback_rate": rate("answer_fallback_used"),
        "trace_completeness_rate": rate("trace_complete"),
        "end_to_end_success_rate": rate("end_to_end_success"),
    }


def run_agent_eval(
    agent: Any,
    cases: Iterable[Mapping[str, Any]],
    catalog: BookCatalog,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for case in cases:
        result = agent.run(case["prompt"])
        rows.append(score_case(case, result, catalog))

    report = {
        "started_at": started_at,
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "model": getattr(agent, "model_name", None),
        "summary": summarize_results(rows),
        "cases": rows,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def _recommendation_satisfies_constraints(
    cited_ids: list[str],
    detail_ids: list[str],
    constraints: Mapping[str, Any],
    catalog: BookCatalog,
) -> bool:
    if not cited_ids or not set(cited_ids).issubset(detail_ids):
        return False
    for item_id in cited_ids:
        item = catalog.get_item(item_id)
        if item is None:
            return False
        categories = _tag_set(item.get("item_categories"))
        keywords = _tag_set(item.get("item_keywords"))
        if constraints.get("category") and constraints["category"].casefold() not in categories:
            return False
        if constraints.get("keyword") and constraints["keyword"].casefold() not in keywords:
            return False
        price = _number(item.get("price"))
        minimum = constraints.get("min_price")
        maximum = constraints.get("max_price")
        if minimum is not None and (price is None or price < float(minimum)):
            return False
        if maximum is not None and (price is None or price > float(maximum)):
            return False
    return True


def _answer_is_grounded(
    answer: str,
    cited_ids: list[str],
    detail_ids: list[str],
    catalog: BookCatalog,
) -> bool:
    if not cited_ids or not set(cited_ids).issubset(detail_ids):
        return False
    for item_id in cited_ids:
        item = catalog.get_item(item_id)
        if item is None or not item.get("name") or str(item["name"]) not in answer:
            return False
        evidence = _tag_set(item.get("item_categories")) | _tag_set(item.get("item_keywords"))
        if not any(value in answer for value in evidence):
            return False
    return True


def _personalization_claim_is_accurate(
    case: Mapping[str, Any],
    answer: str,
    tool_calls: list[Mapping[str, Any]],
) -> bool:
    if not case.get("expect_no_personalization_claim"):
        return True
    rank_calls = [call for call in tool_calls if call.get("name") == "rank_candidates_for_user"]
    if not rank_calls:
        return False
    context = rank_calls[-1].get("user_context", {})
    if context.get("personalization_applied") is not False:
        return False
    claim_pattern = re.compile(
        r"(?:根据|根據).{0,8}(?:阅读|閱讀|浏览|瀏覽|购买|購買).{0,6}(?:历史|歷史|记录|記錄)"
        r"|(?:结合|結合).{0,8}(?:历史|歷史)|(?:个性化|個性化)推荐",
    )
    for claim in claim_pattern.finditer(answer):
        clause = re.split(r"[，,。；;\n]", answer[: claim.start()])[-1]
        if re.search(
            r"(?:不是|并非|并没有|没有|無|无|未|并未|不能|无法|不)(?:.{0,4})$",
            clause,
        ):
            continue
        return False
    return True


def _stated_facts_are_supported(
    answer: str,
    cited_ids: list[str],
    catalog: BookCatalog,
) -> bool:
    verified_items = {
        item_id: item
        for item_id in cited_ids
        if (item := catalog.get_item(item_id)) is not None
    }
    return answer_facts_supported(answer, cited_ids, verified_items)


def _tag_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        values = value.split(";")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    return {str(item).strip().casefold() for item in values if str(item).strip()}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None
