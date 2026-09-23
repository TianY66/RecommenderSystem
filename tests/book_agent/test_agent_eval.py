import json
import re

from book_agent.agent import AgentResult, BookAgent
from book_agent.agent_eval import load_cases, run_agent_eval, score_case, summarize_results
from book_agent.catalog import BookCatalog
from book_agent.data import load_demo_data
from book_agent.providers import ProviderTurn, ToolCall
from book_agent.ranking import RecommendationEngine
from book_agent.tools import BookTools


def result(answer, calls, detail_ids, *, success=True, error=None):
    return AgentResult(
        success=success,
        answer=answer,
        error=error,
        trace={
            "success": success,
            "error": error,
            "tool_calls": [
                {"call_id": f"call-{index}", "name": name, "error": None}
                for index, name in enumerate(calls)
            ],
            "detail_item_ids": detail_ids,
            "response_ids": ["r1"],
            "duration_ms": 12,
        },
    )


def test_score_case_checks_sequence_constraints_and_grounded_detail_evidence():
    catalog = BookCatalog(
        {
            "I1": {
                "item_id": "I1",
                "name": "算法入门",
                "item_categories": ("科技",),
                "item_keywords": ("算法", "编程"),
                "price": 45.0,
            },
            "I2": {
                "item_id": "I2",
                "name": "算法进阶",
                "item_categories": ("科技",),
                "item_keywords": ("算法",),
                "price": 88.0,
            },
        }
    )
    case = {
        "case_id": "personal-budget",
        "prompt": "结合 U1 的阅读历史推荐算法书",
        "expected_tools": ["search_catalog", "rank_candidates_for_user", "get_book_details"],
        "expected_outcome": "recommend",
        "constraints": {"category": "科技", "keyword": "算法", "max_price": 50},
    }
    good = result(
        "推荐《算法入门》[I1]。科技类，包含算法和编程关键词。",
        case["expected_tools"],
        ["I1"],
    )
    scored = score_case(case, good, catalog)
    assert scored["tool_sequence_correct"] is True
    assert scored["constraints_satisfied"] is True
    assert scored["grounded_answer"] is True
    assert scored["facts_consistent"] is True
    assert scored["end_to_end_success"] is True

    bad = result("推荐《算法进阶》[I2]。", case["expected_tools"], ["I2"])
    assert score_case(case, bad, catalog)["constraints_satisfied"] is False

    invented_price = result(
        "推荐《算法入门》[I1]。关键词包括算法，价格为 900 元。",
        case["expected_tools"],
        ["I1"],
    )
    assert score_case(case, invented_price, catalog)["facts_consistent"] is False


def test_score_case_accepts_quoted_supported_labels_and_rejects_unknown_labels():
    catalog = BookCatalog(
        {
            "I1": {
                "item_id": "I1",
                "name": "算法入门",
                "item_categories": ("科技",),
                "item_keywords": ("算法",),
                "price": 45.0,
            }
        }
    )
    case = {
        "case_id": "quoted-facts",
        "prompt": "推荐算法书",
        "expected_tools": ["search_catalog", "get_book_details"],
        "expected_outcome": "recommend",
        "constraints": {},
    }
    supported = result(
        "推荐《算法入门》[I1]。类别为「科技」，关键词含“算法”，价格为 45 元。",
        case["expected_tools"],
        ["I1"],
    )
    unsupported = result(
        "推荐《算法入门》[I1]。关键词含“伪造标签”。",
        case["expected_tools"],
        ["I1"],
    )

    assert score_case(case, supported, catalog)["facts_consistent"] is True
    assert score_case(case, unsupported, catalog)["facts_consistent"] is False


def test_score_case_checks_clarification_empty_results_and_unverified_citations():
    catalog = BookCatalog({"I1": {"item_id": "I1", "name": "算法入门", "price": 30}})
    clarify = {
        "case_id": "clarify",
        "prompt": "按我的历史推荐一本",
        "expected_tools": [],
        "expected_outcome": "clarify",
        "constraints": {},
    }
    clarified = result("请提供你的用户编号，我再结合历史推荐。", [], [])
    assert score_case(clarify, clarified, catalog)["failure_handled"] is True

    no_results = {
        "case_id": "empty",
        "prompt": "找不存在分类的书",
        "expected_tools": ["search_catalog"],
        "expected_outcome": "no_results",
        "constraints": {},
    }
    no_result = result("没有找到符合条件的图书。", ["search_catalog"], [])
    assert score_case(no_results, no_result, catalog)["failure_handled"] is True

    recommend = {
        "case_id": "fake-citation",
        "prompt": "推荐一本",
        "expected_tools": ["search_catalog", "get_book_details"],
        "expected_outcome": "recommend",
        "constraints": {},
    }
    fabricated = result("推荐《其他书》[I2]。", recommend["expected_tools"], ["I1"])
    scored = score_case(recommend, fabricated, catalog)
    assert scored["grounded_answer"] is False
    assert scored["trace_complete"] is True


def test_summary_aggregates_case_metrics_and_writes_no_hidden_assumptions():
    rows = [
        {"tool_sequence_correct": True, "constraints_satisfied": True, "grounded_answer": True,
         "failure_handled": True, "trace_complete": True, "end_to_end_success": True},
        {"tool_sequence_correct": False, "constraints_satisfied": True, "grounded_answer": False,
         "failure_handled": False, "trace_complete": True, "end_to_end_success": False},
    ]
    summary = summarize_results(rows)
    assert summary["case_count"] == 2
    assert summary["tool_sequence_accuracy"] == 0.5
    assert summary["constraint_satisfaction_rate"] == 1.0
    assert summary["grounded_answer_rate"] == 0.5
    assert summary["end_to_end_success_rate"] == 0.5


def test_unknown_user_case_flags_claims_of_history_personalization():
    catalog = BookCatalog(
        {
            "I1": {
                "item_id": "I1",
                "name": "算法入门",
                "item_categories": ("科技",),
                "item_keywords": ("算法",),
                "price": 45,
            }
        }
    )
    case = {
        "case_id": "unknown-user",
        "prompt": "為 U404 推薦算法書",
        "expected_tools": ["search_catalog", "rank_candidates_for_user", "get_book_details"],
        "expected_outcome": "recommend",
        "constraints": {},
        "expect_no_personalization_claim": True,
    }
    honest = result("推荐《算法入门》[I1]。科技类，关键词是算法。", case["expected_tools"], ["I1"])
    honest.trace["tool_calls"][1]["user_context"] = {"personalization_applied": False}
    assert score_case(case, honest, catalog)["personalization_claim_accurate"] is True

    misleading = result("根据你的阅读历史，推荐《算法入门》[I1]。关键词是算法。", case["expected_tools"], ["I1"])
    misleading.trace["tool_calls"][1]["user_context"] = {"personalization_applied": False}
    assert score_case(case, misleading, catalog)["personalization_claim_accurate"] is False

    explicitly_unpersonalized = result(
        "该用户没有可用历史，这不是个性化推荐。推荐《算法入门》[I1]，标签是算法。",
        case["expected_tools"],
        ["I1"],
    )
    explicitly_unpersonalized.trace["tool_calls"][1]["user_context"] = {
        "personalization_applied": False
    }
    assert score_case(case, explicitly_unpersonalized, catalog)[
        "personalization_claim_accurate"
    ] is True


class CaseDrivenFakeProvider:
    """Deterministic protocol simulator; this verifies harness plumbing, not model quality."""

    def __init__(self, cases):
        self.cases = {case["prompt"]: case for case in cases}
        self.case = None
        self.step = 0
        self.model_name = "offline-case-simulator"

    def create_response(self, *, input_items, tools, previous_response_id, instructions):
        if previous_response_id is None:
            prompt = input_items[0]["content"]
            self.case = self.cases[prompt]
            self.step = 0

        expected_tools = self.case["expected_tools"]
        if self.step < len(expected_tools):
            tool_name = expected_tools[self.step]
            arguments = self._arguments_for(tool_name, input_items)
            self.step += 1
            call = ToolCall(
                call_id=f"fake-{self.case['case_id']}-{self.step}",
                name=tool_name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            )
            return ProviderTurn(
                response_id=f"fake-response-{self.case['case_id']}-{self.step}",
                tool_calls=(call,),
            )

        return ProviderTurn(
            response_id=f"fake-response-{self.case['case_id']}-final",
            output_text=self._answer(input_items),
        )

    def _arguments_for(self, tool_name, input_items):
        constraints = self.case["constraints"]
        if tool_name == "search_catalog":
            return {
                "query": self.case["prompt"],
                "category": constraints.get("category"),
                "keyword": constraints.get("keyword"),
                "min_price": constraints.get("min_price"),
                "max_price": constraints.get("max_price"),
                "limit": 50,
            }
        previous = json.loads(input_items[-1]["output"])
        if tool_name == "rank_candidates_for_user":
            match = re.search(r"U\d+", self.case["prompt"])
            return {
                "user_id": match.group(0) if match else "U99999",
                "query": self.case["prompt"],
                "candidate_item_ids": [row["item_id"] for row in previous["candidates"]],
                "limit": 20,
            }
        if tool_name == "get_book_details":
            ranked = previous.get("ranked_candidates", previous.get("candidates", []))
            count = int(self.case.get("min_recommendations", 1))
            return {"item_ids": [row["item_id"] for row in ranked[:count]]}
        raise AssertionError(f"Unexpected tool in case: {tool_name}")

    def _answer(self, input_items):
        outcome = self.case["expected_outcome"]
        if outcome == "clarify":
            if self.case["case_id"] == "clarify_missing_rating":
                return "目录没有星级评分数据，不能核验四星条件。请补充类别或关键词等筛选条件吗？"
            return "请提供你的用户编号，我再结合阅读历史排序推荐。"
        previous = json.loads(input_items[-1]["output"])
        if outcome == "no_results" or not previous.get("items"):
            return "没有找到符合条件的图书。"
        parts = []
        for item in previous["items"]:
            categories = item.get("item_categories", [])
            keywords = item.get("item_keywords", [])
            price = item.get("price")
            reason = "、".join(categories + keywords)
            parts.append(f"推荐《{item['name']}》[{item['item_id']}]。目录标签为{reason}，价格为{price}元。")
        if self.case.get("expect_no_personalization_claim"):
            parts.insert(0, "该用户没有可用的历史或画像，以下按检索相关度排序。")
        return " ".join(parts)


def test_all_25_cases_pass_through_offline_agent_tool_flow(tmp_path):
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    data = load_demo_data(project_root / "data")
    cases = load_cases(project_root / "data" / "agent_eval_cases.jsonl")
    catalog = BookCatalog(data.items)
    recommender = RecommendationEngine(catalog, data.users).fit(data.interactions)
    provider = CaseDrivenFakeProvider(cases)
    agent = BookAgent(
        BookTools(catalog, recommender),
        provider,
        model_name=provider.model_name,
    )

    report = run_agent_eval(agent, cases, catalog, output_path=tmp_path / "offline-eval.json")

    assert report["summary"]["case_count"] == 25
    assert report["summary"]["end_to_end_success_rate"] == 1.0
