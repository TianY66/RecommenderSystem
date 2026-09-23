import json

import pandas as pd

from book_agent.agent import BookAgent
from book_agent.catalog import BookCatalog
from book_agent.providers import ProviderTurn, ToolCall
from book_agent.ranking import RecommendationEngine
from book_agent.tools import BookTools


class ScriptedProvider:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def create_response(self, *, input_items, tools, previous_response_id, instructions):
        self.requests.append(
            {
                "input_items": input_items,
                "tools": tools,
                "previous_response_id": previous_response_id,
                "instructions": instructions,
            }
        )
        return self.turns.pop(0)


def make_agent(turns, *, max_tool_rounds=6):
    items = {
        "I1": {
            "item_id": "I1",
            "name": "算法基础",
            "author": "作者甲",
            "item_categories": ("科技",),
            "item_keywords": ("算法", "编程"),
            "description": "算法入门",
            "price": 30.0,
        },
        "I2": {
            "item_id": "I2",
            "name": "算法实践",
            "author": "作者乙",
            "item_categories": ("科技",),
            "item_keywords": ("算法", "实践"),
            "description": "算法实践练习",
            "price": 45.0,
        },
        "I3": {
            "item_id": "I3",
            "name": "生活随笔",
            "author": "作者丙",
            "item_categories": ("文学",),
            "item_keywords": ("随笔",),
            "description": "生活故事",
            "price": 25.0,
        },
    }
    users = {"U1": {"user_categories": ("科技",), "user_keywords": ("算法",)}}
    interactions = pd.DataFrame(
        [("U1", "I1", 1), ("U2", "I1", 1), ("U2", "I2", 1)],
        columns=["user_id", "item_id", "rating"],
    )
    catalog = BookCatalog(items)
    recommender = RecommendationEngine(catalog, users).fit(interactions)
    provider = ScriptedProvider(turns)
    agent = BookAgent(BookTools(catalog, recommender), provider, model_name="test-model", max_tool_rounds=max_tool_rounds)
    return agent, provider


def call(call_id, name, arguments):
    return ToolCall(call_id=call_id, name=name, arguments=json.dumps(arguments, ensure_ascii=False))


def valid_flow_turns(final_answer="推荐《算法实践》[I2]。它属于科技类，关键词包括算法和实践，价格为45元。"):
    return [
        ProviderTurn(
            response_id="response-search",
            tool_calls=(
                call(
                    "call-search",
                    "search_catalog",
                    {
                        "query": "算法",
                        "category": "科技",
                        "keyword": "算法",
                        "min_price": None,
                        "max_price": 50,
                        "limit": 10,
                    },
                ),
            ),
        ),
        ProviderTurn(
            response_id="response-rank",
            tool_calls=(
                call(
                    "call-rank",
                    "rank_candidates_for_user",
                    {
                        "user_id": "U1",
                        "query": "算法",
                        "candidate_item_ids": ["I1", "I2"],
                        "limit": 5,
                    },
                ),
            ),
        ),
        ProviderTurn(
            response_id="response-details",
            tool_calls=(call("call-details", "get_book_details", {"item_ids": ["I2"]}),),
        ),
        ProviderTurn(response_id="response-final", output_text=final_answer),
    ]


def test_agent_runs_retrieval_personalization_details_and_grounded_answer():
    agent, provider = make_agent(valid_flow_turns())

    result = agent.run("结合 U1 的阅读历史，推荐一本 50 元以内的算法实践书，并说明依据。")

    assert result.success is True
    assert result.answer.startswith("推荐")
    assert [entry["name"] for entry in result.trace["tool_calls"]] == [
        "search_catalog",
        "rank_candidates_for_user",
        "get_book_details",
    ]
    assert result.trace["detail_item_ids"] == ["I2"]
    assert provider.requests[1]["previous_response_id"] == "response-search"
    assert provider.requests[1]["input_items"][0]["type"] == "function_call_output"
    assert provider.requests[1]["input_items"][0]["call_id"] == "call-search"
    assert [[tool["name"] for tool in request["tools"]] for request in provider.requests] == [
        ["search_catalog"],
        ["rank_candidates_for_user"],
        ["get_book_details"],
        [],
    ]


def test_agent_rejects_book_ids_not_verified_by_details():
    agent, _ = make_agent(valid_flow_turns("推荐《生活随笔》[I3]。"))

    result = agent.run("结合 U1 的阅读历史，推荐一本书。")

    assert result.success is False
    assert result.error == "answer_contains_unverified_item_id"
    assert "未核验" in result.answer


def test_agent_rejects_a_verified_id_paired_with_a_different_book_title():
    agent, _ = make_agent(valid_flow_turns("推荐《伪造书名》[I2]。它属于科技类，关键词有算法。"))

    result = agent.run("结合 U1 的阅读历史，推荐一本书。")

    assert result.success is False
    assert result.error == "answer_book_title_mismatch"


def test_agent_trace_keeps_minimal_verified_metadata_for_citations():
    agent, _ = make_agent(valid_flow_turns())

    result = agent.run("结合 U1 的阅读历史，推荐一本书。")

    assert result.trace["detail_evidence"]["I2"]["name"] == "算法实践"
    assert result.trace["detail_evidence"]["I2"]["item_keywords"] == ["算法", "实践"]


def test_agent_returns_clear_error_when_tool_round_limit_is_reached():
    repeated_search = call(
        "call-search",
        "search_catalog",
        {
            "query": "算法",
            "category": None,
            "keyword": None,
            "min_price": None,
            "max_price": None,
            "limit": 5,
        },
    )
    turns = [
        ProviderTurn(response_id=f"response-{index}", tool_calls=(repeated_search,))
        for index in range(3)
    ]
    agent, _ = make_agent(turns, max_tool_rounds=2)

    result = agent.run("搜索算法书")

    assert result.success is False
    assert result.error == "maximum_tool_rounds_reached"


def test_agent_sends_tool_errors_back_to_model_and_records_them():
    malformed = ToolCall(call_id="call-bad", name="search_catalog", arguments="{")
    agent, provider = make_agent(
        [
            ProviderTurn(response_id="response-error", tool_calls=(malformed,)),
            ProviderTurn(response_id="response-final", output_text="没有得到可核验的候选。"),
        ]
    )

    result = agent.run("搜索算法书")

    assert result.success is True
    assert result.trace["tool_calls"][0]["error"] == "invalid_json_arguments"
    tool_output = provider.requests[1]["input_items"][0]
    assert tool_output["type"] == "function_call_output"
    assert "JSON" in tool_output["output"]


def test_agent_returns_unknown_tool_error_to_provider():
    agent, provider = make_agent(
        [
            ProviderTurn(
                response_id="response-unknown-tool",
                tool_calls=(ToolCall(call_id="call-unknown", name="run_shell", arguments="{}"),),
            ),
            ProviderTurn(response_id="response-after-error", output_text="我无法执行该工具。"),
        ]
    )

    result = agent.run("搜索算法书")

    assert result.success is True
    assert result.trace["tool_calls"][0]["error"] == "unknown_tool"
    output = json.loads(provider.requests[1]["input_items"][0]["output"])
    assert output["error"] == "unknown_tool"
