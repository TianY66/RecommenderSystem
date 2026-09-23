import pandas as pd
import pytest

from book_agent.catalog import BookCatalog
from book_agent.ranking import RecommendationEngine
from book_agent.tools import BookTools, ToolValidationError


@pytest.fixture
def tools():
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
    engine = RecommendationEngine(catalog, users).fit(interactions)
    return BookTools(catalog, engine)


def test_tool_definitions_are_strict_and_allow_only_three_read_tools(tools):
    definitions = tools.tool_definitions()

    assert {definition["name"] for definition in definitions} == {
        "search_catalog",
        "rank_candidates_for_user",
        "get_book_details",
    }
    assert all(definition["type"] == "function" for definition in definitions)
    assert all(definition["strict"] is True for definition in definitions)
    assert all(definition["parameters"]["additionalProperties"] is False for definition in definitions)
    category_description = next(
        definition["parameters"]["properties"]["category"]["description"]
        for definition in definitions
        if definition["name"] == "search_catalog"
    )
    assert "科技" in category_description
    assert "主题" in category_description


def test_search_treats_string_null_as_an_unset_optional_filter(tools):
    result = tools.new_session().call(
        "search_catalog",
        {
            "query": "随笔",
            "category": "null",
            "keyword": "随笔",
            "min_price": None,
            "max_price": None,
            "limit": 5,
        },
    )

    assert [row["item_id"] for row in result["candidates"]] == ["I3"]


def test_search_rank_details_flow_preserves_candidates_and_real_metadata(tools):
    session = tools.new_session()

    search = session.call(
        "search_catalog",
        {
            "query": "算法",
            "category": "科技",
            "keyword": "算法",
            "min_price": None,
            "max_price": 50,
            "limit": 10,
        },
    )
    candidate_ids = [item["item_id"] for item in search["candidates"]]
    ranked = session.call(
        "rank_candidates_for_user",
        {"user_id": "U1", "query": "算法", "candidate_item_ids": candidate_ids, "limit": 5},
    )
    selected_id = ranked["ranked_candidates"][0]["item_id"]
    details = session.call("get_book_details", {"item_ids": [selected_id]})

    assert candidate_ids == ["I1", "I2"]
    assert selected_id == "I2"
    assert details["items"][0]["name"] == "算法实践"
    assert details["items"][0]["price"] == 45.0
    assert details["items"][0]["item_categories"] == ["科技"]
    assert "image" not in details["items"][0]


def test_rank_and_details_reject_ids_not_returned_by_search(tools):
    session = tools.new_session()
    with pytest.raises(ToolValidationError, match="search candidates"):
        session.call(
            "rank_candidates_for_user",
            {"user_id": "U1", "query": "算法", "candidate_item_ids": ["I2"], "limit": 2},
        )

    session.call(
        "search_catalog",
        {
            "query": "算法",
            "category": "科技",
            "keyword": "算法",
            "min_price": None,
            "max_price": 50,
            "limit": 10,
        },
    )
    with pytest.raises(ToolValidationError, match="search candidates"):
        session.call("get_book_details", {"item_ids": ["I3"]})


def test_tool_arguments_are_validated_and_unknown_items_are_reported(tools):
    session = tools.new_session()
    with pytest.raises(ToolValidationError, match="Unexpected"):
        session.call(
            "search_catalog",
            {
                "query": "算法",
                "category": None,
                "keyword": None,
                "min_price": None,
                "max_price": None,
                "limit": 10,
                "sql": "DROP TABLE books",
            },
        )
    with pytest.raises(ToolValidationError, match="limit"):
        session.call(
            "search_catalog",
            {
                "query": "算法",
                "category": None,
                "keyword": None,
                "min_price": None,
                "max_price": None,
                "limit": 0,
            },
        )

    session.call(
        "search_catalog",
        {
            "query": "算法",
            "category": None,
            "keyword": None,
            "min_price": None,
            "max_price": None,
            "limit": 10,
        },
    )
    result = session.call("get_book_details", {"item_ids": ["I2", "I999"]})
    assert [item["item_id"] for item in result["items"]] == ["I2"]
    assert result["not_found"] == ["I999"]


def test_tool_session_requires_an_object_for_arguments(tools):
    session = tools.new_session()
    with pytest.raises(ToolValidationError, match="object"):
        session.call("search_catalog", "query: 算法")
    with pytest.raises(ToolValidationError, match="Unknown tool"):
        session.call("run_shell", {})


def test_rank_result_discloses_when_user_context_is_unavailable(tools):
    session = tools.new_session()
    session.call(
        "search_catalog",
        {
            "query": "算法",
            "category": None,
            "keyword": None,
            "min_price": None,
            "max_price": None,
            "limit": 10,
        },
    )

    ranked = session.call(
        "rank_candidates_for_user",
        {
            "user_id": "U404",
            "query": "算法",
            "candidate_item_ids": ["I1", "I2"],
            "limit": 5,
        },
    )

    assert ranked["user_context"] == {
        "known_user": False,
        "profile_available": False,
        "history_item_count": 0,
        "personalization_applied": False,
    }


def test_rank_distinguishes_known_user_from_matching_personalization_signals(tools):
    session = tools.new_session()
    session.call(
        "search_catalog",
        {
            "query": "随笔",
            "category": "文学",
            "keyword": "随笔",
            "min_price": None,
            "max_price": None,
            "limit": 5,
        },
    )

    ranked = session.call(
        "rank_candidates_for_user",
        {
            "user_id": "U1",
            "query": "随笔",
            "candidate_item_ids": ["I3"],
            "limit": 5,
        },
    )

    assert ranked["user_context"]["known_user"] is True
    assert ranked["user_context"]["history_item_count"] == 1
    assert ranked["user_context"]["personalization_applied"] is False
