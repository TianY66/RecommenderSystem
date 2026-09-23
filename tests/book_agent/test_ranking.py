import pandas as pd

from book_agent.catalog import BookCatalog
from book_agent.ranking import RecommendationEngine


ITEMS = {
    "I1": {
        "item_id": "I1",
        "name": "算法基础",
        "item_categories": ("科技",),
        "item_keywords": ("算法", "编程"),
        "description": "算法入门",
        "price": 30.0,
    },
    "I2": {
        "item_id": "I2",
        "name": "算法实践",
        "item_categories": ("科技",),
        "item_keywords": ("算法", "编程"),
        "description": "算法实践练习",
        "price": 45.0,
    },
    "I3": {
        "item_id": "I3",
        "name": "生活随笔",
        "item_categories": ("文学",),
        "item_keywords": ("随笔",),
        "description": "生活故事",
        "price": 25.0,
    },
}
USERS = {
    "U1": {
        "user_id": "U1",
        "user_categories": ("科技",),
        "user_keywords": ("算法", "编程"),
    }
}


def make_engine():
    catalog = BookCatalog(ITEMS)
    interactions = pd.DataFrame(
        [
            ("U1", "I1", 1.0),
            ("U2", "I1", 1.0),
            ("U2", "I2", 1.0),
            ("U3", "I2", 1.0),
            ("U3", "I3", 1.0),
        ],
        columns=["user_id", "item_id", "rating"],
    )
    engine = RecommendationEngine(catalog, USERS)
    engine.fit(interactions)
    return engine


def test_personalized_ranking_only_reorders_candidates_and_explains_fit():
    engine = make_engine()

    ranked = engine.rank_candidates_for_user("U1", "算法", ["I1", "I3", "I2"])

    assert [row["item_id"] for row in ranked] == ["I2", "I3"]
    assert ranked[0]["personalized_score"] > ranked[1]["personalized_score"]
    assert ranked[0]["history_score"] > 0
    assert "category:科技" in ranked[0]["reason_signals"]
    assert "keyword:算法" in ranked[0]["reason_signals"]


def test_unknown_user_falls_back_to_query_score_and_known_items():
    engine = make_engine()

    ranked = engine.rank_candidates_for_user("missing", "算法", ["I2", "bad-id", "I3"])

    assert {row["item_id"] for row in ranked} == {"I2", "I3"}
    assert all(row["history_score"] == 0 for row in ranked)


def test_popularity_and_itemcf_baselines_filter_seen_items():
    engine = make_engine()

    assert "I1" not in engine.recommend_popularity("U1", limit=3)
    assert engine.recommend_itemcf("U1", limit=2)[0] == "I2"
    assert "I1" not in engine.recommend_itemcf("U1", limit=3)
