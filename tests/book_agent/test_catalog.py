import pytest

from book_agent.catalog import BookCatalog


@pytest.fixture
def catalog():
    return BookCatalog(
        {
            "I1": {
                "item_id": "I1",
                "name": "Python 数据分析入门",
                "author": "作者甲",
                "item_categories": ("科技",),
                "item_keywords": ("Python", "编程"),
                "description": "面向初学者的实践指南",
                "price": 49.0,
            },
            "I2": {
                "item_id": "I2",
                "name": "Python 深度学习",
                "author": "作者乙",
                "item_categories": ("科技",),
                "item_keywords": ("Python", "神经网络"),
                "description": "从基础概念到模型实践",
                "price": 120.0,
            },
            "I3": {
                "item_id": "I3",
                "name": "文学小书",
                "author": "作者丙",
                "item_categories": ("文学",),
                "item_keywords": ("随笔",),
                "description": "关于生活的短篇随笔",
                "price": 30.0,
            },
        }
    )


def test_search_uses_text_relevance_and_applies_metadata_filters(catalog):
    results = catalog.search("Python", category="科技", keyword="编程", max_price=100)

    assert [result["item_id"] for result in results] == ["I1"]
    assert results[0]["retrieval_score"] > 0


def test_empty_query_is_stable_and_limit_is_validated(catalog):
    results = catalog.search("", limit=2)

    assert [result["item_id"] for result in results] == ["I1", "I2"]
    with pytest.raises(ValueError, match="limit"):
        catalog.search("Python", limit=0)


def test_nonempty_query_with_no_text_match_returns_no_candidates(catalog):
    assert catalog.search("星际鲸鱼XYZ") == []


def test_search_rejects_invalid_price_ranges(catalog):
    with pytest.raises(ValueError, match="max_price"):
        catalog.search("Python", min_price=100, max_price=20)
