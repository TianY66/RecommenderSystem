"""Strict, read-only tool contracts for the book agent."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .catalog import BookCatalog
from .ranking import RecommendationEngine


class ToolValidationError(ValueError):
    """A model-provided tool call failed schema or workflow validation."""


_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Book topic or query text."},
        "category": {"type": ["string", "null"]},
        "keyword": {"type": ["string", "null"]},
        "min_price": {"type": ["number", "null"]},
        "max_price": {"type": ["number", "null"]},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
    },
    "required": ["query", "category", "keyword", "min_price", "max_price", "limit"],
    "additionalProperties": False,
}
_RANK_SCHEMA = {
    "type": "object",
    "properties": {
        "user_id": {"type": "string"},
        "query": {"type": "string"},
        "candidate_item_ids": {"type": "array", "items": {"type": "string"}},
        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
    },
    "required": ["user_id", "query", "candidate_item_ids", "limit"],
    "additionalProperties": False,
}
_DETAILS_SCHEMA = {
    "type": "object",
    "properties": {
        "item_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["item_ids"],
    "additionalProperties": False,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def _exact_arguments(arguments: Any, expected_keys: set[str]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ToolValidationError("Tool arguments must be a JSON object")
    missing = expected_keys - set(arguments)
    extra = set(arguments) - expected_keys
    if missing:
        raise ToolValidationError(f"Missing required arguments: {', '.join(sorted(missing))}")
    if extra:
        raise ToolValidationError(f"Unexpected arguments: {', '.join(sorted(extra))}")
    return arguments


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ToolValidationError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 256:
        raise ToolValidationError(f"{name} must be a string of at most 256 characters or null")
    return value.strip() or None


class BookTools:
    def __init__(self, catalog: BookCatalog, recommender: RecommendationEngine) -> None:
        self.catalog = catalog
        self.recommender = recommender

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "search_catalog",
                "description": (
                    "Retrieve book candidates by topic and exact catalog filters. "
                    "Returns only item IDs and retrieval scores; use the returned IDs "
                    "for personalization and detail verification."
                ),
                "parameters": _SEARCH_SCHEMA,
                "strict": True,
            },
            {
                "type": "function",
                "name": "rank_candidates_for_user",
                "description": (
                    "Rerank IDs returned by the immediately preceding search_catalog call "
                    "using this user's interaction history and profile. The response states "
                    "whether user context was available. Never add candidates."
                ),
                "parameters": _RANK_SCHEMA,
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_book_details",
                "description": (
                    "Verify metadata for IDs returned by a prior search or ranking call. "
                    "Use only returned fields as factual evidence; the catalog has no star ratings."
                ),
                "parameters": _DETAILS_SCHEMA,
                "strict": True,
            },
        ]

    def new_session(self) -> "BookToolSession":
        return BookToolSession(self)


class BookToolSession:
    """Track candidate provenance for one Agent run without shared mutable state."""

    def __init__(self, tools: BookTools) -> None:
        self.tools = tools
        self.search_candidate_ids: set[str] = set()
        self.detail_allowed_ids: set[str] = set()

    def call(self, tool_name: str, arguments: Any) -> dict[str, Any]:
        if tool_name == "search_catalog":
            return self._search(arguments)
        if tool_name == "rank_candidates_for_user":
            return self._rank(arguments)
        if tool_name == "get_book_details":
            return self._details(arguments)
        raise ToolValidationError(f"Unknown tool: {tool_name}")

    def _search(self, arguments: Any) -> dict[str, Any]:
        values = _exact_arguments(
            arguments,
            {"query", "category", "keyword", "min_price", "max_price", "limit"},
        )
        query = values["query"]
        if not isinstance(query, str) or len(query) > 2000:
            raise ToolValidationError("query must be a string of at most 2000 characters")
        category = _optional_text(values["category"], "category")
        keyword = _optional_text(values["keyword"], "keyword")
        min_price = self._optional_price(values["min_price"], "min_price")
        max_price = self._optional_price(values["max_price"], "max_price")
        limit = _bounded_int(values["limit"], "limit", 1, 50)
        candidates = self.tools.catalog.search(
            query,
            category=category,
            keyword=keyword,
            min_price=min_price,
            max_price=max_price,
            limit=limit,
        )
        self.search_candidate_ids = {row["item_id"] for row in candidates}
        self.detail_allowed_ids = set(self.search_candidate_ids)
        return {"candidates": candidates, "count": len(candidates)}

    def _rank(self, arguments: Any) -> dict[str, Any]:
        values = _exact_arguments(
            arguments,
            {"user_id", "query", "candidate_item_ids", "limit"},
        )
        user_id, query = values["user_id"], values["query"]
        if not isinstance(user_id, str) or not user_id.strip() or len(user_id) > 128:
            raise ToolValidationError("user_id must be a non-empty string of at most 128 characters")
        if not isinstance(query, str) or len(query) > 2000:
            raise ToolValidationError("query must be a string of at most 2000 characters")
        candidate_ids = values["candidate_item_ids"]
        if not isinstance(candidate_ids, list) or len(candidate_ids) > 100:
            raise ToolValidationError("candidate_item_ids must be a list of at most 100 IDs")
        if any(not isinstance(item_id, str) or not item_id.strip() for item_id in candidate_ids):
            raise ToolValidationError("candidate_item_ids must contain non-empty strings")
        requested_ids = set(candidate_ids)
        outside_search = requested_ids - self.search_candidate_ids
        if outside_search:
            raise ToolValidationError("candidate_item_ids must come from search candidates")
        limit = _bounded_int(values["limit"], "limit", 1, 20)
        ranked = self.tools.recommender.rank_candidates_for_user(
            user_id.strip(), query, candidate_ids, limit=limit
        )
        self.detail_allowed_ids = {row["item_id"] for row in ranked}
        history_count = len(self.tools.recommender.user_history.get(user_id.strip(), {}))
        profile_available = user_id.strip() in self.tools.recommender.users
        profile_categories, profile_keywords = self.tools.recommender._user_preferences(
            user_id.strip()
        )
        return {
            "ranked_candidates": ranked,
            "count": len(ranked),
            "user_context": {
                "known_user": profile_available or history_count > 0,
                "profile_available": profile_available,
                "history_item_count": history_count,
                "personalization_applied": bool(
                    history_count or profile_categories or profile_keywords
                ),
            },
        }

    def _details(self, arguments: Any) -> dict[str, Any]:
        values = _exact_arguments(arguments, {"item_ids"})
        item_ids = values["item_ids"]
        if not isinstance(item_ids, list) or not 1 <= len(item_ids) <= 20:
            raise ToolValidationError("item_ids must contain between 1 and 20 IDs")
        if any(not isinstance(item_id, str) or not item_id.strip() for item_id in item_ids):
            raise ToolValidationError("item_ids must contain non-empty strings")

        normalized_ids = list(dict.fromkeys(item_id.strip() for item_id in item_ids))
        outside_search = {
            item_id
            for item_id in normalized_ids
            if item_id in self.tools.catalog.items and item_id not in self.detail_allowed_ids
        }
        if outside_search:
            raise ToolValidationError("item_ids must come from search candidates")

        answer_fields = (
            "item_id",
            "name",
            "author",
            "item_categories",
            "item_keywords",
            "description",
            "price",
        )
        found = []
        for item_id in normalized_ids:
            item = self.tools.catalog.get_item(item_id)
            if item is not None:
                found.append(
                    _jsonable({key: item[key] for key in answer_fields if key in item})
                )
        not_found = [item_id for item_id in normalized_ids if item_id not in self.tools.catalog.items]
        return {"items": found, "not_found": not_found}

    @staticmethod
    def _optional_price(value: Any, name: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolValidationError(f"{name} must be a non-negative number or null")
        price = float(value)
        if not math.isfinite(price) or price < 0:
            raise ToolValidationError(f"{name} must be a non-negative finite number or null")
        return price
