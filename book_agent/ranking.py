"""Popularity, item-based collaborative filtering, and candidate personalization."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .catalog import BookCatalog, _values


def _normalized(scores: Mapping[str, float]) -> dict[str, float]:
    maximum = max(scores.values(), default=0.0)
    if maximum <= 0:
        return {item_id: 0.0 for item_id in scores}
    return {item_id: max(0.0, score / maximum) for item_id, score in scores.items()}


def _overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class RecommendationEngine:
    """Fit only on supplied interactions and rerank only caller-provided candidates."""

    def __init__(
        self,
        catalog: BookCatalog,
        users: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        max_neighbors: int = 100,
    ) -> None:
        self.catalog = catalog
        self.users = {str(user_id): dict(user) for user_id, user in (users or {}).items()}
        self.max_neighbors = max_neighbors
        self.user_history: dict[str, dict[str, float]] = {}
        self.popularity: Counter[str] = Counter()
        self.item_neighbors: dict[str, dict[str, float]] = {}
        self._popularity_order_cache: tuple[str, ...] | None = None

    def fit(self, interactions: pd.DataFrame) -> "RecommendationEngine":
        self._popularity_order_cache = None
        required = {"user_id", "item_id"}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"interactions missing columns: {sorted(missing)}")

        frame = interactions.copy()
        frame["user_id"] = frame["user_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)
        if "rating" in frame:
            strengths = pd.to_numeric(frame["rating"], errors="coerce").fillna(1.0).clip(lower=0.0)
        else:
            strengths = pd.Series(1.0, index=frame.index)
        frame["_strength"] = strengths
        frame = frame[frame["item_id"].isin(self.catalog.items)]
        grouped = frame.groupby(["user_id", "item_id"], sort=True)["_strength"].sum()

        self.user_history = defaultdict(dict)
        self.popularity = Counter()
        for (user_id, item_id), strength in grouped.items():
            if strength <= 0:
                continue
            self.user_history[str(user_id)][str(item_id)] = float(strength)
            self.popularity[str(item_id)] += 1

        item_norm_sq: Counter[str] = Counter()
        cooccurrence: dict[str, Counter[str]] = defaultdict(Counter)
        for history in self.user_history.values():
            item_strengths = sorted(history.items())
            for item_id, strength in item_strengths:
                item_norm_sq[item_id] += strength * strength
            for index, (left_id, left_strength) in enumerate(item_strengths):
                for right_id, right_strength in item_strengths[index + 1 :]:
                    cooccurrence[left_id][right_id] += left_strength * right_strength
                    cooccurrence[right_id][left_id] += left_strength * right_strength

        self.item_neighbors = {}
        for item_id, neighbors in cooccurrence.items():
            scored = []
            for neighbor_id, dot in neighbors.items():
                denominator = math.sqrt(item_norm_sq[item_id] * item_norm_sq[neighbor_id])
                if denominator:
                    scored.append((neighbor_id, dot / denominator))
            scored.sort(key=lambda pair: (-pair[1], pair[0]))
            self.item_neighbors[item_id] = dict(scored[: self.max_neighbors])
        self._popularity_order_cache = self._build_popularity_order()
        return self

    def _build_popularity_order(self) -> tuple[str, ...]:
        scored = sorted(self.popularity.items(), key=lambda pair: (-pair[1], pair[0]))
        observed = [item_id for item_id, _ in scored]
        missing = [item_id for item_id in self.catalog.item_ids if item_id not in self.popularity]
        return tuple(observed + missing)

    def _popularity_order(self) -> tuple[str, ...]:
        if self._popularity_order_cache is None:
            self._popularity_order_cache = self._build_popularity_order()
        return self._popularity_order_cache

    def recommend_popularity(self, user_id: str | None, *, limit: int = 20) -> list[str]:
        if limit <= 0:
            raise ValueError("limit must be a positive integer")
        seen = set(self.user_history.get(str(user_id), {})) if user_id is not None else set()
        return [item_id for item_id in self._popularity_order() if item_id not in seen][:limit]

    def _itemcf_scores(self, user_id: str) -> dict[str, float]:
        history = self.user_history.get(user_id, {})
        scores: dict[str, float] = defaultdict(float)
        for history_item, strength in history.items():
            for candidate_id, similarity in self.item_neighbors.get(history_item, {}).items():
                if candidate_id not in history:
                    scores[candidate_id] += strength * similarity
        return dict(scores)

    def recommend_itemcf(self, user_id: str | None, *, limit: int = 20) -> list[str]:
        if limit <= 0:
            raise ValueError("limit must be a positive integer")
        normalized_user_id = str(user_id) if user_id is not None else ""
        history = self.user_history.get(normalized_user_id, {})
        seen = set(history)
        scores = self._itemcf_scores(normalized_user_id)
        ordered = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
        ordered.extend(
            item_id
            for item_id in self._popularity_order()
            if item_id not in seen and item_id not in scores
        )
        return ordered[:limit]

    def _user_preferences(self, user_id: str) -> tuple[set[str], set[str]]:
        profile = self.users.get(user_id, {})
        categories = {value.casefold() for value in _values(profile.get("user_categories"))}
        keywords = {value.casefold() for value in _values(profile.get("user_keywords"))}
        return categories, keywords

    def _history_preferences(self, user_id: str) -> tuple[set[str], set[str]]:
        history = self.user_history.get(user_id, {})
        categories: set[str] = set()
        keywords: set[str] = set()
        for item_id in history:
            item = self.catalog.items[item_id]
            categories.update(value.casefold() for value in _values(item.get("item_categories")))
            keywords.update(value.casefold() for value in _values(item.get("item_keywords")))
        return categories, keywords

    def rank_candidates_for_user(
        self,
        user_id: str,
        query: str,
        candidate_item_ids: Sequence[str],
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be a positive integer")
        normalized_user_id = str(user_id)
        history = self.user_history.get(normalized_user_id, {})
        candidates = [
            item_id
            for item_id in dict.fromkeys(str(value) for value in candidate_item_ids)
            if item_id in self.catalog.items and item_id not in history
        ]
        if not candidates:
            return []

        query_scores = _normalized(self.catalog.query_scores(query, candidates))
        raw_cf_scores = self._itemcf_scores(normalized_user_id)
        cf_scores = _normalized({item_id: raw_cf_scores.get(item_id, 0.0) for item_id in candidates})
        profile_categories, profile_keywords = self._user_preferences(normalized_user_id)
        history_categories, history_keywords = self._history_preferences(normalized_user_id)

        results: list[dict[str, Any]] = []
        for item_id in candidates:
            item = self.catalog.items[item_id]
            item_categories = {value.casefold() for value in _values(item.get("item_categories"))}
            item_keywords = {value.casefold() for value in _values(item.get("item_keywords"))}
            profile_parts = []
            if profile_categories:
                profile_parts.append(_overlap_score(profile_categories, item_categories))
            if profile_keywords:
                profile_parts.append(_overlap_score(profile_keywords, item_keywords))
            profile_score = sum(profile_parts) / len(profile_parts) if profile_parts else 0.0

            history_parts = []
            if history_categories:
                history_parts.append(_overlap_score(history_categories, item_categories))
            if history_keywords:
                history_parts.append(_overlap_score(history_keywords, item_keywords))
            content_history_score = sum(history_parts) / len(history_parts) if history_parts else 0.0
            history_score = 0.7 * cf_scores[item_id] + 0.3 * content_history_score

            has_query = bool(query and query.strip())
            has_preference = bool(history or profile_categories or profile_keywords)
            if has_query and has_preference:
                personalized_score = (
                    0.50 * query_scores[item_id]
                    + 0.35 * history_score
                    + 0.15 * profile_score
                )
            elif has_query:
                personalized_score = query_scores[item_id]
            elif has_preference:
                personalized_score = 0.70 * history_score + 0.30 * profile_score
            else:
                personalized_score = 0.0

            reason_signals: list[str] = []
            if query_scores[item_id] > 0:
                reason_signals.append("query_match")
            reason_signals.extend(
                f"category:{value}"
                for value in sorted(item_categories & (profile_categories | history_categories))
            )
            reason_signals.extend(
                f"keyword:{value}"
                for value in sorted(item_keywords & (profile_keywords | history_keywords))
            )
            if cf_scores[item_id] > 0:
                reason_signals.append("co_read_similarity")

            results.append(
                {
                    "item_id": item_id,
                    "query_score": round(query_scores[item_id], 6),
                    "history_score": round(history_score, 6),
                    "profile_score": round(profile_score, 6),
                    "personalized_score": round(personalized_score, 6),
                    "reason_signals": reason_signals,
                }
            )

        results.sort(key=lambda row: (-row["personalized_score"], row["item_id"]))
        return results[:limit]
