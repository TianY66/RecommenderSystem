"""Offline text retrieval over the local book catalog."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer


def _values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = value.split(";")
    elif isinstance(value, (tuple, list, set)):
        values = value
    else:
        try:
            import pandas as pd

            if pd.isna(value):
                return ()
        except (ImportError, TypeError, ValueError):
            pass
        values = (value,)
    return tuple(str(item).strip() for item in values if str(item).strip())


def _text(item: Mapping[str, Any]) -> str:
    fields = (
        "name",
        "author",
        "item_categories",
        "item_keywords",
        "description",
    )
    return " ".join(
        value
        for field in fields
        for value in _values(item.get(field))
    )


class BookCatalog:
    """Build a local character n-gram index and expose filtered search."""

    def __init__(self, items: Mapping[str, Mapping[str, Any]]) -> None:
        self.items = {str(item_id): dict(item) for item_id, item in items.items()}
        self.item_ids = tuple(sorted(self.items))
        documents = [_text(self.items[item_id]) for item_id in self.item_ids]
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        if documents and any(document.strip() for document in documents):
            self.vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 4),
                lowercase=True,
                sublinear_tf=True,
            )
            self.matrix = self.vectorizer.fit_transform(documents)
        self._id_to_row = {item_id: index for index, item_id in enumerate(self.item_ids)}

    def get_item(self, item_id: str) -> dict[str, Any] | None:
        item = self.items.get(str(item_id))
        return dict(item) if item is not None else None

    def query_scores(self, query: str, item_ids: list[str] | tuple[str, ...]) -> dict[str, float]:
        valid_ids = list(dict.fromkeys(str(item_id) for item_id in item_ids if str(item_id) in self.items))
        if not query or not query.strip() or self.vectorizer is None or self.matrix is None:
            return {item_id: 0.0 for item_id in valid_ids}

        query_vector = self.vectorizer.transform([query.strip()])
        if query_vector.nnz == 0:
            return {item_id: 0.0 for item_id in valid_ids}

        similarities = (self.matrix @ query_vector.T).toarray().reshape(-1)
        return {
            item_id: float(similarities[self._id_to_row[item_id]])
            for item_id in valid_ids
        }

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        keyword: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be a positive integer")
        if min_price is not None and min_price < 0:
            raise ValueError("min_price must be non-negative")
        if max_price is not None and max_price < 0:
            raise ValueError("max_price must be non-negative")
        if min_price is not None and max_price is not None and min_price > max_price:
            raise ValueError("min_price cannot exceed max_price")

        normalized_category = category.casefold() if category else None
        normalized_keyword = keyword.casefold() if keyword else None
        candidates: list[str] = []
        for item_id in self.item_ids:
            item = self.items[item_id]
            categories = {value.casefold() for value in _values(item.get("item_categories"))}
            keywords = {value.casefold() for value in _values(item.get("item_keywords"))}
            if normalized_category and normalized_category not in categories:
                continue
            if normalized_keyword and normalized_keyword not in keywords:
                continue

            price_value = item.get("price")
            try:
                price = float(price_value) if price_value is not None else None
            except (TypeError, ValueError):
                price = None
            if (min_price is not None or max_price is not None) and price is None:
                continue
            if min_price is not None and price is not None and price < min_price:
                continue
            if max_price is not None and price is not None and price > max_price:
                continue
            candidates.append(item_id)

        scores = self.query_scores(query, candidates)
        if query and query.strip():
            candidates = [item_id for item_id in candidates if scores[item_id] > 0]
        ordered = sorted(candidates, key=lambda item_id: (-scores[item_id], item_id))
        return [
            {"item_id": item_id, "retrieval_score": scores[item_id]}
            for item_id in ordered[:limit]
        ]
