"""Ranking metrics with explicit empty-result behavior."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set
from typing import Any


def ranking_metrics(
    ranked_item_ids: Sequence[str],
    relevant_item_ids: Set[str],
    k: int,
) -> dict[str, float]:
    if k <= 0:
        raise ValueError("k must be a positive integer")

    relevant = set(relevant_item_ids)
    if not relevant:
        return {f"recall@{k}": 0.0, f"ndcg@{k}": 0.0, f"mrr@{k}": 0.0}

    top_k: list[str] = []
    seen: set[str] = set()
    for item_id in ranked_item_ids:
        if item_id not in seen:
            top_k.append(item_id)
            seen.add(item_id)
        if len(top_k) == k:
            break

    hits = [index for index, item_id in enumerate(top_k, start=1) if item_id in relevant]
    recall = len(hits) / len(relevant)
    dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    reciprocal_rank = 1.0 / hits[0] if hits else 0.0
    return {
        f"recall@{k}": recall,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
        f"mrr@{k}": reciprocal_rank,
    }


def aggregate_ranking_metrics(
    rankings: Mapping[str, Sequence[str]],
    relevant_items: Mapping[str, Set[str]],
    ks: Sequence[int] = (10, 20),
) -> dict[str, Any]:
    evaluable_users = [
        user_id for user_id, relevant in relevant_items.items() if relevant
    ]
    result: dict[str, Any] = {"users": len(evaluable_users)}
    for k in ks:
        if k <= 0:
            raise ValueError("all k values must be positive integers")
        metric_names = (f"recall@{k}", f"ndcg@{k}", f"mrr@{k}")
        totals = dict.fromkeys(metric_names, 0.0)
        for user_id in evaluable_users:
            values = ranking_metrics(rankings.get(user_id, ()), relevant_items[user_id], k)
            for metric_name in metric_names:
                totals[metric_name] += values[metric_name]
        for metric_name, total in totals.items():
            result[metric_name] = total / len(evaluable_users) if evaluable_users else 0.0
    return result
