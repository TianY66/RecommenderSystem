"""Validate recommendation statements against verified catalog details."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_CATEGORY_CLAIM = re.compile(
    r"(?:属于|类别\s*(?:是|为|：|:)|分类\s*(?:是|为|：|:))\s*([^，,。；;\n]+)"
)
_KEYWORD_CLAIM = re.compile(
    r"(?:关键词|关键字|标签)(?:包括|包含|含有|含|有|是|为|：|:)?\s*([^，,。；;\n]+)"
)
_PRICE_CLAIM = re.compile(
    r"(?:价格|售价|定价)(?:为|是|约|：|:)?\s*[￥¥]?\s*(\d+(?:\.\d+)?)\s*元"
)
_QUOTED_VALUE = re.compile(r"[\"“‘「『]([^\"“”‘’「」『』]+)[\"”’」』]")
_CLAUSE_END = re.compile(r"[，,。；;\n]")


def field_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = value.split(";")
    elif isinstance(value, (tuple, list, set)):
        values = value
    else:
        values = (value,)
    return tuple(str(item).strip() for item in values if str(item).strip())


def answer_facts_supported(
    answer: str,
    cited_item_ids: list[str],
    verified_items: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Check explicit category, keyword, and price claims against cited items."""

    items = [verified_items.get(item_id) for item_id in cited_item_ids]
    if not cited_item_ids or any(item is None for item in items):
        return not cited_item_ids

    known_tags: set[str] = set()
    known_prices: set[float] = set()
    for item in items:
        assert item is not None
        known_tags.update(
            value.casefold()
            for field in ("item_categories", "item_keywords")
            for value in field_values(item.get(field))
        )
        price = _number(item.get("price"))
        if price is not None:
            known_prices.add(price)

    for claim in _CATEGORY_CLAIM.findall(answer):
        values = _quoted_values(claim) or _split_labels(claim)
        if any(value not in known_tags for value in values):
            return False

    for claim in _KEYWORD_CLAIM.findall(answer):
        if claim.lstrip().startswith(("依据", "/", "|")):
            continue
        values = _quoted_values(claim) or _keyword_labels(claim)
        if any(value not in known_tags for value in values):
            return False

    for claim in _PRICE_CLAIM.findall(answer):
        price = float(claim)
        if not any(abs(price - known) < 1e-6 for known in known_prices):
            return False
    return True


def _quoted_values(claim: str) -> list[str]:
    return [value for raw in _QUOTED_VALUE.findall(claim) if (value := _normalize_label(raw))]


def _keyword_labels(claim: str) -> list[str]:
    claim = re.split(
        r"(?:价格|售价|定价|作者|简介|描述|适合|推荐|类别|分类|因此|其中)",
        claim,
        maxsplit=1,
    )[0]
    claim = re.sub(r"^(?:包括|包含|含有|含|有|是|为|：|:)\s*", "", claim.strip())
    values = re.split(r"[、/]|和|与|及", claim)
    return [value for raw in values if (value := _normalize_label(raw))]


def _split_labels(claim: str) -> list[str]:
    claim = _CLAUSE_END.split(claim, maxsplit=1)[0]
    values = re.split(r"[、/]|和|与|及", claim)
    return [value for raw in values if (value := _normalize_label(raw))]


def _normalize_label(value: str) -> str:
    normalized = value.strip(" \t\r\n\"'“”‘’「」『』【】[]()（）`*_:-：")
    normalized = normalized.removesuffix("类")
    return normalized.casefold()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None
