"""Deterministic loading and temporal splitting for the local book dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DemoData:
    items: dict[str, dict[str, Any]]
    users: dict[str, dict[str, Any]]
    interactions: pd.DataFrame


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    test_targets: dict[str, str]
    skipped_user_count: int


def _split_tags(value: Any) -> tuple[str, ...]:
    if value is None or pd.isna(value):
        return ()
    if isinstance(value, (tuple, list)):
        values = value
    else:
        values = str(value).split(";")
    return tuple(tag.strip() for tag in values if tag and tag.strip())


def _records_by_id(
    frame: pd.DataFrame,
    id_column: str,
    tag_columns: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    frame = frame.loc[:, [column for column in frame.columns if not column.startswith("Unnamed:")]].copy()
    if id_column not in frame:
        raise ValueError(f"Missing required column: {id_column}")
    if frame[id_column].isna().any():
        raise ValueError(f"Column {id_column} contains an empty ID")

    frame[id_column] = frame[id_column].astype(str)
    if frame[id_column].duplicated().any():
        raise ValueError(f"Column {id_column} must contain unique IDs")

    result: dict[str, dict[str, Any]] = {}
    for record in frame.to_dict(orient="records"):
        record_id = str(record[id_column])
        clean_record: dict[str, Any] = {}
        for key, value in record.items():
            if key in tag_columns:
                clean_record[key] = _split_tags(value)
            elif value is None or (not isinstance(value, (tuple, list, dict)) and pd.isna(value)):
                clean_record[key] = None
            elif hasattr(value, "item"):
                clean_record[key] = value.item()
            else:
                clean_record[key] = value
        result[record_id] = clean_record
    return result


def load_demo_data(data_dir: str | Path) -> DemoData:
    """Load the three checked-in CSV files without network or image dependencies."""

    root = Path(data_dir)
    items_frame = pd.read_csv(root / "items_new.csv", encoding="utf-8-sig")
    users_frame = pd.read_csv(root / "users_new.csv", encoding="utf-8-sig")
    interactions = pd.read_csv(root / "interactions_new.csv", encoding="utf-8-sig")

    items = _records_by_id(items_frame, "item_id", ("item_categories", "item_keywords"))
    users = _records_by_id(users_frame, "user_id", ("user_categories", "user_keywords"))

    interactions = interactions.loc[
        :, [column for column in interactions.columns if not column.startswith("Unnamed:")]
    ].copy()
    if "event_time" not in interactions:
        if "datetime" not in interactions:
            raise ValueError("Interactions must include event_time or datetime")
        interactions = interactions.rename(columns={"datetime": "event_time"})

    for column in ("user_id", "item_id"):
        if column not in interactions:
            raise ValueError(f"Missing required column: {column}")
        if interactions[column].isna().any():
            raise ValueError(f"Column {column} contains an empty ID")
        interactions[column] = interactions[column].astype(str)

    interactions["event_time"] = pd.to_datetime(
        interactions["event_time"], errors="raise", utc=True
    )
    if "rating" not in interactions:
        interactions["rating"] = 1.0
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="raise").fillna(0.0)

    return DemoData(items=items, users=users, interactions=interactions)


def temporal_leave_last_out(interactions: pd.DataFrame) -> TemporalSplit:
    """Hold out each eligible user's latest distinct item and prevent pair leakage."""

    if interactions.empty:
        return TemporalSplit(train=interactions.copy(), test_targets={}, skipped_user_count=0)

    frame = interactions.copy().reset_index(drop=True)
    if "event_time" not in frame:
        if "datetime" not in frame:
            raise ValueError("Interactions must include event_time or datetime")
        frame = frame.rename(columns={"datetime": "event_time"})
    for column in ("user_id", "item_id"):
        if column not in frame:
            raise ValueError(f"Missing required column: {column}")
        if frame[column].isna().any():
            raise ValueError(f"Column {column} contains an empty ID")
        frame[column] = frame[column].astype(str)
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="raise", utc=True)
    frame["_input_order"] = range(len(frame))

    ordered = frame.sort_values(
        ["user_id", "event_time", "_input_order"], kind="mergesort"
    )
    distinct_pairs = ordered.drop_duplicates(["user_id", "item_id"], keep="last")
    counts = distinct_pairs.groupby("user_id", sort=True)["item_id"].size()
    eligible_users = set(counts[counts >= 2].index)
    skipped_user_count = int((counts < 2).sum())

    last_rows = distinct_pairs.groupby("user_id", sort=True).tail(1)
    held_out = last_rows[last_rows["user_id"].isin(eligible_users)]
    test_targets = {
        str(row.user_id): str(row.item_id)
        for row in held_out.itertuples(index=False)
    }
    train = distinct_pairs.loc[~distinct_pairs.index.isin(held_out.index)].drop(
        columns=["_input_order"]
    )
    train = train.sort_values(
        ["user_id", "event_time", "item_id"], kind="mergesort"
    ).reset_index(drop=True)
    return TemporalSplit(train=train, test_targets=test_targets, skipped_user_count=skipped_user_count)
