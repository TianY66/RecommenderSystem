import pandas as pd

from book_agent.data import load_demo_data, temporal_leave_last_out


def test_load_demo_data_handles_utf8_bom_and_normalizes_fields(tmp_path):
    pd.DataFrame(
        [
            {
                "item_id": "I1",
                "name": "算法入门",
                "author": "作者甲",
                "item_categories": "科技;教育",
                "item_keywords": "算法;编程",
                "price": 49.5,
                "description": "面向初学者",
            }
        ]
    ).to_csv(tmp_path / "items_new.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "user_id": "U1",
                "user_categories": "科技;文学",
                "user_keywords": "编程;成长",
            }
        ]
    ).to_csv(tmp_path / "users_new.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "user_id": "U1",
                "item_id": "I1",
                "rating": 1,
                "datetime": "2025/3/24 11:13",
                "click": 1,
                "cart": 0,
                "forward": 0,
                "buy": 0,
            }
        ]
    ).to_csv(tmp_path / "interactions_new.csv", index=False, encoding="utf-8-sig")

    data = load_demo_data(tmp_path)

    assert data.items["I1"]["item_keywords"] == ("算法", "编程")
    assert data.users["U1"]["user_categories"] == ("科技", "文学")
    assert data.interactions.loc[0, "user_id"] == "U1"
    assert str(data.interactions.loc[0, "event_time"].tz) == "UTC"


def test_temporal_leave_last_out_deduplicates_pairs_without_leakage():
    interactions = pd.DataFrame(
        [
            ("U1", "I1", "2025-01-01T00:00:00Z"),
            ("U1", "I2", "2025-01-02T00:00:00Z"),
            ("U1", "I3", "2025-01-03T00:00:00Z"),
            ("U1", "I2", "2025-01-04T00:00:00Z"),
            ("U2", "I4", "2025-01-01T00:00:00Z"),
            ("U3", "I5", "2025-01-01T00:00:00Z"),
            ("U3", "I6", "2025-01-02T00:00:00Z"),
        ],
        columns=["user_id", "item_id", "event_time"],
    )
    interactions["event_time"] = pd.to_datetime(interactions["event_time"], utc=True)

    split = temporal_leave_last_out(interactions)

    assert split.test_targets == {"U1": "I2", "U3": "I6"}
    assert not ((split.train["user_id"] == "U1") & (split.train["item_id"] == "I2")).any()
    assert ((split.train["user_id"] == "U1") & (split.train["item_id"] == "I3")).any()
    assert ((split.train["user_id"] == "U2") & (split.train["item_id"] == "I4")).any()
    assert split.skipped_user_count == 1
