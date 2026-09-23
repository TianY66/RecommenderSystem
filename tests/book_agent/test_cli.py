import pandas as pd
import pytest

from book_agent.cli import evaluate_recommenders, main
from book_agent.data import DemoData


def test_recommender_evaluation_fits_only_temporal_training_data():
    data = DemoData(
        items={
            item_id: {
                "item_id": item_id,
                "name": item_id,
                "item_categories": ("科技",),
                "item_keywords": ("算法",),
            }
            for item_id in ("I1", "I2", "I3")
        },
        users={},
        interactions=pd.DataFrame(
            [
                ("U1", "I1", 1, "2024-01-01T00:00:00Z"),
                ("U1", "I2", 1, "2024-01-02T00:00:00Z"),
                ("U2", "I1", 1, "2024-01-01T00:00:00Z"),
                ("U2", "I3", 1, "2024-01-03T00:00:00Z"),
            ],
            columns=["user_id", "item_id", "rating", "event_time"],
        ),
    )

    report = evaluate_recommenders(data, ks=(1, 2))

    assert report["split"]["test_users"] == 2
    assert report["split"]["train_events"] == 2
    assert set(report["models"]) == {"popularity", "itemcf"}
    assert report["models"]["itemcf"]["users"] == 2
    assert report["models"]["itemcf"]["recall@2"] == 1.0


def test_real_agent_cli_reports_missing_credentials_without_stack_trace(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(SystemExit) as exit_info:
        main(["evaluate-agent"])

    output = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "OPENAI_API_KEY" in output.err
    assert "Traceback" not in output.err


def test_real_agent_cli_selects_deepseek_and_reports_missing_key(monkeypatch, capsys):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exit_info:
        main(["evaluate-agent", "--provider", "deepseek"])

    output = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "DEEPSEEK_API_KEY" in output.err
    assert "Traceback" not in output.err
