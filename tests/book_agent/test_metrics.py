import math

from book_agent.metrics import aggregate_ranking_metrics, ranking_metrics


def test_ranking_metrics_match_hand_calculated_single_relevant_item():
    metrics = ranking_metrics(["I1", "I2", "I3"], {"I2"}, k=2)

    assert metrics["recall@2"] == 1.0
    assert metrics["mrr@2"] == 0.5
    assert math.isclose(metrics["ndcg@2"], 1 / math.log2(3))


def test_ranking_metrics_return_zero_when_no_relevant_items_or_results():
    assert ranking_metrics([], {"I1"}, k=5) == {
        "recall@5": 0.0,
        "ndcg@5": 0.0,
        "mrr@5": 0.0,
    }
    assert ranking_metrics(["I1"], set(), k=5) == {
        "recall@5": 0.0,
        "ndcg@5": 0.0,
        "mrr@5": 0.0,
    }


def test_aggregate_ranking_metrics_keeps_empty_recommendations_as_zero():
    result = aggregate_ranking_metrics(
        rankings={"U1": ["I1", "I2"], "U2": [], "U3": ["I3"]},
        relevant_items={"U1": {"I2"}, "U2": {"I4"}, "U3": set()},
        ks=(2,),
    )

    assert result["users"] == 2
    assert result["recall@2"] == 0.5
    assert result["mrr@2"] == 0.25
