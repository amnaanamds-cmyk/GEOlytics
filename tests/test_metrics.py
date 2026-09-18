"""Ranking metrics, checked against hand-computed values."""

from __future__ import annotations

import math

import pytest

from geolytics.evaluation.metrics import (
    average_precision,
    evaluate_ranking,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from geolytics.evaluation.qrels import Query, Relevance

RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = frozenset({"b", "d", "f"})


class TestPrecision:
    def test_divides_by_k_not_by_retrieved(self):
        # Two relevant in the top 5 -> 0.4, and P@10 is 0.2 even though only
        # five results exist. Returning fewer results must not inflate scores.
        assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.4)
        assert precision_at_k(RANKED, RELEVANT, 10) == pytest.approx(0.2)

    def test_at_one(self):
        assert precision_at_k(RANKED, RELEVANT, 1) == 0.0
        assert precision_at_k(["b"], RELEVANT, 1) == 1.0

    def test_no_relevant_documents(self):
        assert precision_at_k(RANKED, frozenset(), 5) == 0.0

    def test_rejects_non_positive_k(self):
        with pytest.raises(ValueError):
            precision_at_k(RANKED, RELEVANT, 0)


class TestRecall:
    def test_counts_against_all_relevant(self):
        # 'f' was never retrieved, so recall is capped at 2/3.
        assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 3)

    def test_deeper_cutoff_cannot_decrease(self):
        values = [recall_at_k(RANKED, RELEVANT, k) for k in (1, 2, 3, 4, 5)]
        assert values == sorted(values)

    def test_perfect_recall(self):
        assert recall_at_k(["b", "d", "f"], RELEVANT, 3) == pytest.approx(1.0)


class TestMRR:
    def test_reciprocal_of_first_hit(self):
        assert mrr(RANKED, RELEVANT, 5) == pytest.approx(0.5)

    def test_zero_when_nothing_relevant_in_cutoff(self):
        assert mrr(RANKED, RELEVANT, 1) == 0.0

    def test_first_position(self):
        assert mrr(["b", "a"], RELEVANT, 2) == pytest.approx(1.0)


class TestAveragePrecision:
    def test_hand_computed(self):
        # Hits at ranks 2 and 4 -> (1/2 + 2/4) / 3 relevant total.
        expected = (1 / 2 + 2 / 4) / 3
        assert average_precision(RANKED, RELEVANT, 5) == pytest.approx(expected)

    def test_penalises_unretrieved_relevant(self):
        # Same ranking, but only two relevant items exist -> higher AP.
        smaller = average_precision(RANKED, frozenset({"b", "d"}), 5)
        assert smaller > average_precision(RANKED, RELEVANT, 5)


class TestNDCG:
    def test_binary_hand_computed(self):
        grades = {"b": 1, "d": 1, "f": 1}
        dcg = 1 / math.log2(3) + 1 / math.log2(5)
        idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        assert ndcg_at_k(RANKED, grades, 5) == pytest.approx(dcg / idcg)

    def test_perfect_ranking_is_one(self):
        grades = {"a": 2, "b": 1}
        assert ndcg_at_k(["a", "b"], grades, 2) == pytest.approx(1.0)

    def test_graded_relevance_rewards_ordering(self):
        grades = {"a": 1, "b": 2}
        assert ndcg_at_k(["b", "a"], grades, 2) > ndcg_at_k(["a", "b"], grades, 2)

    def test_exponential_and_linear_agree_on_binary(self):
        grades = {"b": 1, "d": 1}
        assert ndcg_at_k(RANKED, grades, 5, gain="exponential") == pytest.approx(
            ndcg_at_k(RANKED, grades, 5, gain="linear")
        )

    def test_exponential_and_linear_differ_on_graded(self):
        grades = {"a": 1, "b": 3}
        assert ndcg_at_k(["a", "b"], grades, 2, gain="exponential") != pytest.approx(
            ndcg_at_k(["a", "b"], grades, 2, gain="linear")
        )

    def test_no_judgments_scores_zero(self):
        assert ndcg_at_k(RANKED, {}, 5) == 0.0

    def test_fewer_relevant_than_k_can_still_reach_one(self):
        assert ndcg_at_k(["b", "x", "y"], {"b": 1}, 3) == pytest.approx(1.0)


class TestHitRate:
    def test_binary(self):
        assert hit_rate_at_k(RANKED, RELEVANT, 2) == 1.0
        assert hit_rate_at_k(RANKED, RELEVANT, 1) == 0.0


class TestEvaluateRanking:
    def test_produces_metric_at_k_keys(self):
        query = Query(
            query_id="q1",
            text="t",
            judgments=(Relevance("b", 1), Relevance("d", 1)),
        )
        scores = evaluate_ranking(RANKED, query, metrics=("precision", "ndcg"), cutoffs=(1, 5))
        assert set(scores) == {"precision@1", "precision@5", "ndcg@1", "ndcg@5"}

    def test_rejects_unknown_metric(self):
        query = Query(query_id="q1", text="t", judgments=(Relevance("b"),))
        with pytest.raises(ValueError, match="unknown metric"):
            evaluate_ranking(RANKED, query, metrics=("nope",), cutoffs=(5,))
