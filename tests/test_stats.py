"""Paired statistics."""

from __future__ import annotations

import numpy as np
import pytest

from geolytics.evaluation.stats import (
    apply_correction,
    bootstrap_ci,
    cliffs_delta,
    compare_paired,
    holm_bonferroni,
)


class TestHolmBonferroni:
    def test_single_p_value_unchanged(self):
        assert holm_bonferroni([0.03]) == pytest.approx([0.03])

    def test_step_down_scaling(self):
        # Sorted: 0.01*3=0.03, 0.04*2=0.08, 0.05*1=0.05 -> monotone -> 0.08.
        assert holm_bonferroni([0.01, 0.04, 0.05]) == pytest.approx([0.03, 0.08, 0.08])

    def test_preserves_input_order(self):
        assert holm_bonferroni([0.05, 0.01]) == pytest.approx([0.05, 0.02])

    def test_is_monotone_non_decreasing(self):
        adjusted = holm_bonferroni([0.001, 0.002, 0.3, 0.31, 0.9])
        raw = [0.001, 0.002, 0.3, 0.31, 0.9]
        ordered = [a for _, a in sorted(zip(raw, adjusted, strict=True))]
        assert ordered == sorted(ordered)

    def test_clipped_at_one(self):
        assert all(p <= 1.0 for p in holm_bonferroni([0.4, 0.5, 0.6]))

    def test_less_conservative_than_bonferroni(self):
        p = [0.01, 0.04, 0.05]
        holm = holm_bonferroni(p)
        bonferroni = [min(1.0, x * len(p)) for x in p]
        assert all(h <= b for h, b in zip(holm, bonferroni, strict=True))

    def test_empty(self):
        assert holm_bonferroni([]) == []

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            holm_bonferroni([1.5])


class TestCliffsDelta:
    def test_complete_dominance(self):
        assert cliffs_delta([4, 5, 6], [1, 2, 3]) == pytest.approx(1.0)

    def test_complete_subordination(self):
        assert cliffs_delta([1, 2, 3], [4, 5, 6]) == pytest.approx(-1.0)

    def test_identical_samples(self):
        assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_bounded(self):
        rng = np.random.default_rng(0)
        d = cliffs_delta(rng.normal(size=50), rng.normal(size=50))
        assert -1.0 <= d <= 1.0


class TestBootstrapCI:
    def test_brackets_the_mean(self):
        rng = np.random.default_rng(1)
        diffs = rng.normal(loc=0.1, scale=0.05, size=200)
        low, high = bootstrap_ci(diffs)
        assert low < diffs.mean() < high

    def test_constant_differences_collapse(self):
        assert bootstrap_ci([0.2] * 30) == pytest.approx((0.2, 0.2))

    def test_reproducible_with_same_seed(self):
        diffs = [0.1, -0.2, 0.3, 0.05, -0.01] * 10
        assert bootstrap_ci(diffs, seed=42) == bootstrap_ci(diffs, seed=42)

    def test_wider_interval_for_noisier_data(self):
        rng = np.random.default_rng(2)
        tight = bootstrap_ci(rng.normal(0.1, 0.01, 100), seed=3)
        loose = bootstrap_ci(rng.normal(0.1, 0.50, 100), seed=3)
        assert (loose[1] - loose[0]) > (tight[1] - tight[0])


class TestComparePaired:
    def test_detects_a_consistent_improvement(self):
        rng = np.random.default_rng(5)
        base = rng.uniform(0.2, 0.8, size=60)
        improved = np.clip(base + 0.08, 0, 1)
        result = compare_paired(improved, base, "new", "old", "ndcg@10")
        assert result.mean_diff > 0
        assert result.p_value < 0.05
        assert result.n_wins_a == 60

    def test_no_difference_is_not_significant(self):
        rng = np.random.default_rng(6)
        a = rng.uniform(0, 1, size=60)
        b = a + rng.normal(0, 1e-9, size=60)
        assert compare_paired(a, b, "a", "b", "ndcg@10").p_value > 0.05

    def test_identical_vectors_short_circuit(self):
        scores = [0.5] * 20
        result = compare_paired(scores, scores, "a", "b", "ndcg@10")
        assert result.p_value == 1.0
        assert result.n_ties == 20
        assert "identical" in result.notes[0]

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal-length"):
            compare_paired([1, 2], [1, 2, 3], "a", "b", "m")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            compare_paired([], [], "a", "b", "m")

    def test_flags_small_samples(self):
        result = compare_paired([0.5, 0.6, 0.7], [0.1, 0.2, 0.3], "a", "b", "m")
        assert any("small" in note for note in result.notes)

    def test_effect_label_thresholds(self):
        rng = np.random.default_rng(7)
        a = rng.uniform(0.6, 1.0, size=40)
        b = rng.uniform(0.0, 0.4, size=40)
        assert compare_paired(a, b, "a", "b", "m").effect_label == "large"

    def test_summary_is_readable(self):
        result = compare_paired([0.6] * 20, [0.4] * 20, "new", "old", "ndcg@10")
        text = result.summary()
        assert "ndcg@10" in text and "new" in text and "n=20" in text


class TestApplyCorrection:
    def test_attaches_adjusted_p_values(self):
        rng = np.random.default_rng(8)
        base = rng.uniform(0.2, 0.8, size=40)
        comparisons = [
            compare_paired(np.clip(base + delta, 0, 1), base, f"c{i}", "base", "ndcg@10")
            for i, delta in enumerate([0.15, 0.001, 0.0005])
        ]
        corrected = apply_correction(comparisons)
        assert all(c.p_adjusted is not None for c in corrected)
        assert all(c.p_adjusted >= c.p_value for c in corrected)

    def test_significance_uses_adjusted_value(self):
        rng = np.random.default_rng(9)
        base = rng.uniform(0.2, 0.8, size=30)
        comparison = compare_paired(np.clip(base + 0.2, 0, 1), base, "a", "b", "m")
        assert comparison.significant
        from dataclasses import replace

        assert not replace(comparison, p_adjusted=0.9).significant
