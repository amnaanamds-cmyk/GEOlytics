"""Paired significance testing for retrieval experiments.

Why this module exists: reporting one mean per condition cannot answer "is
semantic chunking actually better, or did it win by noise?". Because every
condition is run over the *same* queries, the comparisons are paired, and
pairing is what gives a modest number of queries enough power to detect a
small difference.

Defaults chosen here, and the reasoning to put in the report:

* **Wilcoxon signed-rank**, not a paired t-test. Per-query retrieval metrics
  are bounded, discrete and heavily skewed (nDCG@10 piles up at 0 and 1), so
  the t-test's normality assumption is violated. Wilcoxon only assumes a
  symmetric difference distribution.
* **Bootstrap CI on the mean difference**, reported alongside p. A p-value says
  whether an effect exists; the interval says how big it plausibly is, which is
  the part a reader actually needs.
* **Cliff's delta** as a non-parametric effect size, for the same reason.
* **Holm-Bonferroni** across a family of comparisons. Four chunkers compared
  pairwise is six tests; at alpha = 0.05 the chance of at least one false
  positive is about 26% uncorrected.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats as scipy_stats

Alternative = Literal["two-sided", "greater", "less"]


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """The result of comparing two conditions over the same queries."""

    name_a: str
    name_b: str
    metric: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    median_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    statistic: float
    effect_size: float
    n_ties: int
    n_wins_a: int
    n_wins_b: int
    p_adjusted: float | None = None
    alpha: float = 0.05
    notes: list[str] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        """Uses the adjusted p-value when a correction has been applied."""
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        return p < self.alpha

    @property
    def effect_label(self) -> str:
        """Romano et al. (2006) thresholds for Cliff's delta."""
        d = abs(self.effect_size)
        if d < 0.147:
            return "negligible"
        if d < 0.33:
            return "small"
        if d < 0.474:
            return "medium"
        return "large"

    def summary(self) -> str:
        direction = "higher" if self.mean_diff > 0 else "lower"
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        label = "p_adj" if self.p_adjusted is not None else "p"
        return (
            f"{self.metric}: {self.name_a} {self.mean_a:.4f} vs "
            f"{self.name_b} {self.mean_b:.4f} "
            f"(diff {self.mean_diff:+.4f} {direction}, "
            f"95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}], "
            f"{label}={p:.4g}, delta={self.effect_size:+.3f} {self.effect_label}, n={self.n})"
        )


def bootstrap_ci(
    differences: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 20240501,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences.

    Resamples *queries* (the differences), which preserves the pairing. A fixed
    seed keeps the reported interval reproducible -- record it in the report.
    """
    diffs = np.asarray(differences, dtype=np.float64)
    if diffs.size == 0:
        return (float("nan"), float("nan"))
    if np.allclose(diffs, diffs[0]):
        return (float(diffs[0]), float(diffs[0]))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    means = diffs[idx].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.percentile(means, 100 * tail)),
        float(np.percentile(means, 100 * (1.0 - tail))),
    )


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta: P(a > b) - P(a < b), in [-1, 1].

    Computed via rank-sum rather than the O(n*m) pairwise loop so it stays
    cheap on large query sets.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    n, m = x.size, y.size
    if n == 0 or m == 0:
        return float("nan")

    combined = np.concatenate([x, y])
    ranks = scipy_stats.rankdata(combined)
    rank_sum_x = float(ranks[:n].sum())
    # U statistic for x over y, then map onto [-1, 1].
    u = rank_sum_x - n * (n + 1) / 2.0
    return float(2.0 * u / (n * m) - 1.0)


def compare_paired(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    name_a: str,
    name_b: str,
    metric: str,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 20240501,
) -> PairedComparison:
    """Compare two conditions scored on the same, identically ordered queries."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"paired comparison needs equal-length score vectors, got {a.shape} and {b.shape}"
        )
    if a.size == 0:
        raise ValueError("cannot compare empty score vectors")

    diffs = a - b
    notes: list[str] = []
    n_ties = int(np.sum(diffs == 0))

    if n_ties == diffs.size:
        # scipy refuses an all-zero difference vector; the correct conclusion
        # is simply "no evidence of a difference".
        statistic, p_value = 0.0, 1.0
        notes.append("identical on every query; no test performed")
    else:
        if diffs.size < 10:
            notes.append(
                f"n={diffs.size} is small for Wilcoxon; p-value is approximate"
            )
        result = scipy_stats.wilcoxon(
            a, b, alternative=alternative, zero_method="wilcox", method="auto"
        )
        statistic, p_value = float(result.statistic), float(result.pvalue)

    ci_low, ci_high = bootstrap_ci(diffs, n_resamples=n_resamples, seed=seed)

    return PairedComparison(
        name_a=name_a,
        name_b=name_b,
        metric=metric,
        n=int(diffs.size),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=float(diffs.mean()),
        median_diff=float(np.median(diffs)),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        statistic=statistic,
        effect_size=cliffs_delta(a, b),
        n_ties=n_ties,
        n_wins_a=int(np.sum(diffs > 0)),
        n_wins_b=int(np.sum(diffs < 0)),
        alpha=alpha,
        notes=notes,
    )


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order.

    Uniformly more powerful than plain Bonferroni at the same family-wise error
    rate, so there is no reason to use Bonferroni instead. Adjusted values are
    made monotone non-decreasing along the sorted order, per the standard
    definition, and clipped at 1.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        return []
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")

    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * float(p[idx]))
        adjusted_sorted[i] = min(1.0, running)

    out = np.empty(m, dtype=np.float64)
    out[order] = adjusted_sorted
    return [float(v) for v in out]


def apply_correction(
    comparisons: Sequence[PairedComparison], alpha: float = 0.05
) -> list[PairedComparison]:
    """Attach Holm-adjusted p-values to a family of comparisons.

    Call this once per *family* -- typically all pairwise comparisons of one
    factor on one primary metric. Correcting across unrelated metrics as if
    they were one family is over-conservative and will hide real effects.
    """
    adjusted = holm_bonferroni([c.p_value for c in comparisons], alpha=alpha)
    from dataclasses import replace

    return [
        replace(c, p_adjusted=p, alpha=alpha)
        for c, p in zip(comparisons, adjusted, strict=True)
    ]
