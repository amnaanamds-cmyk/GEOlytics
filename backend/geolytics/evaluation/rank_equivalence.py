"""Whether cosine, dot product and Euclidean distance are separate conditions.

They usually are not, and a project that tabulates them as three retrieval
variants will report three identical columns.

For L2-normalised vectors a and b:

    dot(a, b) = cos(a, b)                     (norms are 1)
    ||a - b||^2 = 2 - 2 * dot(a, b)           (expand the square)

Euclidean distance is therefore a strictly decreasing function of the dot
product, so ranking by *any* of the three produces the same order. Sentence-
Transformers models normalise by default, and Qdrant normalises on upsert for
`Cosine` collections, so the identity holds end to end in this system.

The comparison is only meaningful for a model trained for dot-product retrieval
on unnormalised vectors -- the `multi-qa-*-dot-v1` family -- where the vector
norm encodes something (roughly, passage specificity) that cosine discards.

`verify_rank_equivalence` turns the argument into a measurement: run it, report
Kendall's tau = 1.0 for the normalised case, and treat the similarity metric as
a fixed setting rather than an experimental factor. Then spend the freed axis
on something that varies -- the embedding model, the hybrid fusion weight, or a
reranker.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from geolytics.embedding.base import Embedder
from geolytics.index.base import Metric
from geolytics.index.memory import score_matrix


@dataclass(frozen=True, slots=True)
class EquivalenceResult:
    metric_a: Metric
    metric_b: Metric
    normalized: bool
    n_queries: int
    mean_kendall_tau: float
    identical_top_k_rate: float
    top_k: int
    n_tied_queries: int = 0
    tolerance: float = 1e-6

    @property
    def rank_equivalent(self) -> bool:
        return self.identical_top_k_rate == 1.0

    def summary(self) -> str:
        verdict = "rank-equivalent" if self.rank_equivalent else "NOT rank-equivalent"
        tied = (
            f", {self.n_tied_queries} of them with tied scores"
            if self.n_tied_queries
            else ""
        )
        return (
            f"{self.metric_a} vs {self.metric_b} "
            f"(normalized={self.normalized}): {verdict}; "
            f"mean Kendall tau={self.mean_kendall_tau:.4f}, "
            f"identical top-{self.top_k} on {self.identical_top_k_rate:.1%} of "
            f"{self.n_queries} queries{tied}"
        )


def _canonical_order(scores: np.ndarray, k: int, tolerance: float) -> list[int]:
    """Rank indices by score, breaking ties deterministically.

    Float32 embeddings make genuinely tied passages differ in the last bits,
    and the two metrics round that noise differently -- cosine divides by a
    norm that is only approximately 1, dot does not. Comparing raw argsorts
    then reports a ranking difference where there is only tie-breaking noise.

    Scores are therefore quantised to a relative tolerance before sorting, and
    remaining ties are broken by index, which is identical under both metrics.
    Quantisation is relative because the metrics live on different scales:
    cosine in [-1, 1], dot unbounded, negative Euclidean in (-inf, 0].
    """
    quantised = _quantise(scores, tolerance)
    # lexsort's last key is primary: sort by descending score, then by index.
    order = np.lexsort((np.arange(scores.size), -quantised))
    return [int(i) for i in order[:k]]


def _quantise(scores: np.ndarray, tolerance: float) -> np.ndarray:
    scale = float(np.max(np.abs(scores))) or 1.0
    return np.round(scores / scale / tolerance) * tolerance


def _has_ties(scores: np.ndarray, k: int, tolerance: float) -> bool:
    top = np.sort(_quantise(scores, tolerance))[::-1][: k + 1]
    return bool(np.any(np.diff(top) == 0))


def verify_rank_equivalence(
    embedder: Embedder,
    passages: Sequence[str],
    queries: Sequence[str],
    metric_a: Metric = "cosine",
    metric_b: Metric = "dot",
    top_k: int = 10,
    tolerance: float = 1e-6,
) -> EquivalenceResult:
    """Empirically check whether two metrics induce the same ranking.

    `tolerance` is a *relative* score quantisation applied before comparing, so
    that floating-point noise between two mathematically equivalent metrics is
    not reported as a ranking difference. Lower it to make the check stricter.
    """
    if not passages or not queries:
        raise ValueError("need at least one passage and one query")

    doc_vectors = np.asarray(embedder.embed(list(passages)), dtype=np.float64)
    query_vectors = np.asarray(embedder.embed_query(list(queries)), dtype=np.float64)

    taus: list[float] = []
    identical = 0
    tied = 0
    k = min(top_k, len(passages))

    for query_vector in query_vectors:
        scores_a = score_matrix(doc_vectors, query_vector, metric_a)
        scores_b = score_matrix(doc_vectors, query_vector, metric_b)

        if len(passages) > 1:
            # Tau is computed on the quantised scores for the same reason the
            # order comparison is: otherwise last-bit noise between two
            # equivalent metrics reports tau < 1 next to a verdict of
            # "rank-equivalent", and the two numbers appear to contradict.
            tau = scipy_stats.kendalltau(
                _quantise(scores_a, tolerance), _quantise(scores_b, tolerance)
            ).statistic
            taus.append(1.0 if np.isnan(tau) else float(tau))

        if _has_ties(scores_a, k, tolerance) or _has_ties(scores_b, k, tolerance):
            tied += 1

        if _canonical_order(scores_a, k, tolerance) == _canonical_order(
            scores_b, k, tolerance
        ):
            identical += 1

    return EquivalenceResult(
        metric_a=metric_a,
        metric_b=metric_b,
        normalized=embedder.normalized,
        n_queries=len(queries),
        mean_kendall_tau=float(np.mean(taus)) if taus else 1.0,
        identical_top_k_rate=identical / len(queries),
        top_k=k,
        n_tied_queries=tied,
        tolerance=tolerance,
    )
