"""Ranking metrics.

Every function takes the ranked chunk ids and the query's judgments and returns
a score for *one query*. Per-query scores are what the harness stores, because
the paired significance tests in `stats.py` need them -- a table of means alone
cannot be tested.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from geolytics.evaluation.qrels import Query

Gain = Literal["linear", "exponential"]


def precision_at_k(ranked_ids: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Fraction of the top-k that is relevant.

    Divides by `k`, not by the number retrieved (the TREC convention). A system
    that returns 3 documents and gets all 3 right scores P@10 = 0.3, not 1.0 --
    returning fewer results is not rewarded.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return hits / k


def recall_at_k(ranked_ids: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Fraction of all relevant chunks found in the top-k."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return hits / len(relevant)


def hit_rate_at_k(ranked_ids: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """1.0 if any relevant chunk is in the top-k (a.k.a. success@k)."""
    if k <= 0:
        raise ValueError("k must be positive")
    return 1.0 if any(cid in relevant for cid in ranked_ids[:k]) else 0.0


def mrr(ranked_ids: Sequence[str], relevant: frozenset[str], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant chunk, 0 if none within `k`."""
    cutoff = len(ranked_ids) if k is None else k
    for rank, cid in enumerate(ranked_ids[:cutoff], start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(
    ranked_ids: Sequence[str], relevant: frozenset[str], k: int | None = None
) -> float:
    """Average of P@i over the positions holding a relevant chunk.

    Normalised by the total number of relevant chunks, so unretrieved relevant
    chunks are penalised (the standard TREC definition).
    """
    if not relevant:
        return 0.0
    cutoff = len(ranked_ids) if k is None else k
    hits = 0
    total = 0.0
    for rank, cid in enumerate(ranked_ids[:cutoff], start=1):
        if cid in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def ndcg_at_k(
    ranked_ids: Sequence[str],
    grades: Mapping[str, int],
    k: int,
    gain: Gain = "exponential",
) -> float:
    """Normalised discounted cumulative gain.

    `gain="exponential"` uses (2^rel - 1), the standard for graded relevance;
    it is identical to linear gain when judgments are binary. The ideal ranking
    is computed over *all* judged chunks truncated at k, so a query with fewer
    than k relevant chunks can still reach 1.0.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    def g(rel: float) -> float:
        return (2.0**rel - 1.0) if gain == "exponential" else float(rel)

    dcg = sum(
        g(grades.get(cid, 0)) / math.log2(rank + 1)
        for rank, cid in enumerate(ranked_ids[:k], start=1)
    )
    ideal = sorted((v for v in grades.values() if v > 0), reverse=True)[:k]
    idcg = sum(g(rel) / math.log2(rank + 1) for rank, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


MetricFn = Callable[[Sequence[str], Query, int], float]

METRIC_FUNCTIONS: dict[str, MetricFn] = {
    "precision": lambda ids, q, k: precision_at_k(ids, q.relevant_ids, k),
    "recall": lambda ids, q, k: recall_at_k(ids, q.relevant_ids, k),
    "hit_rate": lambda ids, q, k: hit_rate_at_k(ids, q.relevant_ids, k),
    "mrr": lambda ids, q, k: mrr(ids, q.relevant_ids, k),
    "map": lambda ids, q, k: average_precision(ids, q.relevant_ids, k),
    "ndcg": lambda ids, q, k: ndcg_at_k(ids, q.grades, k),
}

DEFAULT_METRICS = ("precision", "recall", "mrr", "ndcg")
DEFAULT_CUTOFFS = (1, 3, 5, 10)


def evaluate_ranking(
    ranked_ids: Sequence[str],
    query: Query,
    metrics: Sequence[str] = DEFAULT_METRICS,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> dict[str, float]:
    """Score one ranking against one query. Keys are ``"<metric>@<k>"``."""
    out: dict[str, float] = {}
    for metric in metrics:
        try:
            fn = METRIC_FUNCTIONS[metric]
        except KeyError:
            raise ValueError(
                f"unknown metric {metric!r}; expected one of {sorted(METRIC_FUNCTIONS)}"
            ) from None
        for k in cutoffs:
            out[f"{metric}@{k}"] = fn(ranked_ids, query, k)
    return out
