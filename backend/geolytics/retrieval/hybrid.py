"""Hybrid retrieval: fuse a dense and a lexical ranking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from geolytics.index.base import ScoredChunk
from geolytics.retrieval.base import Retriever, renumber

FusionMethod = Literal["rrf", "weighted"]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], k: float = 60.0
) -> list[ScoredChunk]:
    """Cormack et al. (2009) reciprocal rank fusion.

    Score contribution is 1/(k + rank), which uses only *rank*, never the raw
    score. That is the point: BM25 scores are unbounded and corpus-dependent
    while cosine scores live in [-1, 1], so they cannot be added directly. RRF
    sidesteps normalisation entirely, which is why it is the default here and
    the sensible baseline for the report.
    """
    totals: dict[str, float] = {}
    chunks: dict[str, ScoredChunk] = {}

    for ranking in rankings:
        for result in ranking:
            cid = result.chunk_id
            totals[cid] = totals.get(cid, 0.0) + 1.0 / (k + result.rank)
            chunks.setdefault(cid, result)

    ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return renumber(
        [ScoredChunk(chunk=chunks[cid].chunk, score=score, rank=0) for cid, score in ordered]
    )


def weighted_fusion(
    dense: Sequence[ScoredChunk],
    lexical: Sequence[ScoredChunk],
    alpha: float = 0.5,
) -> list[ScoredChunk]:
    """Min-max normalise each ranking, then blend: alpha*dense + (1-alpha)*lexical.

    `alpha` is a genuine experimental factor -- sweeping it from 0 (pure BM25)
    to 1 (pure dense) produces a curve worth a figure in the report.

    Caveat to state when reporting: min-max normalisation is computed over the
    retrieved candidates only, so the normalised scores depend on how deep each
    retriever was run. Keep the candidate depth fixed across conditions.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    dense_scores = _min_max(dense)
    lexical_scores = _min_max(lexical)
    chunks = {r.chunk_id: r for r in [*dense, *lexical]}

    combined = {
        cid: alpha * dense_scores.get(cid, 0.0) + (1.0 - alpha) * lexical_scores.get(cid, 0.0)
        for cid in chunks
    }
    ordered = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))
    return renumber(
        [ScoredChunk(chunk=chunks[cid].chunk, score=score, rank=0) for cid, score in ordered]
    )


def _min_max(results: Sequence[ScoredChunk]) -> dict[str, float]:
    if not results:
        return {}
    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        # Every candidate tied: a constant 1.0 preserves "all equally good"
        # instead of collapsing the whole ranking to 0.
        return {r.chunk_id: 1.0 for r in results}
    return {r.chunk_id: (r.score - lo) / (hi - lo) for r in results}


class HybridRetriever(Retriever):
    """Run a dense and a lexical retriever, fuse their rankings."""

    name = "hybrid"

    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        method: FusionMethod = "rrf",
        alpha: float = 0.5,
        rrf_k: float = 60.0,
        candidate_multiplier: int = 4,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.method = method
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.candidate_multiplier = max(1, candidate_multiplier)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha if self.method == "weighted" else None,
            "rrf_k": self.rrf_k if self.method == "rrf" else None,
            "candidate_multiplier": self.candidate_multiplier,
            "dense": self.dense.describe(),
            "lexical": self.lexical.describe(),
        }

    def retrieve(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        # Fetch deeper than top_k from each arm: a chunk ranked 30th by one
        # retriever and 2nd by the other should still be able to surface.
        depth = top_k * self.candidate_multiplier
        dense_hits = self.dense.retrieve(query, top_k=depth)
        lexical_hits = self.lexical.retrieve(query, top_k=depth)

        if self.method == "rrf":
            fused = reciprocal_rank_fusion([dense_hits, lexical_hits], k=self.rrf_k)
        elif self.method == "weighted":
            fused = weighted_fusion(dense_hits, lexical_hits, alpha=self.alpha)
        else:
            raise ValueError(f"unknown fusion method {self.method!r}")

        return fused[:top_k]
