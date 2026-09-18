"""The experiment harness: run a grid of conditions, keep per-query scores.

One `Condition` is one point in the experimental grid -- a chunking strategy
plus a retriever. The harness builds an index per condition, runs every query,
and stores the *per-query* score vector, which is what `stats.compare_paired`
needs. Aggregate means are derived, never the primary record.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from geolytics.chunking.base import Chunk, ChunkingStrategy, Document, WhitespaceTokenCounter
from geolytics.embedding.base import Embedder
from geolytics.evaluation.metrics import DEFAULT_CUTOFFS, DEFAULT_METRICS, evaluate_ranking
from geolytics.evaluation.projection import ProjectionStats, project_query_set
from geolytics.evaluation.qrels import QuerySet
from geolytics.evaluation.stats import PairedComparison, apply_correction, compare_paired
from geolytics.index.base import VectorStore
from geolytics.index.memory import InMemoryVectorStore
from geolytics.retrieval.base import Retriever


@dataclass(frozen=True, slots=True)
class RetrievalContext:
    """Everything a retriever factory needs to build itself for one condition."""

    chunks: list[Chunk]
    embedder: Embedder
    store: VectorStore
    collection: str


RetrieverFactory = Callable[[RetrievalContext], Retriever]


@dataclass(frozen=True, slots=True)
class Condition:
    """One cell of the experimental grid."""

    name: str
    chunker: ChunkingStrategy
    retriever_factory: RetrieverFactory
    notes: str = ""


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Corpus shape for one chunking. Report these next to every result table.

    Chunk count and mean length are confounders: a strategy producing many
    small chunks has more chances to hit and shorter passages to match. A
    difference in retrieval quality is only interesting once a reader can see
    that it is not purely a size effect.
    """

    n_chunks: int
    mean_tokens: float
    median_tokens: float
    p10_tokens: float
    p90_tokens: float
    total_tokens: int


@dataclass(slots=True)
class RunResult:
    """The complete record of one condition."""

    condition: str
    chunker: dict[str, Any]
    retriever: dict[str, Any]
    per_query: dict[str, dict[str, float]]
    ranked: dict[str, list[str]]
    index_stats: IndexStats
    projection: ProjectionStats
    n_queries_scored: int
    elapsed_seconds: float
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def query_ids(self) -> list[str]:
        return sorted(self.per_query)

    def scores(self, metric: str, query_ids: Sequence[str] | None = None) -> list[float]:
        """Per-query scores for `metric`, in `query_ids` order (sorted by default)."""
        ids = list(query_ids) if query_ids is not None else self.query_ids
        try:
            return [self.per_query[qid][metric] for qid in ids]
        except KeyError as exc:
            raise KeyError(
                f"condition {self.condition!r} has no score for {exc.args[0]!r}"
            ) from None

    def mean(self, metric: str) -> float:
        values = [scores[metric] for scores in self.per_query.values()]
        return float(np.mean(values)) if values else float("nan")

    def aggregate(self) -> dict[str, float]:
        if not self.per_query:
            return {}
        metrics = next(iter(self.per_query.values())).keys()
        return {m: self.mean(m) for m in metrics}

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "chunker": self.chunker,
            "retriever": self.retriever,
            "index_stats": asdict(self.index_stats),
            "projection": asdict(self.projection),
            "n_queries_scored": self.n_queries_scored,
            "elapsed_seconds": self.elapsed_seconds,
            "aggregate": self.aggregate(),
            "per_query": self.per_query,
            "ranked": self.ranked,
            "notes": self.notes,
            "metadata": self.metadata,
        }


class ExperimentHarness:
    """Runs conditions over a fixed document corpus and query set."""

    def __init__(
        self,
        documents: Sequence[Document],
        query_set: QuerySet,
        embedder: Embedder,
        store: VectorStore | None = None,
        metrics: Sequence[str] = DEFAULT_METRICS,
        cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
        top_k: int = 10,
        min_coverage: float = 0.5,
        drop_unjudged: bool = True,
    ) -> None:
        if top_k < max(cutoffs):
            raise ValueError(
                f"top_k={top_k} is below the largest cutoff {max(cutoffs)}; "
                "metrics above top_k would be silently truncated"
            )
        self.documents = list(documents)
        self.query_set = query_set
        self.embedder = embedder
        self.store = store or InMemoryVectorStore()
        self.metrics = list(metrics)
        self.cutoffs = list(cutoffs)
        self.top_k = top_k
        self.min_coverage = min_coverage
        self.drop_unjudged = drop_unjudged
        self._tokens = WhitespaceTokenCounter()

    def run(self, condition: Condition) -> RunResult:
        started = time.perf_counter()

        chunks = [c for doc in self.documents for c in condition.chunker.chunk(doc)]
        if not chunks:
            raise ValueError(f"condition {condition.name!r} produced no chunks")

        collection = f"eval__{condition.name}"
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.create_collection(collection, dim=self.embedder.dim)
        self.store.upsert(collection, chunks, vectors)

        retriever = condition.retriever_factory(
            RetrievalContext(
                chunks=chunks,
                embedder=self.embedder,
                store=self.store,
                collection=collection,
            )
        )

        projected, projection = project_query_set(
            self.query_set, chunks, min_coverage=self.min_coverage
        )
        scoring_set = projected.judged() if self.drop_unjudged else projected

        per_query: dict[str, dict[str, float]] = {}
        ranked: dict[str, list[str]] = {}
        for query in scoring_set:
            hits = retriever.retrieve(query.text, top_k=self.top_k)
            ids = [h.chunk_id for h in hits]
            ranked[query.query_id] = ids
            per_query[query.query_id] = evaluate_ranking(
                ids, query, metrics=self.metrics, cutoffs=self.cutoffs
            )

        return RunResult(
            condition=condition.name,
            chunker=condition.chunker.describe(),
            retriever=retriever.describe(),
            per_query=per_query,
            ranked=ranked,
            index_stats=self._index_stats(chunks),
            projection=projection,
            n_queries_scored=len(per_query),
            elapsed_seconds=time.perf_counter() - started,
            notes=condition.notes,
        )

    def run_all(self, conditions: Sequence[Condition]) -> list[RunResult]:
        return [self.run(c) for c in conditions]

    def _index_stats(self, chunks: Sequence[Chunk]) -> IndexStats:
        lengths = np.array([self._tokens.count(c.text) for c in chunks], dtype=np.float64)
        return IndexStats(
            n_chunks=len(chunks),
            mean_tokens=float(lengths.mean()),
            median_tokens=float(np.median(lengths)),
            p10_tokens=float(np.percentile(lengths, 10)),
            p90_tokens=float(np.percentile(lengths, 90)),
            total_tokens=int(lengths.sum()),
        )


def common_query_ids(runs: Sequence[RunResult]) -> list[str]:
    """Query ids scored under *every* run.

    Projection can drop a different subset per chunking strategy, so the paired
    tests must run on the intersection -- comparing means over different query
    subsets is not a paired comparison at all.
    """
    if not runs:
        return []
    shared = set(runs[0].per_query)
    for run in runs[1:]:
        shared &= set(run.per_query)
    return sorted(shared)


def compare_runs(
    runs: Sequence[RunResult],
    metric: str,
    baseline: str | None = None,
    alpha: float = 0.05,
    correct: bool = True,
) -> list[PairedComparison]:
    """Pairwise-compare conditions on one metric, Holm-corrected as one family.

    With `baseline`, every other condition is compared against it (k-1 tests);
    otherwise all pairs are compared (k(k-1)/2 tests). Prefer a baseline when
    the research question is "does X beat the standard approach?" -- fewer
    tests means a less punishing correction and more power.
    """
    if len(runs) < 2:
        raise ValueError("need at least two runs to compare")

    shared = common_query_ids(runs)
    if not shared:
        raise ValueError("runs share no scored queries; cannot compare pairwise")

    by_name = {run.condition: run for run in runs}
    if baseline is not None and baseline not in by_name:
        raise ValueError(f"baseline {baseline!r} is not among {sorted(by_name)}")

    pairs: list[tuple[str, str]] = []
    names = [run.condition for run in runs]
    if baseline is not None:
        pairs = [(n, baseline) for n in names if n != baseline]
    else:
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]

    comparisons = [
        compare_paired(
            by_name[a].scores(metric, shared),
            by_name[b].scores(metric, shared),
            name_a=a,
            name_b=b,
            metric=metric,
            alpha=alpha,
        )
        for a, b in pairs
    ]
    return apply_correction(comparisons, alpha=alpha) if correct else comparisons


def save_runs(runs: Sequence[RunResult], path: str | Path) -> Path:
    """Persist full run records as JSON, so a result can be re-analysed later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([r.to_dict() for r in runs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
