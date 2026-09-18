"""The headline experiment: chunking strategy x retrieval strategy.

Run it with `geolytics experiment <url>`. It produces the result table, the
paired-comparison table and the methods paragraph that the report needs.

Design notes that belong in the methodology chapter:

* The similarity metric is **not** a factor. `geolytics check-metrics` shows
  why: for normalised embeddings, cosine, dot product and Euclidean distance
  are rank-equivalent. The freed axis is spent on retrieval strategy, which
  does vary.
* Gold spans are generated once, from a segmentation that is not one of the
  conditions, and projected onto each chunking independently.
* `fixed+dense` is the baseline. Comparing k-1 conditions against a baseline
  rather than all k(k-1)/2 pairs means a gentler Holm correction and more
  power to detect a real effect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from geolytics.chunking.registry import build_chunker
from geolytics.config import get_settings
from geolytics.crawl.crawler import Crawler
from geolytics.embedding.registry import build_embedder
from geolytics.evaluation.harness import (
    Condition,
    ExperimentHarness,
    RetrievalContext,
    compare_runs,
    save_runs,
)
from geolytics.evaluation.qagen import QAGenerationConfig, generate_query_set
from geolytics.evaluation.report import comparison_table, methods_note, results_table
from geolytics.llm import build_llm
from geolytics.retrieval.bm25 import BM25Retriever
from geolytics.retrieval.dense import DenseRetriever
from geolytics.retrieval.hybrid import HybridRetriever


def dense_factory(context: RetrievalContext) -> DenseRetriever:
    return DenseRetriever(context.store, context.embedder, context.collection)


def hybrid_factory(context: RetrievalContext) -> HybridRetriever:
    return HybridRetriever(
        dense=DenseRetriever(context.store, context.embedder, context.collection),
        lexical=BM25Retriever(context.chunks),
        method="rrf",
    )


def build_conditions(embedder: Any) -> list[Condition]:
    """The grid. Every chunker is size-matched as closely as its design allows."""
    chunkers = {
        "fixed": build_chunker("fixed", chunk_tokens=256, overlap_tokens=32),
        "sentence": build_chunker("sentence", max_tokens=256, overlap_sentences=1),
        "semantic": build_chunker(
            "semantic", embedder=embedder, breakpoint_percentile=90.0, max_tokens=256
        ),
        "parent_child": build_chunker(
            "parent_child",
            parent={"name": "sentence", "max_tokens": 512},
            child={"name": "sentence", "max_tokens": 128, "overlap_sentences": 0},
        ),
    }
    retrievers = {"dense": dense_factory, "hybrid": hybrid_factory}

    return [
        Condition(name=f"{c_name}+{r_name}", chunker=chunker, retriever_factory=factory)
        for c_name, chunker in chunkers.items()
        for r_name, factory in retrievers.items()
    ]


def run_chunking_experiment(
    url: str,
    max_pages: int = 10,
    metric: str = "ndcg@10",
    baseline: str = "fixed+dense",
    out_path: Path | None = None,
    questions_per_unit: int = 2,
) -> str:
    settings = get_settings()
    embedder = build_embedder(settings)

    with Crawler(settings=settings) as crawler:
        results = crawler.crawl(url, max_pages=max_pages)
    if not results:
        raise RuntimeError(f"no pages crawled from {url}: {crawler.stats.summary()}")
    documents = [r.document for r in results]

    query_set, generation = generate_query_set(
        documents,
        llm=build_llm(settings),
        config=QAGenerationConfig(questions_per_unit=questions_per_unit),
    )
    if len(query_set) == 0:
        raise RuntimeError(
            "query generation produced nothing; check that the LLM backend is reachable"
        )

    harness = ExperimentHarness(documents, query_set, embedder, top_k=10)
    runs = harness.run_all(build_conditions(embedder))
    comparisons = compare_runs(runs, metric=metric, baseline=baseline)

    if out_path is not None:
        save_runs(runs, out_path)

    return "\n\n".join(
        [
            f"# Chunking x retrieval on {url}",
            (
                f"{generation.n_generated} queries from {generation.n_units} generation units "
                f"(mean lexical overlap {generation.mean_lexical_overlap:.2f}, "
                f"{generation.n_dropped_overlap} dropped above threshold, "
                f"{generation.n_failed_units} units failed)"
            ),
            "## Results",
            results_table(runs, [metric, "recall@10", "mrr@10", "precision@5"]),
            f"## Paired comparisons vs {baseline}",
            comparison_table(comparisons),
            "## Methods",
            methods_note(runs, metric=metric),
            (
                "Saved to " + str(out_path) if out_path else "Not saved (pass --out to persist)"
            ),
        ]
    )
