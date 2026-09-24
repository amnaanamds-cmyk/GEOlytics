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

from collections.abc import Sequence
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
from geolytics.evaluation.heuristic_qagen import generate_heuristic_query_set
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


def build_query_set(
    documents: Sequence[Any],
    settings: Any,
    generator: str = "auto",
    questions_per_unit: int = 2,
) -> tuple[Any, Any]:
    """Generate the evaluation query set, LLM or template-based.

    "auto" uses the LLM when one is configured and falls back to templates
    otherwise, so a clean clone runs end to end without a model server. The
    fallback is visible, never silent: heuristic queries carry
    `provenance="heuristic"` and the returned report names the generator.
    """
    config = QAGenerationConfig(questions_per_unit=questions_per_unit)

    if generator == "heuristic" or (generator == "auto" and settings.llm_backend == "none"):
        return generate_heuristic_query_set(documents, config=config)

    return generate_query_set(documents, llm=build_llm(settings), config=config)


def run_chunking_experiment(
    url: str,
    max_pages: int = 10,
    metric: str = "ndcg@10",
    baseline: str = "fixed+dense",
    out_path: Path | None = None,
    questions_per_unit: int = 2,
    generator: str = "auto",
    persist_as: str | None = None,
    settings: Any = None,
) -> str:
    settings = settings or get_settings()
    embedder = build_embedder(settings)

    with Crawler(settings=settings) as crawler:
        results = crawler.crawl(url, max_pages=max_pages)
        crawl_summary = crawler.stats.summary()
    if not results:
        raise RuntimeError(f"no pages crawled from {url}: {crawl_summary}")
    documents = [r.document for r in results]

    query_set, generation = build_query_set(
        documents, settings, generator=generator, questions_per_unit=questions_per_unit
    )
    if len(query_set) == 0:
        raise RuntimeError(
            "query generation produced nothing; "
            "check the LLM backend, or pass --generator heuristic"
        )

    harness = ExperimentHarness(documents, query_set, embedder, top_k=10)
    runs = harness.run_all(build_conditions(embedder))
    comparisons = compare_runs(runs, metric=metric, baseline=baseline)

    if out_path is not None:
        save_runs(runs, out_path)

    persisted = ""
    if persist_as:
        from geolytics.pipeline import persist_runs

        run_ids = persist_runs(persist_as, runs, query_set)
        persisted = (
            f"Persisted {len(run_ids)} runs as experiment {persist_as!r}; "
            f"read them back from GET /experiments/{persist_as}"
        )

    return "\n\n".join(
        [
            f"# Chunking x retrieval on {url}",
            crawl_summary,
            generation.summary(),
            (
                "NOTE: template-generated queries are lexically biased toward their "
                "source passage and must not be used for reported results."
                if generation.config.get("generator") == "heuristic"
                else ""
            ),
            "## Results",
            results_table(runs, [metric, "recall@10", "mrr@10", "precision@5"]),
            f"## Paired comparisons vs {baseline}",
            comparison_table(comparisons),
            "## Methods",
            methods_note(runs, metric=metric),
            ("Saved to " + str(out_path) if out_path else "Not written to disk"),
            persisted,
        ]
    )
