"""The audit pipeline: crawl -> extract -> chunk -> index -> signal -> score.

This is the application path. The research path (`evaluation.harness`) shares
the same chunking, embedding, indexing and retrieval components, which is the
point of the layering -- an experimental finding about chunking transfers
directly into what the product does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from geolytics.chunking.base import Chunk, ChunkingStrategy, Document
from geolytics.chunking.registry import build_chunker
from geolytics.config import Settings, get_settings
from geolytics.crawl.crawler import Crawler
from geolytics.db.models import Audit, Crawl, Organization, Page, PageScore, Site
from geolytics.db.session import session_scope
from geolytics.embedding.base import Embedder
from geolytics.embedding.registry import build_embedder
from geolytics.geo.scoring import GEOScore, GEOScorer, SignalWeights
from geolytics.geo.signals import compute_signals
from geolytics.index.base import VectorStore
from geolytics.index.memory import InMemoryVectorStore
from geolytics.tenancy.quotas import PAGES_CRAWLED, record_usage


@dataclass(slots=True)
class AuditOutcome:
    """Everything one audit produced, before persistence."""

    site_url: str
    documents: list[Document]
    chunks: list[Chunk]
    scores: list[GEOScore]
    overall_score: float
    weights_fitted: bool
    crawl_summary: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def run_audit(
    url: str,
    max_pages: int = 25,
    chunker: ChunkingStrategy | str = "sentence",
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    weights: SignalWeights | None = None,
    settings: Settings | None = None,
    collection: str | None = None,
) -> AuditOutcome:
    """Crawl a site, index it, and score every page."""
    settings = settings or get_settings()
    embedder = embedder or build_embedder(settings)
    store = store or InMemoryVectorStore()
    strategy = build_chunker(chunker, embedder=embedder) if isinstance(chunker, str) else chunker
    scorer = GEOScorer(weights=weights)

    with Crawler(settings=settings) as crawler:
        results = crawler.crawl(url, max_pages=max_pages)
        crawl_summary = crawler.stats.summary()

    warnings: list[str] = []
    if not results:
        warnings.append(
            "no pages were fetched; check robots.txt, the URL, and network access"
        )
        return AuditOutcome(
            site_url=url,
            documents=[],
            chunks=[],
            scores=[],
            overall_score=0.0,
            weights_fitted=scorer.weights.fitted,
            crawl_summary=crawl_summary,
            warnings=warnings,
        )

    documents = [r.document for r in results]
    html_by_doc = {r.document.doc_id: r.html for r in results}

    chunks_by_doc: dict[str, list[Chunk]] = {}
    all_chunks: list[Chunk] = []
    for document in documents:
        produced = strategy.chunk(document)
        chunks_by_doc[document.doc_id] = produced
        all_chunks.extend(produced)

    name = collection or f"audit__{documents[0].doc_id}"
    if all_chunks:
        vectors = embedder.embed([c.text for c in all_chunks])
        store.create_collection(name, dim=embedder.dim)
        store.upsert(name, all_chunks, vectors)
    else:
        warnings.append("crawl produced pages but no chunks; check content extraction")

    scores = [
        scorer.score(
            compute_signals(
                document,
                chunks=chunks_by_doc.get(document.doc_id, []),
                html=html_by_doc.get(document.doc_id),
            )
        )
        for document in documents
    ]

    if not scorer.weights.fitted:
        warnings.append(scorer.weights.provenance())

    overall = sum(s.score for s in scores) / len(scores) if scores else 0.0

    return AuditOutcome(
        site_url=url,
        documents=documents,
        chunks=all_chunks,
        scores=scores,
        overall_score=round(overall, 2),
        weights_fitted=scorer.weights.fitted,
        crawl_summary=crawl_summary,
        warnings=warnings,
        metadata={
            "collection": name,
            "chunker": strategy.describe(),
            "embedder": embedder.describe(),
            "n_pages": len(documents),
            "n_chunks": len(all_chunks),
        },
    )


def persist_audit(audit_id: int, outcome: AuditOutcome) -> None:
    """Write an outcome into the audit, crawl, page and score tables."""
    with session_scope() as session:
        audit = session.get(Audit, audit_id)
        if audit is None:
            raise ValueError(f"audit {audit_id} does not exist")

        site = session.get(Site, audit.site_id)
        if site is None:
            raise ValueError(f"audit {audit_id} has no site")

        settings = get_settings()
        crawl = Crawl(
            org_id=audit.org_id,
            site_id=site.id,
            status="complete",
            pages_fetched=len(outcome.documents),
            user_agent=settings.crawl_user_agent,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )
        session.add(crawl)
        session.flush()

        pages_by_doc: dict[str, Page] = {}
        for document in outcome.documents:
            page = Page(
                crawl_id=crawl.id,
                doc_id=document.doc_id,
                url=document.url,
                title=document.title,
                text=document.text,
                sections=[
                    {
                        "text": s.text,
                        "heading_path": list(s.heading_path),
                        "start_char": s.start_char,
                    }
                    for s in document.sections
                ],
                extra=document.metadata,
            )
            session.add(page)
            pages_by_doc[document.doc_id] = page
        session.flush()

        for score in outcome.scores:
            page = pages_by_doc.get(score.doc_id)
            if page is None:
                continue
            session.add(
                PageScore(
                    audit_id=audit.id,
                    page_id=page.id,
                    score=score.score,
                    signals=score.signals,
                    contributions=score.contributions,
                    recommendations=score.recommendations,
                )
            )

        audit.crawl_id = crawl.id
        audit.status = "complete"
        audit.overall_score = outcome.overall_score
        audit.weights_fitted = outcome.weights_fitted
        audit.summary = {
            "crawl": outcome.crawl_summary,
            "warnings": outcome.warnings,
            **outcome.metadata,
        }

        # Metered after the fact, on pages actually fetched. Charging for the
        # requested page count would bill for pages robots.txt or the URL
        # policy refused.
        if outcome.documents:
            record_usage(session, audit.org_id, PAGES_CRAWLED, len(outcome.documents))


def mark_audit_failed(audit_id: int, error: str) -> None:
    with session_scope() as session:
        audit = session.get(Audit, audit_id)
        if audit is not None:
            audit.status = "failed"
            audit.error = error


def persist_runs(
    experiment: str,
    runs: Sequence[Any],
    query_set: Any = None,
    site_id: int | None = None,
    org_id: int | None = None,
) -> list[int]:
    """Write experiment runs and their per-query scores to the database.

    Per-query rows are the point of this function. The aggregate on
    `ExperimentRun` is a convenience for listing conditions; every significance
    test is recomputed from `QueryRecord`, so storing only means would make the
    `/experiments` endpoint unable to produce a p-value at all.
    """
    from geolytics.db.models import ExperimentRun, QueryRecord

    queries = {q.query_id: q for q in (query_set or [])}
    run_ids: list[int] = []

    with session_scope() as session:
        for run in runs:
            row = ExperimentRun(
                experiment=experiment,
                condition=run.condition,
                org_id=org_id or _default_org_id(session),
                site_id=site_id,
                chunker=run.chunker,
                retriever=run.retriever,
                index_stats={
                    "n_chunks": run.index_stats.n_chunks,
                    "mean_tokens": run.index_stats.mean_tokens,
                    "median_tokens": run.index_stats.median_tokens,
                    "p10_tokens": run.index_stats.p10_tokens,
                    "p90_tokens": run.index_stats.p90_tokens,
                    "total_tokens": run.index_stats.total_tokens,
                },
                projection={
                    "n_queries": run.projection.n_queries,
                    "n_projected": run.projection.n_projected,
                    "n_unmatched": run.projection.n_unmatched,
                    "n_judgments": run.projection.n_judgments,
                },
                aggregate=run.aggregate(),
                n_queries=run.n_queries_scored,
                elapsed_seconds=run.elapsed_seconds,
                notes=run.notes or None,
            )
            session.add(row)
            session.flush()
            run_ids.append(row.id)

            for query_id, metrics in run.per_query.items():
                query = queries.get(query_id)
                session.add(
                    QueryRecord(
                        run_id=row.id,
                        query_id=query_id,
                        query_text=query.text if query else "",
                        provenance=query.provenance if query else "synthetic",
                        metrics=metrics,
                        ranked_chunk_ids=run.ranked.get(query_id, []),
                        n_relevant=query.n_relevant if query else 0,
                    )
                )

    return run_ids


def _default_org_id(session: Any) -> int:
    """The organisation to attribute CLI-run experiments to.

    The CLI has no authenticated caller. Rather than invent a null tenant --
    which would create rows no API request could ever read back -- it uses the
    oldest organisation, and refuses when there is none.
    """
    org_id = session.scalar(select(Organization.id).order_by(Organization.id))
    if org_id is None:
        raise ValueError(
            "no organisation exists to attribute this run to; "
            "create one via /auth/signup or `geolytics create-org` first"
        )
    return int(org_id)


def site_documents(session: Any, crawl_id: int) -> Sequence[Document]:
    """Rehydrate `Document` objects from a stored crawl, without re-crawling."""
    from geolytics.chunking.base import Section

    pages = session.scalars(select(Page).where(Page.crawl_id == crawl_id)).all()
    return [
        Document(
            doc_id=page.doc_id,
            url=page.url,
            title=page.title or "",
            text=page.text,
            sections=tuple(
                Section(
                    text=s["text"],
                    heading_path=tuple(s.get("heading_path", ())),
                    start_char=int(s.get("start_char", 0)),
                )
                for s in (page.sections or [])
            ),
            metadata=page.extra or {},
        )
        for page in pages
    ]
