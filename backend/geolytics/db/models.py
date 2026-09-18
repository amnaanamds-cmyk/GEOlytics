"""Relational schema.

Two kinds of data live here and they are kept separate on purpose:

* **Audit data** (Site, Crawl, Page, Audit, PageScore) -- what the application
  shows a user about one website.
* **Experiment data** (ExperimentRun, QueryRecord) -- the research record.
  `QueryRecord` stores one row per (run, query): the per-query metric scores
  that every significance test in the report is computed from. Storing only
  aggregate means here would make the statistics unreproducible after the fact.

Chunks are deliberately *not* a table. They are derived from a page plus a
chunking configuration, they multiply by the number of strategies under test,
and the vectors live in Qdrant. Persisting them relationally would duplicate
the index with no query that needs it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Site(Base, TimestampMixin):
    __tablename__ = "sites"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True)
    host: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(255))

    crawls: Mapped[list[Crawl]] = relationship(back_populates="site", cascade="all, delete-orphan")
    audits: Mapped[list[Audit]] = relationship(back_populates="site", cascade="all, delete-orphan")


class Crawl(Base, TimestampMixin):
    __tablename__ = "crawls"

    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    pages_fetched: Mapped[int] = mapped_column(Integer, default=0)
    pages_skipped_robots: Mapped[int] = mapped_column(Integer, default=0)
    user_agent: Mapped[str] = mapped_column(String(255), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)

    site: Mapped[Site] = relationship(back_populates="crawls")
    pages: Mapped[list[Page]] = relationship(back_populates="crawl", cascade="all, delete-orphan")


class Page(Base, TimestampMixin):
    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("crawl_id", "url", name="uq_page_crawl_url"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    crawl_id: Mapped[int] = mapped_column(ForeignKey("crawls.id", ondelete="CASCADE"), index=True)
    doc_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    title: Mapped[str | None] = mapped_column(String(1024))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Heading structure, as extracted. Kept so chunking can be re-run later
    # without re-crawling.
    sections: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    crawl: Mapped[Crawl] = relationship(back_populates="pages")
    scores: Mapped[list[PageScore]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )


class Audit(Base, TimestampMixin):
    __tablename__ = "audits"

    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    crawl_id: Mapped[int | None] = mapped_column(ForeignKey("crawls.id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    overall_score: Mapped[float | None] = mapped_column(Float)
    # Whether the weights behind overall_score were fitted or placeholders. A
    # score from unfitted weights must never be presented as a finding.
    weights_fitted: Mapped[bool] = mapped_column(default=False, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)

    site: Mapped[Site] = relationship(back_populates="audits")
    scores: Mapped[list[PageScore]] = relationship(
        back_populates="audit", cascade="all, delete-orphan"
    )


class PageScore(Base, TimestampMixin):
    __tablename__ = "page_scores"
    __table_args__ = (UniqueConstraint("audit_id", "page_id", name="uq_score_audit_page"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    audit_id: Mapped[int] = mapped_column(ForeignKey("audits.id", ondelete="CASCADE"), index=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id", ondelete="CASCADE"), index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    signals: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    contributions: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    recommendations: Mapped[list[str]] = mapped_column(JSON, default=list)

    audit: Mapped[Audit] = relationship(back_populates="scores")
    page: Mapped[Page] = relationship(back_populates="scores")


class ExperimentRun(Base, TimestampMixin):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    condition: Mapped[str] = mapped_column(String(128), nullable=False)
    site_id: Mapped[int | None] = mapped_column(ForeignKey("sites.id", ondelete="SET NULL"))
    chunker: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    retriever: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    index_stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    projection: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    aggregate: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    n_queries: Mapped[int] = mapped_column(Integer, default=0)
    elapsed_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[str | None] = mapped_column(Text)

    queries: Mapped[list[QueryRecord]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class QueryRecord(Base):
    """One query's scores under one run -- the unit the paired tests consume."""

    __tablename__ = "query_records"
    __table_args__ = (
        UniqueConstraint("run_id", "query_id", name="uq_query_run"),
        Index("ix_query_records_run_query", "run_id", "query_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_runs.id", ondelete="CASCADE"), index=True
    )
    query_id: Mapped[str] = mapped_column(String(64), nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    provenance: Mapped[str] = mapped_column(String(16), default="synthetic", nullable=False)
    metrics: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    ranked_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    n_relevant: Mapped[int] = mapped_column(Integer, default=0)

    run: Mapped[ExperimentRun] = relationship(back_populates="queries")
