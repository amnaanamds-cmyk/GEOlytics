"""Relational schema.

Three kinds of data live here and they are kept separate on purpose:

* **Tenancy** (Organization, User, Membership, ApiKey, UsageCounter,
  AuditLogEntry) -- who the customer is and what they may do. Every
  customer-owned row below carries `org_id`, and every query is scoped by it.
  Tenant isolation in this system is enforced in the query layer, so the rule
  is absolute: no endpoint takes an organisation id from the request.


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

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
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


class Organization(Base, TimestampMixin):
    """A tenant. Every customer-owned row hangs off one of these."""

    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    plan: Mapped[str] = mapped_column(String(32), default="free", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False)
    # Bespoke limits for one customer, so a negotiated deal does not require a
    # new plan tier in code. Keys that are not real limits are ignored.
    limit_overrides: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    billing_customer_id: Mapped[str | None] = mapped_column(String(255), index=True)
    billing_subscription_id: Mapped[str | None] = mapped_column(String(255))
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    memberships: Mapped[list[Membership]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )
    api_keys: Mapped[list[ApiKey]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )
    sites: Mapped[list[Site]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )

    @property
    def is_active(self) -> bool:
        return self.status == "active" and self.deleted_at is None


class User(Base, TimestampMixin):
    """A person. Users are global; access comes from Membership."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Stored lowercased so "A@b.com" and "a@b.com" cannot become two accounts.
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Bumped on password change or forced logout; refresh tokens issued before
    # this instant are rejected, which is how a session is revoked without
    # keeping a server-side session table.
    tokens_valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    memberships: Mapped[list[Membership]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Membership(Base, TimestampMixin):
    """A user's role in one organisation."""

    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("user_id", "org_id", name="uq_membership_user_org"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    org_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(32), default="member", nullable=False)

    user: Mapped[User] = relationship(back_populates="memberships")
    organization: Mapped[Organization] = relationship(back_populates="memberships")


class ApiKey(Base, TimestampMixin):
    """A machine credential scoped to one organisation.

    Only the hash of the secret is stored. `lookup_id` is the indexed public
    half, so verifying a key is one row fetch rather than a scan.
    """

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    lookup_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    secret_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    display_hint: Mapped[str] = mapped_column(String(64), nullable=False)
    scopes: Mapped[list[str]] = mapped_column(JSON, default=list)
    environment: Mapped[str] = mapped_column(String(8), default="live", nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    organization: Mapped[Organization] = relationship(back_populates="api_keys")

    @property
    def is_usable(self) -> bool:
        if self.revoked_at is not None:
            return False
        if self.expires_at is None:
            return True
        # Not every backend returns the offset it was given (SQLite does not),
        # and comparing a naive value against an aware one raises. Everything
        # written here is UTC, so a missing offset means UTC.
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        return expires > datetime.now(UTC)


class UsageCounter(Base, TimestampMixin):
    """Metered usage for one organisation in one billing period.

    Kept as a counter row per (org, period, metric) rather than derived from
    the audit tables, because retention deletes old audits while billing
    history must survive them.
    """

    __tablename__ = "usage_counters"
    __table_args__ = (
        UniqueConstraint("org_id", "period", "metric", name="uq_usage_org_period_metric"),
        Index("ix_usage_org_period", "org_id", "period"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    period: Mapped[str] = mapped_column(String(7), nullable=False)  # YYYY-MM
    metric: Mapped[str] = mapped_column(String(48), nullable=False)
    value: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class AuditLogEntry(Base, TimestampMixin):
    """Security-relevant actions, for answering "who did that" after the fact."""

    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_log_org_created", "org_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    target: Mapped[str | None] = mapped_column(String(255))
    ip_address: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(255))
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Site(Base, TimestampMixin):
    __tablename__ = "sites"
    __table_args__ = (UniqueConstraint("org_id", "url", name="uq_site_org_url"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    # Sites are per-organisation: two customers auditing the same public URL
    # are two independent rows, and neither can see the other's.
    org_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    host: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(255))

    organization: Mapped[Organization] = relationship(back_populates="sites")
    crawls: Mapped[list[Crawl]] = relationship(back_populates="site", cascade="all, delete-orphan")
    audits: Mapped[list[Audit]] = relationship(back_populates="site", cascade="all, delete-orphan")


class Crawl(Base, TimestampMixin):
    __tablename__ = "crawls"

    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
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
    # Denormalised from Site so that every scoped query filters on one indexed
    # column without a join. A missed join is how cross-tenant reads happen.
    org_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
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
    org_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
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
