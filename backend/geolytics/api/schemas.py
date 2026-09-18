"""Request and response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator


class AuditRequest(BaseModel):
    url: HttpUrl
    max_pages: int = Field(default=25, ge=1, le=500)
    chunker: str = Field(default="sentence")
    retriever: str = Field(default="hybrid")
    generate_queries: bool = True
    questions_per_unit: int = Field(default=2, ge=1, le=5)

    @field_validator("chunker")
    @classmethod
    def _known_chunker(cls, value: str) -> str:
        from geolytics.chunking.registry import CHUNKERS

        if value not in CHUNKERS:
            raise ValueError(f"chunker must be one of {CHUNKERS}")
        return value

    @field_validator("retriever")
    @classmethod
    def _known_retriever(cls, value: str) -> str:
        allowed = ("dense", "bm25", "hybrid", "reranked")
        if value not in allowed:
            raise ValueError(f"retriever must be one of {allowed}")
        return value


class AuditAccepted(BaseModel):
    audit_id: int
    status: str
    message: str


class PageScoreOut(BaseModel):
    url: str
    title: str | None = None
    score: float
    signals: dict[str, float]
    contributions: dict[str, float]
    recommendations: list[str]


class AuditOut(BaseModel):
    audit_id: int
    site_url: str
    status: str
    overall_score: float | None
    weights_fitted: bool
    # Surfaced at the top level, not buried in a footnote: a score computed
    # from unfitted weights is a placeholder, and the dashboard must say so.
    score_caveat: str | None = None
    pages: list[PageScoreOut] = []
    summary: dict[str, Any] = {}
    created_at: datetime | None = None


class ExperimentRunOut(BaseModel):
    condition: str
    chunker: dict[str, Any]
    retriever: dict[str, Any]
    index_stats: dict[str, Any]
    aggregate: dict[str, float]
    n_queries: int


class ComparisonOut(BaseModel):
    name_a: str
    name_b: str
    metric: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adjusted: float | None
    effect_size: float
    effect_label: str
    significant: bool


class ExperimentOut(BaseModel):
    experiment: str
    metric: str
    runs: list[ExperimentRunOut]
    comparisons: list[ComparisonOut]
    methods_note: str


class HealthOut(BaseModel):
    status: str
    version: str
    embedding_backend: str
    services: dict[str, str]
