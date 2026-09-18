"""Audit endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from geolytics.api.schemas import AuditAccepted, AuditOut, AuditRequest, PageScoreOut
from geolytics.db.models import Audit, Page, PageScore, Site
from geolytics.db.session import session_scope

router = APIRouter(prefix="/audits", tags=["audits"])

UNFITTED_CAVEAT = (
    "This score was computed with placeholder (unfitted) signal weights. "
    "It ranks pages within this site consistently, but the absolute value "
    "carries no validated meaning. Fit weights against measured visibility "
    "before reporting it."
)


@router.post("", response_model=AuditAccepted, status_code=status.HTTP_202_ACCEPTED)
def create_audit(request: AuditRequest) -> AuditAccepted:
    """Queue an audit. Crawling is slow, so the work happens on the worker."""
    from geolytics.tasks.queue import enqueue_audit

    with session_scope() as session:
        url = str(request.url)
        site = session.scalar(select(Site).where(Site.url == url))
        if site is None:
            from urllib.parse import urlparse

            site = Site(url=url, host=urlparse(url).netloc.lower())
            session.add(site)
            session.flush()

        audit = Audit(site_id=site.id, status="pending", config=request.model_dump(mode="json"))
        session.add(audit)
        session.flush()
        audit_id = audit.id

    enqueue_audit(audit_id)
    return AuditAccepted(
        audit_id=audit_id,
        status="pending",
        message=f"Audit queued for {request.url}",
    )


@router.get("/{audit_id}", response_model=AuditOut)
def get_audit(audit_id: int) -> AuditOut:
    with session_scope() as session:
        audit = session.get(Audit, audit_id)
        if audit is None:
            raise HTTPException(status_code=404, detail=f"audit {audit_id} not found")

        site = session.get(Site, audit.site_id)
        rows = session.execute(
            select(PageScore, Page)
            .join(Page, Page.id == PageScore.page_id)
            .where(PageScore.audit_id == audit_id)
            .order_by(PageScore.score.desc())
        ).all()

        return AuditOut(
            audit_id=audit.id,
            site_url=site.url if site else "",
            status=audit.status,
            overall_score=audit.overall_score,
            weights_fitted=audit.weights_fitted,
            score_caveat=None if audit.weights_fitted else UNFITTED_CAVEAT,
            pages=[
                PageScoreOut(
                    url=page.url,
                    title=page.title,
                    score=score.score,
                    signals=score.signals,
                    contributions=score.contributions,
                    recommendations=score.recommendations,
                )
                for score, page in rows
            ],
            summary=audit.summary or {},
            created_at=audit.created_at,
        )
