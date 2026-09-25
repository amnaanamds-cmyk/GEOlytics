"""Audit endpoints, scoped to the caller's organisation."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select

from geolytics.api.deps import CurrentOrg, DbSession, require_scope
from geolytics.api.schemas import AuditAccepted, AuditListOut, AuditOut, AuditRequest, PageScoreOut
from geolytics.crawl.guard import UrlPolicy, UrlPolicyError, validate_url
from geolytics.db.models import Audit, Page, PageScore, Site
from geolytics.tenancy.principal import Principal
from geolytics.tenancy.quotas import (
    AUDITS_RUN,
    QuotaExceeded,
    check_audit_allowed,
    check_site_allowed,
    record_usage,
)

router = APIRouter(prefix="/audits", tags=["audits"])

RequireRead = Annotated[Principal, Depends(require_scope("audits:read"))]
RequireWrite = Annotated[Principal, Depends(require_scope("audits:write"))]

UNFITTED_CAVEAT = (
    "This score was computed with placeholder (unfitted) signal weights. "
    "It ranks pages within this site consistently, but the absolute value "
    "carries no validated meaning. Fit weights against measured visibility "
    "before reporting it."
)


def _quota_error(exc: QuotaExceeded) -> HTTPException:
    """402 rather than 429: the caller is not going too fast, they are out of plan."""
    return HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail={
            "error": "quota_exceeded",
            "limit": exc.limit_name,
            "allowed": exc.limit,
            "used": exc.used,
            "message": str(exc),
        },
    )


@router.post("", response_model=AuditAccepted, status_code=status.HTTP_202_ACCEPTED)
def create_audit(
    payload: AuditRequest,
    request: Request,
    session: DbSession,
    org: CurrentOrg,
    principal: RequireWrite,
) -> AuditAccepted:
    """Queue an audit for this organisation.

    The URL is validated against the SSRF policy here, before anything is
    written, so a customer gets an immediate 422 rather than an audit row that
    fails minutes later inside a worker.
    """
    from geolytics.tasks.queue import enqueue_audit

    url = str(payload.url)
    try:
        validate_url(url, UrlPolicy.from_settings(request.app.state.settings))
    except UrlPolicyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"error": "url_not_allowed", "reason": exc.reason, "message": exc.detail},
        ) from exc

    site = session.scalar(
        select(Site).where(Site.org_id == org.id, Site.url == url)
    )
    if site is None:
        try:
            check_site_allowed(session, org)
        except QuotaExceeded as exc:
            raise _quota_error(exc) from exc
        site = Site(org_id=org.id, url=url, host=urlparse(url).netloc.lower())
        session.add(site)
        session.flush()

    try:
        page_budget = check_audit_allowed(session, org, payload.max_pages)
    except QuotaExceeded as exc:
        raise _quota_error(exc) from exc

    config = payload.model_dump(mode="json")
    # The plan's per-audit ceiling wins over whatever was requested, so the
    # worker cannot be told to crawl more than the customer bought.
    config["max_pages"] = page_budget

    audit = Audit(org_id=org.id, site_id=site.id, status="pending", config=config)
    session.add(audit)
    session.flush()

    # Counted at submission, not completion: a customer must not be able to
    # queue a thousand audits and have them all pass the check.
    record_usage(session, org.id, AUDITS_RUN, 1)
    audit_id = audit.id
    session.commit()

    enqueue_audit(audit_id)
    message = f"Audit queued for {url}"
    if page_budget < payload.max_pages:
        message += (
            f" (limited to {page_budget} pages by your plan; {payload.max_pages} requested)"
        )
    return AuditAccepted(audit_id=audit_id, status="pending", message=message)


@router.get("", response_model=AuditListOut)
def list_audits(
    session: DbSession,
    principal: RequireRead,
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> AuditListOut:
    rows = session.execute(
        select(Audit, Site)
        .join(Site, Site.id == Audit.site_id)
        .where(Audit.org_id == principal.org_id)
        .order_by(Audit.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return AuditListOut(
        items=[
            AuditOut(
                audit_id=a.id,
                site_url=s.url,
                status=a.status,
                overall_score=a.overall_score,
                weights_fitted=a.weights_fitted,
                score_caveat=None if a.weights_fitted else UNFITTED_CAVEAT,
                pages=[],
                summary=a.summary or {},
                created_at=a.created_at,
            )
            for a, s in rows
        ],
        limit=limit,
        offset=offset,
    )


@router.get("/{audit_id}", response_model=AuditOut)
def get_audit(audit_id: int, session: DbSession, principal: RequireRead) -> AuditOut:
    # Scoped in the WHERE clause, not checked after the fetch. Another
    # tenant's id must read as absent so ids cannot be probed for existence.
    audit = session.scalar(
        select(Audit).where(Audit.id == audit_id, Audit.org_id == principal.org_id)
    )
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


@router.delete("/{audit_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_audit(audit_id: int, session: DbSession, principal: RequireWrite) -> None:
    audit = session.scalar(
        select(Audit).where(Audit.id == audit_id, Audit.org_id == principal.org_id)
    )
    if audit is None:
        raise HTTPException(status_code=404, detail=f"audit {audit_id} not found")
    session.delete(audit)
