"""Organisation, membership, API key and usage endpoints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from geolytics.api.deps import (
    CurrentOrg,
    DbSession,
    client_ip,
    issued_scopes_for_key,
    require_role,
    require_scope,
)
from geolytics.api.schemas_auth import (
    ApiKeyCreated,
    ApiKeyCreateRequest,
    ApiKeyOut,
    MemberInviteRequest,
    MemberOut,
    MemberRoleRequest,
    OrganizationOut,
    OrganizationUpdate,
    PlanOut,
    UsageOut,
)
from geolytics.db.models import ApiKey, AuditLogEntry, Membership, Organization, User
from geolytics.security.apikeys import generate_api_key
from geolytics.tenancy import PLANS, get_plan
from geolytics.tenancy.principal import Principal
from geolytics.tenancy.quotas import snapshot

router = APIRouter(prefix="/org", tags=["organization"])

# Annotated aliases rather than Depends(...) in argument defaults: same
# behaviour, and it keeps the dependency out of a mutable default position.
RequireOrgRead = Annotated[Principal, Depends(require_scope("org:read"))]
RequireKeysManage = Annotated[Principal, Depends(require_scope("keys:manage"))]
RequireAdmin = Annotated[Principal, Depends(require_role("owner", "admin"))]
RequireOwner = Annotated[Principal, Depends(require_role("owner"))]


def _log(
    session: DbSession,
    request: Request,
    action: str,
    org_id: int,
    actor: str,
    target: str | None = None,
    **detail: object,
) -> None:
    session.add(
        AuditLogEntry(
            org_id=org_id,
            actor=actor,
            action=action,
            target=target,
            ip_address=client_ip(request),
            user_agent=request.headers.get("user-agent", "")[:255] or None,
            detail=dict(detail),
        )
    )


def _key_out(key: ApiKey) -> ApiKeyOut:
    return ApiKeyOut(
        id=key.id,
        name=key.name,
        display_hint=key.display_hint,
        scopes=list(key.scopes or ()),
        environment=key.environment,
        created_at=key.created_at,
        last_used_at=key.last_used_at,
        expires_at=key.expires_at,
        revoked_at=key.revoked_at,
    )


@router.get("", response_model=OrganizationOut)
def read_org(org: CurrentOrg, _: RequireOrgRead) -> OrganizationOut:
    return _org_out(org)


def _org_out(org: Organization) -> OrganizationOut:
    plan = get_plan(org.plan, org.limit_overrides)
    return OrganizationOut(
        id=org.id,
        slug=org.slug,
        name=org.name,
        plan=org.plan,
        status=org.status,
        created_at=org.created_at,
        current_period_end=org.current_period_end,
        limits={
            "max_sites": plan.max_sites,
            "max_pages_per_audit": plan.max_pages_per_audit,
            "max_pages_per_month": plan.max_pages_per_month,
            "max_audits_per_month": plan.max_audits_per_month,
            "max_concurrent_audits": plan.max_concurrent_audits,
            "experiments_enabled": plan.experiments_enabled,
            "retention_days": plan.retention_days,
        },
    )


@router.patch("", response_model=OrganizationOut)
def update_org(
    payload: OrganizationUpdate,
    request: Request,
    session: DbSession,
    org: CurrentOrg,
    principal: RequireAdmin,
) -> OrganizationOut:
    if payload.name is not None:
        org.name = payload.name.strip()
    if payload.slug is not None and payload.slug != org.slug:
        clash = session.scalar(select(Organization.id).where(Organization.slug == payload.slug))
        if clash is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="slug is taken")
        org.slug = payload.slug
    _log(session, request, "org.updated", org.id, principal.audit_actor)
    session.flush()
    return _org_out(org)


@router.get("/usage", response_model=UsageOut)
def read_usage(
    session: DbSession,
    org: CurrentOrg,
    _: RequireOrgRead,
) -> UsageOut:
    return UsageOut(**snapshot(session, org).to_dict())


@router.get("/plans", response_model=list[PlanOut])
def list_plans() -> list[PlanOut]:
    """Public plan catalogue, for a pricing page."""
    return [
        PlanOut(
            name=p.name,
            display_name=p.display_name,
            monthly_price_cents=p.monthly_price_cents,
            max_sites=p.max_sites,
            max_pages_per_audit=p.max_pages_per_audit,
            max_pages_per_month=p.max_pages_per_month,
            max_audits_per_month=p.max_audits_per_month,
            max_concurrent_audits=p.max_concurrent_audits,
            experiments_enabled=p.experiments_enabled,
            support=p.support,
        )
        for p in PLANS.values()
    ]


# --- API keys ------------------------------------------------------------


@router.get("/keys", response_model=list[ApiKeyOut])
def list_keys(
    session: DbSession,
    principal: RequireKeysManage,
) -> list[ApiKeyOut]:
    keys = session.scalars(
        select(ApiKey).where(ApiKey.org_id == principal.org_id).order_by(ApiKey.id.desc())
    ).all()
    return [_key_out(k) for k in keys]


@router.post("/keys", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
def create_key(
    payload: ApiKeyCreateRequest,
    request: Request,
    session: DbSession,
    org: CurrentOrg,
    principal: RequireAdmin,
) -> ApiKeyCreated:
    """Mint an API key. The secret is returned once and never stored."""
    plan = get_plan(org.plan, org.limit_overrides)
    live = session.scalar(
        select(func.count())
        .select_from(ApiKey)
        .where(ApiKey.org_id == org.id, ApiKey.revoked_at.is_(None))
    )
    if int(live or 0) >= plan.max_api_keys:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"plan '{plan.name}' allows {plan.max_api_keys} active API keys",
        )

    # A key can never exceed its creator's own permissions, and demoting that
    # member later narrows the key too, because scopes are re-intersected on
    # every request against the role.
    scopes = issued_scopes_for_key(principal.role or "member", payload.scopes)
    if not scopes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="requested scopes are not permitted for your role",
        )

    generated = generate_api_key(payload.environment)  # type: ignore[arg-type]
    key = ApiKey(
        org_id=org.id,
        created_by_user_id=principal.user_id,
        name=payload.name.strip(),
        lookup_id=generated.lookup_id,
        secret_hash=generated.secret_hash,
        prefix=generated.prefix,
        display_hint=generated.display_hint,
        scopes=scopes,
        environment=payload.environment,
        expires_at=(
            datetime.now(UTC) + timedelta(days=payload.expires_in_days)
            if payload.expires_in_days
            else None
        ),
    )
    session.add(key)
    try:
        session.flush()
    except IntegrityError as exc:  # pragma: no cover - lookup ids are 128-bit
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="could not allocate a key, please retry",
        ) from exc

    _log(session, request, "apikey.created", org.id, principal.audit_actor, target=key.name)
    return ApiKeyCreated(**_key_out(key).model_dump(), token=generated.token)


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_key(
    key_id: int,
    request: Request,
    session: DbSession,
    principal: RequireAdmin,
) -> None:
    key = session.get(ApiKey, key_id)
    # Scoped by org: a key id from another tenant must read as absent, not as
    # forbidden, so ids cannot be probed for existence.
    if key is None or key.org_id != principal.org_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="key not found")
    if key.revoked_at is None:
        key.revoked_at = datetime.now(UTC)
        _log(
            session, request, "apikey.revoked", principal.org_id,
            principal.audit_actor, target=key.name,
        )


# --- Members -------------------------------------------------------------


@router.get("/members", response_model=list[MemberOut])
def list_members(
    session: DbSession,
    principal: RequireOrgRead,
) -> list[MemberOut]:
    rows = session.execute(
        select(Membership, User)
        .join(User, User.id == Membership.user_id)
        .where(Membership.org_id == principal.org_id)
        .order_by(Membership.id)
    ).all()
    return [
        MemberOut(
            user_id=u.id,
            email=u.email,
            full_name=u.full_name,
            role=m.role,
            joined_at=m.created_at,
        )
        for m, u in rows
    ]


@router.post("/members", response_model=MemberOut, status_code=status.HTTP_201_CREATED)
def add_member(
    payload: MemberInviteRequest,
    request: Request,
    session: DbSession,
    org: CurrentOrg,
    principal: RequireAdmin,
) -> MemberOut:
    """Add an existing user to this organisation.

    Only an existing account can be added. Creating an account on someone's
    behalf from an invite form would let anyone squat an email address, so a
    real invitation flow (emailed token, recipient sets their own password) is
    the correct shape and is listed as not yet built.
    """
    plan = get_plan(org.plan, org.limit_overrides)
    current = session.scalar(
        select(func.count()).select_from(Membership).where(Membership.org_id == org.id)
    )
    if int(current or 0) >= plan.max_members:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"plan '{plan.name}' allows {plan.max_members} members",
        )

    email = payload.email.strip().lower()
    user = session.scalar(select(User).where(func.lower(User.email) == email))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no account with that email; ask them to sign up first",
        )

    existing = session.scalar(
        select(Membership).where(
            Membership.org_id == org.id, Membership.user_id == user.id
        )
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="already a member"
        )

    # Only an owner may mint another owner; an admin promoting someone to
    # owner would be a privilege escalation.
    if payload.role == "owner" and principal.role != "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="only an owner can add an owner"
        )

    membership = Membership(user_id=user.id, org_id=org.id, role=payload.role)
    session.add(membership)
    session.flush()
    _log(
        session, request, "member.added", org.id, principal.audit_actor,
        target=user.email, role=payload.role,
    )
    return MemberOut(
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=membership.role,
        joined_at=membership.created_at,
    )


@router.patch("/members/{user_id}", response_model=MemberOut)
def change_member_role(
    user_id: int,
    payload: MemberRoleRequest,
    request: Request,
    session: DbSession,
    principal: RequireOwner,
) -> MemberOut:
    membership = session.scalar(
        select(Membership).where(
            Membership.org_id == principal.org_id, Membership.user_id == user_id
        )
    )
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not a member")

    if membership.role == "owner" and payload.role != "owner":
        _refuse_if_last_owner(session, principal.org_id)

    membership.role = payload.role
    user = session.get(User, user_id)
    _log(
        session, request, "member.role_changed", principal.org_id,
        principal.audit_actor, target=user.email if user else str(user_id), role=payload.role,
    )
    return MemberOut(
        user_id=user_id,
        email=user.email if user else "",
        full_name=user.full_name if user else None,
        role=membership.role,
        joined_at=membership.created_at,
    )


@router.delete("/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_member(
    user_id: int,
    request: Request,
    session: DbSession,
    principal: RequireAdmin,
) -> None:
    membership = session.scalar(
        select(Membership).where(
            Membership.org_id == principal.org_id, Membership.user_id == user_id
        )
    )
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not a member")
    if membership.role == "owner":
        if principal.role != "owner":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="only an owner can remove an owner"
            )
        _refuse_if_last_owner(session, principal.org_id)

    session.delete(membership)
    _log(
        session, request, "member.removed", principal.org_id,
        principal.audit_actor, target=str(user_id),
    )


def _refuse_if_last_owner(session: DbSession, org_id: int) -> None:
    """An organisation with no owner can never be administered again."""
    owners = session.scalar(
        select(func.count())
        .select_from(Membership)
        .where(Membership.org_id == org_id, Membership.role == "owner")
    )
    if int(owners or 0) <= 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="an organisation must keep at least one owner",
        )
