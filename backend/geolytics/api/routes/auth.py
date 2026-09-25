"""Signup, login, token refresh and account endpoints."""

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from geolytics.api.deps import (
    AppSettings,
    CurrentPrincipal,
    DbSession,
    client_ip,
)
from geolytics.api.schemas_auth import (
    ChangePasswordRequest,
    LoginRequest,
    OrganizationSummary,
    RefreshRequest,
    SignupRequest,
    TokenPair,
    UserOut,
)
from geolytics.db.models import AuditLogEntry, Membership, Organization, User
from geolytics.security.passwords import (
    PasswordPolicyError,
    hash_password,
    needs_rehash,
    verify_password,
)
from geolytics.security.tokens import (
    TokenError,
    as_utc,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from geolytics.tenancy.principal import role_scopes

router = APIRouter(prefix="/auth", tags=["auth"])

# One message for every failure on the login path. Distinguishing "no such
# user" from "wrong password" turns the endpoint into an account enumerator.
_BAD_CREDENTIALS = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="invalid email or password",
    headers={"WWW-Authenticate": "Bearer"},
)

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    base = _SLUG_STRIP.sub("-", name.strip().lower()).strip("-")[:48] or "org"
    if len(base) < 3:
        base = f"{base}-org"
    return base


def _unique_slug(session: DbSession, name: str) -> str:
    base = _slugify(name)
    for _ in range(8):
        candidate = base if _ == 0 else f"{base}-{secrets.token_hex(3)}"
        if session.scalar(select(Organization.id).where(Organization.slug == candidate)) is None:
            return candidate
    return f"{base}-{secrets.token_hex(6)}"


def _log(
    session: DbSession,
    request: Request,
    action: str,
    org_id: int | None,
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


def _summaries(session: DbSession, user_id: int) -> list[OrganizationSummary]:
    rows = session.execute(
        select(Organization, Membership.role)
        .join(Membership, Membership.org_id == Organization.id)
        .where(Membership.user_id == user_id)
        .order_by(Organization.name)
    ).all()
    return [
        OrganizationSummary(id=o.id, slug=o.slug, name=o.name, plan=o.plan, role=role)
        for o, role in rows
    ]


def _issue(
    session: DbSession,
    settings: AppSettings,
    user: User,
    org: Organization,
    role: str,
) -> TokenPair:
    access = create_access_token(
        user.id,
        settings.secret_key,
        org_id=org.id,
        scopes=role_scopes(role),
        expires_in=timedelta(minutes=settings.access_token_minutes),
    )
    refresh = create_refresh_token(
        user.id, settings.secret_key, expires_in=timedelta(days=settings.refresh_token_days)
    )
    return TokenPair(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.access_token_minutes * 60,
        organization=OrganizationSummary(
            id=org.id, slug=org.slug, name=org.name, plan=org.plan, role=role
        ),
    )


def _select_org(
    session: DbSession, user_id: int, slug: str | None
) -> tuple[Organization, str]:
    query = (
        select(Organization, Membership.role)
        .join(Membership, Membership.org_id == Organization.id)
        .where(Membership.user_id == user_id)
    )
    if slug:
        query = query.where(Organization.slug == slug)
    row = session.execute(query.order_by(Membership.id)).first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="no organisation available for this account",
        )
    org, role = row
    if not org.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="organisation is not active"
        )
    return org, role


@router.post("/signup", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
def signup(
    payload: SignupRequest,
    request: Request,
    session: DbSession,
    settings: AppSettings,
) -> TokenPair:
    """Create a user and their first organisation, and sign them in."""
    if not settings.signup_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="signup is disabled"
        )

    email = payload.email.strip().lower()
    try:
        password_hash = hash_password(payload.password)
    except PasswordPolicyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    # Case-insensitive, because "A@b.com" and "a@b.com" are one mailbox and
    # must not become two accounts.
    existing = session.scalar(select(User).where(func.lower(User.email) == email))
    if existing is not None:
        # Deliberately the same shape of answer as success would give an
        # attacker probing for registered addresses: a generic conflict.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="could not create the account with those details",
        )

    user = User(email=email, password_hash=password_hash, full_name=payload.full_name)
    session.add(user)
    org = Organization(
        slug=_unique_slug(session, payload.organization_name),
        name=payload.organization_name.strip(),
        plan="free",
    )
    session.add(org)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="could not create the account with those details",
        ) from exc

    session.add(Membership(user_id=user.id, org_id=org.id, role="owner"))
    session.flush()

    _log(session, request, "auth.signup", org.id, f"user:{user.id}", target=org.slug)
    return _issue(session, settings, user, org, "owner")


@router.post("/login", response_model=TokenPair)
def login(
    payload: LoginRequest,
    request: Request,
    session: DbSession,
    settings: AppSettings,
) -> TokenPair:
    email = payload.email.strip().lower()
    user = session.scalar(select(User).where(func.lower(User.email) == email))

    # Hash a dummy password when the user does not exist, so a missing account
    # and a wrong password take the same time. Argon2 is slow by design, which
    # would otherwise make the difference obvious.
    stored = user.password_hash if user else _DUMMY_HASH
    matched = verify_password(stored, payload.password)
    if user is None or not matched or not user.is_active:
        raise _BAD_CREDENTIALS

    if needs_rehash(user.password_hash):
        # Transparent upgrade when the cost parameters have been raised.
        user.password_hash = hash_password(payload.password)

    org, role = _select_org(session, user.id, payload.organization_slug)
    user.last_login_at = datetime.now(UTC)
    _log(session, request, "auth.login", org.id, f"user:{user.id}")
    return _issue(session, settings, user, org, role)


@router.post("/refresh", response_model=TokenPair)
def refresh(
    payload: RefreshRequest,
    session: DbSession,
    settings: AppSettings,
) -> TokenPair:
    """Exchange a refresh token for a new pair.

    This is where revocation is enforced. Access tokens are deliberately not
    checked against a blocklist on every request; instead a disabled account
    or a password change stops the refresh, so a revoked session survives at
    most one access-token lifetime.
    """
    try:
        claims = decode_token(payload.refresh_token, settings.secret_key, expected_type="refresh")
    except TokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=f"invalid refresh token: {exc}"
        ) from exc

    user = session.get(User, claims.user_id)
    if user is None or not user.is_active:
        raise _BAD_CREDENTIALS
    revoked_before = as_utc(user.tokens_valid_from)
    if revoked_before is not None and claims.issued_at < revoked_before:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="session has been revoked"
        )

    org, role = _select_org(session, user.id, payload.organization_slug)
    return _issue(session, settings, user, org, role)


@router.get("/me", response_model=UserOut)
def me(session: DbSession, principal: CurrentPrincipal) -> UserOut:
    if principal.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="this endpoint is for user sessions, not API keys",
        )
    user = session.get(User, principal.user_id)
    if user is None:
        raise _BAD_CREDENTIALS
    return UserOut(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_verified=user.is_verified,
        created_at=user.created_at,
        organizations=_summaries(session, user.id),
    )


@router.post("/password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    session: DbSession,
    principal: CurrentPrincipal,
) -> None:
    """Change the password and invalidate every existing session."""
    if principal.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="requires a user session"
        )
    user = session.get(User, principal.user_id)
    if user is None or not verify_password(user.password_hash, payload.current_password):
        raise _BAD_CREDENTIALS

    try:
        user.password_hash = hash_password(payload.new_password)
    except PasswordPolicyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    # Everything issued before now stops working, so a stolen session does not
    # survive the password change that was meant to end it.
    user.tokens_valid_from = datetime.now(UTC)
    _log(session, request, "auth.password_changed", principal.org_id, principal.audit_actor)


# A real Argon2 hash of a value nobody knows, used to equalise login timing.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(32))
