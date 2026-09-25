"""Request dependencies: database sessions, authentication and authorisation.

Tenant isolation in this system is enforced here and in the query layer, not
by the database. The rule that makes it work is simple and absolute: the
organisation is taken from the authenticated `Principal` and never from the
request. No path parameter, query string or body field may name an
organisation, because a caller controls all three.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from geolytics.config import Settings, get_settings
from geolytics.db.models import ApiKey, Membership, Organization, User
from geolytics.db.session import get_session_factory
from geolytics.security.apikeys import parse_api_key, verify_api_key
from geolytics.security.tokens import TokenError, as_utc, decode_token
from geolytics.tenancy.principal import Principal, effective_scopes, role_scopes

# auto_error=False so a missing header produces our own 401 with a WWW-
# Authenticate challenge, rather than FastAPI's bare 403.
_bearer = HTTPBearer(auto_error=False, description="JWT access token or API key")

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="authentication required",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_db() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def settings_dep() -> Settings:
    return get_settings()


DbSession = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(settings_dep)]
BearerCredentials = Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]


def _load_org(session: Session, org_id: int) -> Organization:
    org = session.get(Organization, org_id)
    if org is None or not org.is_active:
        # Same 401 as a bad credential: whether an organisation exists is not
        # something an unauthenticated caller gets to learn.
        raise _UNAUTHENTICATED
    return org


def _principal_from_api_key(session: Session, token: str) -> Principal | None:
    parsed = parse_api_key(token)
    if parsed is None:
        return None

    key = session.scalar(select(ApiKey).where(ApiKey.lookup_id == parsed.lookup_id))
    # Verify against a dummy hash when the key is unknown, so a valid lookup id
    # and an invalid one take the same time. Otherwise the response time says
    # which half of the credential was wrong.
    stored = key.secret_hash if key is not None else "0" * 64
    matched = verify_api_key(stored, parsed.secret)
    if key is None or not matched or not key.is_usable:
        raise _UNAUTHENTICATED

    org = _load_org(session, key.org_id)
    key.last_used_at = datetime.now(UTC)

    return Principal(
        org_id=org.id,
        scopes=frozenset(key.scopes or ()),
        kind="api_key",
        api_key_id=key.id,
        user_id=key.created_by_user_id,
        plan_name=org.plan,
        org_slug=org.slug,
    )


def _principal_from_jwt(session: Session, token: str, settings: Settings) -> Principal:
    try:
        payload = decode_token(token, settings.secret_key, expected_type="access")
    except TokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    if payload.org_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="token is not bound to an organisation",
        )

    # Membership and user state are re-read on every request rather than
    # trusted from the token. It costs one indexed query and makes removing a
    # member or disabling an account take effect immediately, instead of
    # whenever their access token happens to expire.
    row = session.execute(
        select(Membership, User)
        .join(User, User.id == Membership.user_id)
        .where(
            Membership.user_id == payload.user_id,
            Membership.org_id == payload.org_id,
        )
    ).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="not a member")

    membership, user = row
    if not user.is_active:
        raise _UNAUTHENTICATED
    revoked_before = as_utc(user.tokens_valid_from)
    if revoked_before is not None and payload.issued_at < revoked_before:
        # Password changed or sessions were revoked after this token was issued.
        raise _UNAUTHENTICATED

    org = _load_org(session, payload.org_id)

    return Principal(
        org_id=org.id,
        scopes=frozenset(role_scopes(membership.role)),
        kind="user",
        user_id=user.id,
        role=membership.role,
        plan_name=org.plan,
        org_slug=org.slug,
    )


def current_principal(
    request: Request,
    session: DbSession,
    settings: AppSettings,
    credentials: BearerCredentials = None,
) -> Principal:
    """Resolve the caller from an API key or a JWT access token."""
    if credentials is None or not credentials.credentials:
        raise _UNAUTHENTICATED

    token = credentials.credentials.strip()
    principal = _principal_from_api_key(session, token)
    if principal is None:
        principal = _principal_from_jwt(session, token, settings)

    request.state.principal = principal
    return principal


CurrentPrincipal = Annotated[Principal, Depends(current_principal)]


def require_scope(*scopes: str):
    """Dependency requiring every named scope.

    Returns 403, not 404: the caller is authenticated and the resource exists,
    they simply may not do this. Hiding that behind a 404 makes support
    tickets unanswerable.
    """

    def dependency(principal: CurrentPrincipal) -> Principal:
        missing = [s for s in scopes if not principal.has(s)]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"missing scope(s): {', '.join(sorted(missing))}",
            )
        return principal

    return dependency


def require_role(*roles: str):
    """Dependency requiring one of several roles. API keys never satisfy it.

    Used for actions that must be taken by a person -- inviting a member,
    changing the plan -- so that a leaked key cannot escalate itself.
    """

    def dependency(principal: CurrentPrincipal) -> Principal:
        if principal.kind != "user":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="this action requires a signed-in user, not an API key",
            )
        if principal.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"requires role: {' or '.join(roles)}",
            )
        return principal

    return dependency


def current_org(session: DbSession, principal: CurrentPrincipal) -> Organization:
    return _load_org(session, principal.org_id)


CurrentOrg = Annotated[Organization, Depends(current_org)]


def client_ip(request: Request, settings: Settings | None = None) -> str:
    """The caller's IP, trusting X-Forwarded-For only from known proxies.

    The header is trivially spoofable when the app is reachable directly, and
    believing it would let one caller exhaust another's rate limit or poison
    an audit log entry.
    """
    settings = settings or get_settings()
    peer = request.client.host if request.client else "unknown"
    if settings.trusted_proxy_ips and peer in settings.trusted_proxy_ips:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return peer


def issued_scopes_for_key(role: str, requested: list[str] | None) -> list[str]:
    """Scopes a new API key may carry, capped by its creator's role."""
    return sorted(effective_scopes(role, requested))
