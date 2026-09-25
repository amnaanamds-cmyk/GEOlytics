"""Short-lived JWT access tokens and longer-lived refresh tokens.

Design decisions worth stating:

* **Access tokens are short-lived and not revocable.** That is the trade for
  statelessness: no database round trip on every request. 15 minutes bounds
  the damage from a leaked token. Anything that must take effect immediately
  -- a disabled user, a revoked key -- is checked against the database on the
  refresh path, not the access path.
* **Refresh tokens carry a `jti`** so a session can be revoked server-side,
  and are typed, so an access token cannot be presented as a refresh token or
  the reverse.
* **The signing secret must be set explicitly in production.** `Settings`
  refuses to start otherwise, because a default secret means anyone holding
  the source can mint a token for any organisation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import jwt

ALGORITHM = "HS256"
TokenType = Literal["access", "refresh"]


class TokenError(Exception):
    """A token was missing, malformed, expired or of the wrong type."""


@dataclass(frozen=True, slots=True)
class TokenPayload:
    """The verified claims of a token."""

    subject: str
    token_type: TokenType
    org_id: int | None
    scopes: tuple[str, ...]
    jti: str
    issued_at: datetime
    expires_at: datetime

    @property
    def user_id(self) -> int:
        return int(self.subject)


def _create(
    subject: str | int,
    secret: str,
    token_type: TokenType,
    expires_in: timedelta,
    org_id: int | None = None,
    scopes: tuple[str, ...] = (),
    issuer: str = "geolytics",
) -> str:
    now = datetime.now(UTC)
    claims: dict[str, Any] = {
        "sub": str(subject),
        "typ": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_in).timestamp()),
        "nbf": int(now.timestamp()),
        "iss": issuer,
        "jti": uuid.uuid4().hex,
    }
    if org_id is not None:
        claims["org"] = org_id
    if scopes:
        claims["scp"] = list(scopes)
    return jwt.encode(claims, secret, algorithm=ALGORITHM)


def create_access_token(
    user_id: int,
    secret: str,
    org_id: int | None = None,
    scopes: tuple[str, ...] = (),
    expires_in: timedelta = timedelta(minutes=15),
    issuer: str = "geolytics",
) -> str:
    return _create(user_id, secret, "access", expires_in, org_id, scopes, issuer)


def create_refresh_token(
    user_id: int,
    secret: str,
    expires_in: timedelta = timedelta(days=30),
    issuer: str = "geolytics",
) -> str:
    return _create(user_id, secret, "refresh", expires_in, None, (), issuer)


def as_utc(value: datetime | None) -> datetime | None:
    """Normalise a datetime to UTC-aware.

    Columns are declared `DateTime(timezone=True)`, but not every backend
    stores the offset -- SQLite hands back a naive value -- so a stored
    timestamp compared against a token claim can raise
    "can't compare offset-naive and offset-aware datetimes". Since every
    timestamp this system writes is UTC, a missing offset means UTC.
    """
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def decode_token(
    token: str,
    secret: str,
    expected_type: TokenType | None = None,
    issuer: str = "geolytics",
) -> TokenPayload:
    """Verify and decode a token.

    `algorithms` is pinned to a single value on purpose: accepting a list that
    includes "none", or allowing the token's own header to choose, is the
    classic JWT forgery. Every failure mode is collapsed into `TokenError` so
    a caller cannot branch on why a token was rejected.
    """
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            issuer=issuer,
            options={"require": ["exp", "iat", "sub", "typ", "jti"]},
        )
    except jwt.PyJWTError as exc:
        raise TokenError(str(exc)) from exc

    token_type = claims.get("typ")
    if token_type not in ("access", "refresh"):
        raise TokenError(f"unknown token type {token_type!r}")
    if expected_type is not None and token_type != expected_type:
        raise TokenError(f"expected a {expected_type} token, got {token_type}")

    return TokenPayload(
        subject=str(claims["sub"]),
        token_type=token_type,
        org_id=claims.get("org"),
        scopes=tuple(claims.get("scp") or ()),
        jti=str(claims["jti"]),
        issued_at=datetime.fromtimestamp(claims["iat"], tz=UTC),
        expires_at=datetime.fromtimestamp(claims["exp"], tz=UTC),
    )
