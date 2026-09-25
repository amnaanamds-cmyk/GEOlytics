"""Who is making a request, and what they are allowed to do.

A `Principal` is produced by the auth dependencies and carries the
organisation every query must be scoped to. Nothing downstream reads an
organisation id from the request body or a path parameter -- that is how
cross-tenant reads happen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["owner", "admin", "member", "viewer"]
Scope = Literal[
    "audits:read",
    "audits:write",
    "experiments:read",
    "experiments:write",
    "sites:read",
    "sites:write",
    "org:read",
    "org:write",
    "keys:manage",
    "billing:manage",
]

ALL_SCOPES: tuple[Scope, ...] = (
    "audits:read",
    "audits:write",
    "experiments:read",
    "experiments:write",
    "sites:read",
    "sites:write",
    "org:read",
    "org:write",
    "keys:manage",
    "billing:manage",
)

_READ_SCOPES: tuple[Scope, ...] = (
    "audits:read",
    "experiments:read",
    "sites:read",
    "org:read",
)

_ROLE_SCOPES: dict[Role, tuple[Scope, ...]] = {
    "owner": ALL_SCOPES,
    "admin": tuple(s for s in ALL_SCOPES if s != "billing:manage"),
    "member": (
        *_READ_SCOPES,
        "audits:write",
        "experiments:write",
        "sites:write",
    ),
    "viewer": _READ_SCOPES,
}


def role_scopes(role: Role) -> tuple[Scope, ...]:
    """Scopes a role grants. An unknown role grants nothing."""
    return _ROLE_SCOPES.get(role, ())


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated caller, already bound to one organisation."""

    org_id: int
    scopes: frozenset[str]
    kind: Literal["user", "api_key"]
    user_id: int | None = None
    api_key_id: int | None = None
    role: Role | None = None
    plan_name: str = "free"
    org_slug: str = ""

    def has(self, scope: str) -> bool:
        return scope in self.scopes

    def require(self, scope: str) -> None:
        if not self.has(scope):
            raise PermissionError(f"missing required scope {scope!r}")

    @property
    def audit_actor(self) -> str:
        """Short identifier for audit logs."""
        if self.kind == "api_key":
            return f"api_key:{self.api_key_id}"
        return f"user:{self.user_id}"


def effective_scopes(role: Role, granted: list[str] | tuple[str, ...] | None) -> frozenset[str]:
    """Intersect a key's requested scopes with what its creator's role allows.

    An API key must never be able to do more than the member who created it.
    Intersecting rather than trusting the stored list also means that demoting
    a member immediately narrows the keys they made.
    """
    allowed = set(role_scopes(role))
    if granted is None:
        return frozenset(allowed)
    return frozenset(allowed & set(granted))
