"""Request and response models for authentication, organisations and billing."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field, field_validator

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=1024)
    full_name: str | None = Field(default=None, max_length=255)
    organization_name: str = Field(min_length=2, max_length=255)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)
    # Optional: a user in several organisations picks one at login.
    organization_slug: str | None = None


class RefreshRequest(BaseModel):
    refresh_token: str
    organization_slug: str | None = None


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    organization: OrganizationSummary


class OrganizationSummary(BaseModel):
    id: int
    slug: str
    name: str
    plan: str
    role: str | None = None


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str | None
    is_active: bool
    is_verified: bool
    created_at: datetime | None = None
    organizations: list[OrganizationSummary] = []


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=1024)
    new_password: str = Field(min_length=12, max_length=1024)


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    scopes: list[str] | None = None
    environment: str = Field(default="live", pattern="^(live|test)$")
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)


class ApiKeyOut(BaseModel):
    id: int
    name: str
    display_hint: str
    scopes: list[str]
    environment: str
    created_at: datetime | None = None
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None


class ApiKeyCreated(ApiKeyOut):
    # Returned exactly once. Only its hash is stored, so it cannot be shown again.
    token: str
    warning: str = "Store this key now. It cannot be retrieved again."


class MemberOut(BaseModel):
    user_id: int
    email: str
    full_name: str | None
    role: str
    joined_at: datetime | None = None


class MemberInviteRequest(BaseModel):
    email: EmailStr
    role: str = Field(default="member", pattern="^(owner|admin|member|viewer)$")


class MemberRoleRequest(BaseModel):
    role: str = Field(pattern="^(owner|admin|member|viewer)$")


class UsageOut(BaseModel):
    period: str
    plan: str
    used: dict[str, int]
    limits: dict[str, int]
    remaining: dict[str, int]
    sites: dict[str, int]
    concurrent_audits: dict[str, int]


class PlanOut(BaseModel):
    name: str
    display_name: str
    monthly_price_cents: int
    max_sites: int
    max_pages_per_audit: int
    max_pages_per_month: int
    max_audits_per_month: int
    max_concurrent_audits: int
    experiments_enabled: bool
    support: str


class OrganizationOut(BaseModel):
    id: int
    slug: str
    name: str
    plan: str
    status: str
    created_at: datetime | None = None
    current_period_end: datetime | None = None
    limits: dict[str, Any] = {}


class OrganizationUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=255)
    slug: str | None = None

    @field_validator("slug")
    @classmethod
    def _valid_slug(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower()
        if not _SLUG_RE.match(value):
            raise ValueError(
                "slug must be 3-64 characters of lowercase letters, digits and hyphens, "
                "starting and ending alphanumeric"
            )
        return value


class CheckoutRequest(BaseModel):
    plan: str = Field(pattern="^(starter|growth)$")
    success_url: str
    cancel_url: str


class CheckoutSession(BaseModel):
    url: str
    session_id: str


TokenPair.model_rebuild()
