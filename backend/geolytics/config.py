"""Application settings, loaded from environment / .env.

Every knob that affects an experiment lives here so that a run can record the
exact configuration it executed under (see `evaluation.harness.RunConfig`).
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Long enough that HMAC-SHA256 gets a full-strength key (RFC 7518 3.2).
MIN_SECRET_LENGTH = 32

# Placeholders that must never reach production. A deployment that ships with
# one of these is one where anyone holding the source can mint a valid token
# for any organisation.
_UNSAFE_SECRETS = frozenset(
    {
        "change-me", "changeme", "secret", "dev", "development", "test",
        "please-change", "insecure", "geolytics",
    }
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GEOLYTICS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    service_name: str = "geolytics-api"

    # --- Security ---------------------------------------------------------
    # Signs access and refresh tokens. Generated per-process when unset, which
    # is fine for a dev restart and fatal for a multi-replica deployment: each
    # replica would sign with a different key and reject the others' tokens.
    # Production therefore refuses to start without an explicit value.
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(48))
    access_token_minutes: int = 15
    refresh_token_days: int = 30
    # Browser origins allowed to call the API. Never "*" in production: the API
    # is credentialed, and a wildcard would let any site spend a logged-in
    # customer's quota.
    cors_origins: list[str] = ["http://localhost:3000"]
    signup_enabled: bool = True
    # Trusted reverse proxies, for reading the client IP from X-Forwarded-For.
    # Empty means the header is ignored -- it is trivially spoofable when the
    # app is exposed directly, which would let one caller forge another's IP
    # for rate limiting.
    trusted_proxy_ips: list[str] = []

    # --- Rate limiting ----------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 120
    rate_limit_burst: int = 30
    auth_rate_limit_per_minute: int = 10

    # --- Billing ----------------------------------------------------------
    stripe_secret_key: str | None = None
    stripe_webhook_secret: str | None = None
    stripe_price_ids: dict[str, str] = {}
    billing_enabled: bool = False

    # Storage
    postgres_dsn: str = "postgresql+psycopg://geolytics:geolytics@localhost:5432/geolytics"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    redis_url: str = "redis://localhost:6379/0"

    # Embeddings
    embedding_backend: Literal["hashing", "sentence-transformers"] = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Crawler
    crawl_user_agent: str = "GEOlyticsBot/0.1 (+https://github.com/amnaanamds-cmyk/GEOlytics)"
    crawl_max_pages: int = 100
    crawl_delay_seconds: float = 1.0
    crawl_respect_robots: bool = True
    crawl_timeout_seconds: float = 30.0

    # --- SSRF policy (see geolytics.crawl.guard) -------------------------
    # The crawler fetches customer-supplied URLs from inside the production
    # network. These two settings are the entire difference between a crawler
    # and an open proxy into your own infrastructure.
    #
    # allow_private_addresses lets the crawler reach loopback, RFC1918 and
    # link-local addresses -- including the cloud metadata endpoint. It exists
    # for the test suite and for self-hosted installs auditing an intranet.
    # It must stay False in any multi-tenant deployment.
    crawl_allow_private_addresses: bool = False
    # Restricting ports stops the crawler being used to probe internal
    # services. Lifting the restriction takes a deliberate boolean rather than
    # an empty or null list: "I cleared the list" must never silently mean
    # "everything is allowed".
    crawl_restrict_ports: bool = True
    crawl_allowed_ports: list[int] = [80, 443, 8080, 8443]
    # Narrow exemption from the address check, for a self-hosted install that
    # must reach one known internal host. Safer than crawl_allow_private_addresses.
    crawl_allowed_hosts: list[str] = []
    crawl_blocked_hosts: list[str] = []
    crawl_max_bytes: int = 5 * 1024 * 1024

    # LLM
    llm_backend: Literal["ollama", "none"] = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"


    @model_validator(mode="after")
    def _enforce_production_safety(self) -> Settings:
        """Refuse to start production with a configuration that is not safe.

        These are all failures that are silent at boot and expensive later: a
        guessable signing key, a wildcard CORS origin on a credentialed API, a
        crawler that will fetch the cloud metadata endpoint, or an embedder
        that produces meaningless numbers. Better to fail loudly here.
        """
        if self.env != "production":
            return self

        problems: list[str] = []

        if len(self.secret_key) < MIN_SECRET_LENGTH:
            problems.append(
                f"GEOLYTICS_SECRET_KEY must be at least {MIN_SECRET_LENGTH} characters"
            )
        if _looks_like_placeholder(self.secret_key):
            problems.append("GEOLYTICS_SECRET_KEY is a placeholder value")
        if "*" in self.cors_origins:
            problems.append("GEOLYTICS_CORS_ORIGINS must not contain '*' in production")
        if self.crawl_allow_private_addresses:
            problems.append(
                "GEOLYTICS_CRAWL_ALLOW_PRIVATE_ADDRESSES must be false in production: "
                "it lets a customer's audit reach the cloud metadata endpoint"
            )
        if not self.crawl_restrict_ports:
            problems.append(
                "GEOLYTICS_CRAWL_RESTRICT_PORTS must be true in production"
            )
        if self.embedding_backend == "hashing":
            problems.append(
                "GEOLYTICS_EMBEDDING_BACKEND=hashing is a test fixture with no semantics"
            )
        if self.billing_enabled and not self.stripe_webhook_secret:
            problems.append(
                "GEOLYTICS_STRIPE_WEBHOOK_SECRET is required when billing is enabled: "
                "unverified webhooks would let anyone grant themselves a plan"
            )

        if problems:
            raise ValueError(
                "unsafe production configuration:\n  - " + "\n  - ".join(problems)
            )
        return self

    @property
    def secret_key_is_ephemeral(self) -> bool:
        """True when no key was supplied, so tokens die with this process."""
        import os

        return not os.environ.get("GEOLYTICS_SECRET_KEY")


def _looks_like_placeholder(secret: str) -> bool:
    """Whether a secret is a stock placeholder rather than a real key.

    Matched on prefix as well as whole value, because the placeholder people
    actually commit is "change-me-in-production" -- long enough to pass a
    length check and no more secret for it.
    """
    value = secret.strip().lower()
    if value in _UNSAFE_SECRETS:
        return True
    return any(
        value.startswith(f"{placeholder}{separator}")
        for placeholder in _UNSAFE_SECRETS
        for separator in ("-", "_", ".", " ")
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
