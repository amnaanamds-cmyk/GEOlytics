"""Application settings, loaded from environment / .env.

Every knob that affects an experiment lives here so that a run can record the
exact configuration it executed under (see `evaluation.harness.RunConfig`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GEOLYTICS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
