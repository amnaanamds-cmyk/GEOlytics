"""Detection and endpoints for the services the integration tests need.

Kept separate from conftest.py so test modules can import the markers and DSNs
by a unique name -- there is already a `conftest` module one directory up, and
two modules with the same name cannot both be imported.
"""

from __future__ import annotations

import functools
import socket

import httpx
import pytest

QDRANT_URL = "http://localhost:6333"
REDIS_URL = "redis://localhost:6379/15"
POSTGRES_DSN = "postgresql+psycopg://geolytics:geolytics@localhost:5432/geolytics_test"


def _tcp_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@functools.cache
def qdrant_available() -> bool:
    if not _tcp_open("localhost", 6333):
        return False
    try:
        return httpx.get(f"{QDRANT_URL}/readyz", timeout=3.0).is_success
    except httpx.HTTPError:
        return False


@functools.cache
def postgres_available() -> bool:
    return _tcp_open("localhost", 5432)


@functools.cache
def redis_available() -> bool:
    return _tcp_open("localhost", 6379)


def make_test_settings(**overrides):
    """Settings for tests that crawl the local fixture server.

    The fixture site is served on 127.0.0.1 on an ephemeral port, which the
    production SSRF policy refuses on both counts. Relaxing it lives here, in
    one place, so no test quietly disables the protection on its own -- tests
    that exercise the policy itself build their own `UrlPolicy`.
    """
    from geolytics.config import Settings

    defaults = {
        "env": "test",
        "crawl_delay_seconds": 0.0,
        "crawl_max_pages": 10,
        "crawl_user_agent": "GEOlyticsBot/0.1 (+test)",
        "crawl_respect_robots": True,
        "crawl_allow_private_addresses": True,
        "crawl_restrict_ports": False,
        "postgres_dsn": POSTGRES_DSN,
        "qdrant_url": QDRANT_URL,
        "redis_url": REDIS_URL,
        "embedding_backend": "hashing",
        "embedding_dim": 128,
        "llm_backend": "none",
    }
    return Settings(**{**defaults, **overrides})


requires_qdrant = pytest.mark.skipif(not qdrant_available(), reason="qdrant not running")
requires_postgres = pytest.mark.skipif(not postgres_available(), reason="postgres not running")
requires_redis = pytest.mark.skipif(not redis_available(), reason="redis not running")
