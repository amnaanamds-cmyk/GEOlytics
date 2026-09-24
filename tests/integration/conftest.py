"""Fixtures for tests that talk to real services.

Every service-backed test skips cleanly when its service is absent, so the
default `pytest` run stays green on a machine with nothing running. Start the
stack with `docker compose up -d` to exercise them.

The crawler tests are the exception: they serve the fixture site from a local
HTTP server, so they always run and need no external network. That is
deliberate -- crawling a real site from a test suite is slow, impolite, and
non-reproducible the moment the site changes.
"""

from __future__ import annotations

import functools
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from services import POSTGRES_DSN, QDRANT_URL, REDIS_URL

FIXTURE_SITE = Path(__file__).resolve().parents[1] / "fixtures" / "site"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the fixture site without logging every request to stderr."""

    def log_message(self, fmt: str, *args: object) -> None:
        pass


@pytest.fixture(scope="session")
def fixture_site() -> Iterator[str]:
    """Serve tests/fixtures/site over HTTP on an ephemeral port."""
    handler = functools.partial(_QuietHandler, directory=str(FIXTURE_SITE))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def crawl_settings(fixture_site: str):
    """Settings pointed at the fixture site, with no politeness delay.

    A real crawl sleeps between requests. The fixture server is local and its
    robots.txt declares `Crawl-delay: 0`, so the floor delay is set to zero
    here to keep the suite fast -- this is the one place that is safe.
    """
    from geolytics.config import Settings

    return Settings(
        env="test",
        crawl_delay_seconds=0.0,
        crawl_max_pages=10,
        crawl_user_agent="GEOlyticsBot/0.1 (+test)",
        crawl_respect_robots=True,
        postgres_dsn=POSTGRES_DSN,
        qdrant_url=QDRANT_URL,
        redis_url=REDIS_URL,
        embedding_backend="hashing",
        embedding_dim=128,
        llm_backend="none",
    )
