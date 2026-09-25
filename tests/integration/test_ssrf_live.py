"""SSRF defences that only a real HTTP server can exercise.

Redirect revalidation and the body-size cap both happen mid-request, so a
server that actually redirects and actually streams bytes is required.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from geolytics.crawl.guard import UrlPolicy, UrlPolicyError, safe_get

# Loopback is the server under test, so private addresses are permitted; the
# redirect *targets* are what most of these tests are about.
LOOPBACK_OK = UrlPolicy(allow_private_addresses=True, allowed_ports=None, max_bytes=4096)
PRODUCTION_LIKE = UrlPolicy(allow_private_addresses=False, allowed_ports=None)
# Exempts only the fixture server's address, so the address filter is still
# live for every other host -- which is what proves a redirect hop is checked
# rather than trusted because the seed was allowed.
SEED_ONLY = UrlPolicy(
    allow_private_addresses=False,
    allowed_hosts=frozenset({"127.0.0.1"}),
    allowed_ports=None,
    max_bytes=4096,
)


class _Handler(BaseHTTPRequestHandler):
    """Routes: /ok, /redirect-to?u=, /chain/<n>, /big, /chunked-big,
    /claims-huge, /loop, /no-location."""

    def log_message(self, fmt: str, *args: object) -> None:
        pass

    def handle_one_request(self) -> None:
        # The size-cap tests abort mid-body on purpose; the resulting broken
        # pipe is the expected outcome, not a server fault worth a traceback.
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path
        if path.startswith("/redirect-to?u="):
            self._redirect(path.split("=", 1)[1])
        elif path.startswith("/chain/"):
            n = int(path.rsplit("/", 1)[1])
            self._redirect("/ok" if n <= 1 else f"/chain/{n - 1}")
        elif path == "/loop":
            self._redirect("/loop")
        elif path == "/big":
            self._body(b"x" * 100_000)
        elif path == "/chunked-big":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for _ in range(20):
                self.wfile.write(b"2000\r\n" + b"z" * 0x2000 + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        elif path == "/claims-huge":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", "99999999")
            self.end_headers()
        elif path == "/no-location":
            self.send_response(302)
            self.end_headers()
        else:
            self._body(b"<html><body>ok</body></html>")

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _body(self, payload: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def client() -> Iterator[httpx.Client]:
    with httpx.Client(follow_redirects=False, timeout=10.0) as c:
        yield c


class TestHappyPath:
    def test_fetches_a_normal_page(self, client, server):
        response = safe_get(client, f"{server}/ok", LOOPBACK_OK)
        assert response.status_code == 200
        assert "ok" in response.text

    def test_reports_the_requested_url_not_the_pinned_ip(self, client, server):
        response = safe_get(client, f"{server}/ok", LOOPBACK_OK)
        assert str(response.request.url) == f"{server}/ok"


class TestRedirects:
    def test_follows_a_redirect_chain(self, client, server):
        response = safe_get(client, f"{server}/chain/3", LOOPBACK_OK)
        assert response.status_code == 200
        assert "ok" in response.text

    def test_redirect_to_the_metadata_endpoint_is_refused(self, client, server):
        """The classic bypass: an allowed URL that 302s to 169.254.169.254."""
        with pytest.raises(UrlPolicyError) as exc:
            safe_get(client, f"{server}/redirect-to?u=http://169.254.169.254/", SEED_ONLY)
        assert exc.value.reason == "blocked_address"
        assert "169.254.169.254" in exc.value.url

    def test_the_seed_exemption_does_not_extend_to_the_hop(self, client, server):
        """Allowing the seed host must not whitelist wherever it redirects."""
        with pytest.raises(UrlPolicyError) as exc:
            safe_get(client, f"{server}/redirect-to?u=http://10.0.0.1/", SEED_ONLY)
        assert exc.value.reason == "blocked_address"

    def test_redirect_to_a_private_address_is_refused_under_production_policy(
        self, client, server
    ):
        # The seed itself is loopback, so this asserts the *hop* is checked
        # with the same policy object rather than only the seed.
        with pytest.raises(UrlPolicyError):
            safe_get(client, f"{server}/redirect-to?u=http://10.0.0.1/", PRODUCTION_LIKE)

    def test_redirect_to_a_forbidden_scheme_is_refused(self, client, server):
        with pytest.raises(UrlPolicyError) as exc:
            safe_get(client, f"{server}/redirect-to?u=file:///etc/passwd", LOOPBACK_OK)
        assert exc.value.reason == "scheme"

    def test_redirect_loop_is_bounded(self, client, server):
        with pytest.raises(UrlPolicyError) as exc:
            safe_get(client, f"{server}/loop", LOOPBACK_OK)
        assert exc.value.reason == "too_many_redirects"

    def test_chain_longer_than_the_limit_is_refused(self, client, server):
        policy = UrlPolicy(allow_private_addresses=True, allowed_ports=None, max_redirects=2)
        with pytest.raises(UrlPolicyError, match="too_many_redirects"):
            safe_get(client, f"{server}/chain/5", policy)

    def test_redirect_without_a_location_is_refused(self, client, server):
        with pytest.raises(UrlPolicyError):
            safe_get(client, f"{server}/no-location", LOOPBACK_OK)


class TestResponseSize:
    def test_body_over_the_cap_is_refused(self, client, server):
        with pytest.raises(UrlPolicyError) as exc:
            safe_get(client, f"{server}/big", LOOPBACK_OK)
        assert exc.value.reason == "too_large"

    def test_a_chunked_body_with_no_length_is_still_capped(self, client, server):
        """Chunked encoding declares no length, so only counting bytes catches it."""
        with pytest.raises(UrlPolicyError) as exc:
            safe_get(client, f"{server}/chunked-big", LOOPBACK_OK)
        assert exc.value.reason == "too_large"

    def test_an_oversized_declared_length_is_rejected_before_reading(self, client, server):
        with pytest.raises(UrlPolicyError) as exc:
            safe_get(client, f"{server}/claims-huge", LOOPBACK_OK)
        assert exc.value.reason == "too_large"

    def test_a_body_under_the_cap_is_returned_whole(self, client, server):
        policy = UrlPolicy(allow_private_addresses=True, allowed_ports=None, max_bytes=1_000_000)
        assert len(safe_get(client, f"{server}/big", policy).content) == 100_000


class TestCrawlerIntegration:
    def test_crawler_refuses_a_policy_violating_seed(self, crawl_settings, server):
        from geolytics.crawl.crawler import Crawler

        with (
            Crawler(settings=crawl_settings, cache_dir=None, policy=PRODUCTION_LIKE) as crawler,
            pytest.raises(UrlPolicyError),
        ):
            crawler.crawl(f"{server}/ok")

    def test_crawler_counts_policy_refusals_separately_from_errors(
        self, crawl_settings, server
    ):
        """A refused URL is a policy refusal, not a robots.txt block.

        The check runs before robots.txt for exactly this reason: fetching a
        blocked host's /robots.txt is itself refused, and reporting that as
        "disallowed by robots.txt" would send an operator hunting the wrong bug.
        """
        from geolytics.crawl.crawler import Crawler

        with Crawler(settings=crawl_settings, cache_dir=None, policy=PRODUCTION_LIKE) as crawler:
            assert crawler.fetch_one(f"{server}/ok") is None
            assert crawler.stats.skipped_policy == 1
            assert crawler.stats.skipped_robots == 0
            assert crawler.stats.errors == []
            assert crawler.stats.policy_refusals[0][1] == "blocked_address"

    def test_summary_mentions_policy_refusals(self, crawl_settings, server):
        from geolytics.crawl.crawler import Crawler

        with Crawler(settings=crawl_settings, cache_dir=None, policy=PRODUCTION_LIKE) as crawler:
            crawler.fetch_one(f"{server}/ok")
            assert "refused by URL policy" in crawler.stats.summary()
