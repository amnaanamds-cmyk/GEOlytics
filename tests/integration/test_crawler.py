"""The crawler against a real HTTP server serving the fixture site."""

from __future__ import annotations

from pathlib import Path

import pytest

from geolytics.crawl.crawler import Crawler
from geolytics.crawl.guard import UrlPolicy
from geolytics.crawl.robots import RobotsPolicy

# The fixture server is on loopback and an ephemeral port.
LOOPBACK_OK = UrlPolicy(allow_private_addresses=True, allowed_ports=None)


@pytest.fixture
def crawler(crawl_settings, tmp_path: Path) -> Crawler:
    with Crawler(settings=crawl_settings, cache_dir=tmp_path / "cache") as c:
        yield c


class TestCrawl:
    def test_deduplicates_url_aliases_of_the_same_page(self, crawler, fixture_site):
        """The nav links "/" while the crawl starts at "/index.html"."""
        results = crawler.crawl(f"{fixture_site}/index.html")
        bodies = [" ".join(r.document.text.split()) for r in results]
        assert len(bodies) == len(set(bodies))
        assert crawler.stats.skipped_duplicate >= 1

    def test_follows_internal_links(self, crawler, fixture_site):
        results = crawler.crawl(f"{fixture_site}/index.html")
        urls = {r.url for r in results}
        assert any(u.endswith("/services.html") for u in urls)
        assert any(u.endswith("/pricing.html") for u in urls)

    def test_honours_robots_disallow(self, crawler, fixture_site):
        results = crawler.crawl(f"{fixture_site}/index.html")
        assert not any("/private/" in r.url for r in results)
        assert crawler.stats.skipped_robots >= 1

    def test_stays_on_host(self, crawler, fixture_site):
        results = crawler.crawl(f"{fixture_site}/index.html")
        assert all("external.example" not in r.url for r in results)

    def test_respects_max_pages(self, crawler, fixture_site):
        assert len(crawler.crawl(f"{fixture_site}/index.html", max_pages=2)) == 2

    def test_extracts_headings_from_real_response(self, crawler, fixture_site):
        result = crawler.fetch_one(f"{fixture_site}/services.html")
        assert result is not None
        paths = {s.heading_path for s in result.document.sections}
        assert ("Services", "Emergency callouts", "Response times") in paths

    def test_extracts_title_and_drops_boilerplate(self, crawler, fixture_site):
        result = crawler.fetch_one(f"{fixture_site}/pricing.html")
        assert "Pricing" in result.document.title
        assert "Copyright 2024" not in result.document.text

    def test_second_crawl_is_served_from_cache(self, crawl_settings, fixture_site, tmp_path):
        cache = tmp_path / "cache"
        with Crawler(settings=crawl_settings, cache_dir=cache) as first:
            first.crawl(f"{fixture_site}/index.html")
            assert first.stats.cached == 0
        assert list(cache.glob("*.html"))

        with Crawler(settings=crawl_settings, cache_dir=cache) as second:
            results = second.crawl(f"{fixture_site}/index.html")
            # `cached` counts cache hits, which includes the page later dropped
            # as duplicate content, so it is >= the number of results returned.
            assert second.stats.cached == second.stats.fetched
            assert second.stats.cached >= len(results)
            assert all(r.from_cache for r in results)

    def test_missing_page_is_recorded_not_raised(self, crawler, fixture_site):
        assert crawler.fetch_one(f"{fixture_site}/nope.html") is None
        assert any("404" in reason for _, reason in crawler.stats.errors)

    def test_stats_summary_is_readable(self, crawler, fixture_site):
        crawler.crawl(f"{fixture_site}/index.html")
        summary = crawler.stats.summary()
        assert "fetched" in summary and "robots.txt" in summary


class TestRobotsAgainstRealServer:
    def test_reads_the_served_robots_txt(self, fixture_site):
        policy = RobotsPolicy(
            user_agent="GEOlyticsBot/0.1 (+test)", default_delay=0.0, policy=LOOPBACK_OK
        )
        assert policy.can_fetch(f"{fixture_site}/index.html")
        assert not policy.can_fetch(f"{fixture_site}/private/internal.html")

    def test_declared_crawl_delay_is_read(self, fixture_site):
        policy = RobotsPolicy(
            user_agent="GEOlyticsBot/0.1 (+test)", default_delay=0.0, policy=LOOPBACK_OK
        )
        # The fixture declares Crawl-delay: 0, and our floor is 0 here.
        assert policy.crawl_delay(f"{fixture_site}/index.html") == 0.0

    def test_floor_delay_wins_over_a_smaller_declared_delay(self, fixture_site):
        policy = RobotsPolicy(
            user_agent="GEOlyticsBot/0.1 (+test)", default_delay=1.5, policy=LOOPBACK_OK
        )
        assert policy.crawl_delay(f"{fixture_site}/index.html") == 1.5
