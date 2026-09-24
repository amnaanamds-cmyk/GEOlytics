"""Extraction and robots handling."""

from __future__ import annotations

import pytest

from geolytics.crawl.crawler import _canonical, _links
from geolytics.crawl.extract import extract_document, extract_sections
from sample_data import SAMPLE_HTML


class TestExtraction:
    def test_extracts_title(self):
        assert extract_document(SAMPLE_HTML, "https://acme.example/").title == "Acme Plumbing"

    def test_falls_back_to_h1_when_no_title(self):
        assert extract_document("<h1>Only Heading</h1>", "u").title == "Only Heading"

    def test_drops_boilerplate_tags(self):
        text = extract_document(SAMPLE_HTML, "u").text
        assert "Copyright" not in text
        assert "Home Services Contact" not in text

    def test_records_which_extractor_ran(self):
        assert extract_document(SAMPLE_HTML, "u").metadata["extractor"] in {
            "trafilatura",
            "bs4",
            "regex",
        }

    def test_doc_id_is_stable_per_url(self):
        a = extract_document(SAMPLE_HTML, "https://acme.example/")
        b = extract_document("<p>different</p>", "https://acme.example/")
        assert a.doc_id == b.doc_id

    def test_unescapes_entities(self):
        assert "&amp;" not in extract_document("<p>Tom &amp; Jerry</p>", "u").text


class TestSections:
    def test_builds_ancestor_heading_paths(self):
        sections = extract_sections(SAMPLE_HTML)
        paths = {s.heading_path for s in sections}
        assert ("Acme Plumbing", "Emergency callouts", "Callout pricing") in paths

    def test_h2_resets_the_h3_level(self):
        paths = {s.heading_path for s in extract_sections(SAMPLE_HTML)}
        assert ("Acme Plumbing", "Bathroom installation") in paths

    def test_preamble_before_first_heading_is_kept(self):
        sections = extract_sections("<p>Intro text here.</p><h1>Title</h1><p>Body.</p>")
        assert sections[0].heading_path == ()
        assert "Intro" in sections[0].text

    def test_no_headings_returns_empty(self):
        assert extract_sections("<p>Just a paragraph.</p>") == ()

    def test_sections_carry_no_markup(self):
        assert all("<" not in s.text for s in extract_sections(SAMPLE_HTML))


class TestUrlHandling:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://a.example/page#frag", "https://a.example/page"),
            ("https://a.example/page/", "https://a.example/page"),
            ("https://A.Example/Page", "https://a.example/Page"),
            ("https://a.example", "https://a.example/"),
            ("https://a.example/p?x=1", "https://a.example/p?x=1"),
        ],
    )
    def test_canonicalisation(self, raw, expected):
        assert _canonical(raw) == expected

    def test_links_are_absolutised(self):
        html = '<a href="/about">About</a><a href="https://other.example/x">Out</a>'
        links = _links(html, "https://acme.example/home")
        assert "https://acme.example/about" in links
        assert "https://other.example/x" in links

    def test_non_http_schemes_are_dropped(self):
        html = '<a href="mailto:a@b.c">Mail</a><a href="tel:123">Call</a><a href="#top">Top</a>'
        assert _links(html, "https://acme.example/") == []


class TestRobotsPolicy:
    def test_unreachable_robots_fails_closed(self):
        from geolytics.crawl.robots import RobotsPolicy

        policy = RobotsPolicy(user_agent="TestBot")
        # Simulate the fetch having failed for this host.
        policy._parsers["blocked.example"] = None
        policy._unreachable.add("blocked.example")
        assert not policy.can_fetch("https://blocked.example/page")

    def test_absent_robots_allows(self):
        from geolytics.crawl.robots import RobotsPolicy

        policy = RobotsPolicy(user_agent="TestBot")
        policy._parsers["open.example"] = None
        assert policy.can_fetch("https://open.example/page")

    def test_disallow_is_honoured(self):
        import urllib.robotparser

        from geolytics.crawl.robots import RobotsPolicy

        parser = urllib.robotparser.RobotFileParser()
        parser.parse(["User-agent: *", "Disallow: /private"])
        policy = RobotsPolicy(user_agent="TestBot")
        policy._parsers["x.example"] = parser
        assert policy.can_fetch("https://x.example/public")
        assert not policy.can_fetch("https://x.example/private/page")

    def test_crawl_delay_never_below_our_floor(self):
        import urllib.robotparser

        from geolytics.crawl.robots import RobotsPolicy

        parser = urllib.robotparser.RobotFileParser()
        parser.parse(["User-agent: *", "Crawl-delay: 0.1"])
        policy = RobotsPolicy(user_agent="TestBot", default_delay=2.0)
        policy._parsers["x.example"] = parser
        assert policy.crawl_delay("https://x.example/") == 2.0

    def test_longer_declared_delay_is_respected(self):
        import urllib.robotparser

        from geolytics.crawl.robots import RobotsPolicy

        parser = urllib.robotparser.RobotFileParser()
        parser.parse(["User-agent: *", "Crawl-delay: 5"])
        policy = RobotsPolicy(user_agent="TestBot", default_delay=1.0)
        policy._parsers["x.example"] = parser
        assert policy.crawl_delay("https://x.example/") == 5.0

    def test_respect_false_bypasses(self):
        from geolytics.crawl.robots import RobotsPolicy

        policy = RobotsPolicy(user_agent="TestBot", respect=False)
        policy._unreachable.add("blocked.example")
        assert policy.can_fetch("https://blocked.example/page")
