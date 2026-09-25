"""robots.txt compliance.

Not optional, and not only an ethics point for the report's appendix: a crawler
that ignores robots.txt or hammers a small business's shared host will get the
project's IP blocked partway through data collection, which is a practical
problem as much as a principled one.

Policy implemented here:
* Fetch and honour robots.txt per host, with the configured user agent.
* Honour `Crawl-delay` when present; fall back to the configured delay.
* Fail closed on a 5xx or an unreachable robots.txt -- an unreadable policy is
  treated as "do not crawl", not as "no restrictions".
* A 404 means no policy exists, which genuinely does mean unrestricted.
"""

from __future__ import annotations

import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx

from geolytics.crawl.guard import UrlPolicy, UrlPolicyError, safe_get


@dataclass
class RobotsPolicy:
    """Caches one parsed robots.txt per host."""

    user_agent: str
    default_delay: float = 1.0
    timeout: float = 10.0
    respect: bool = True
    # robots.txt is fetched from the same untrusted host as the pages, so it
    # goes through the same SSRF policy. Without this, /robots.txt would be an
    # unguarded fetch of a customer-supplied URL.
    policy: UrlPolicy | None = None
    _parsers: dict[str, urllib.robotparser.RobotFileParser | None] = field(
        default_factory=dict, repr=False
    )
    _unreachable: set[str] = field(default_factory=set, repr=False)

    def can_fetch(self, url: str) -> bool:
        if not self.respect:
            return True
        host = _host(url)
        parser = self._parser_for(url)
        if parser is None:
            # Distinguish "no policy" (allowed) from "policy unreadable" (denied).
            return host not in self._unreachable
        return parser.can_fetch(self.user_agent, url)

    def crawl_delay(self, url: str) -> float:
        if not self.respect:
            return self.default_delay
        parser = self._parser_for(url)
        if parser is None:
            return self.default_delay
        declared = parser.crawl_delay(self.user_agent)
        # Never crawl faster than our own configured floor, even if the site
        # permits it.
        return max(self.default_delay, float(declared)) if declared else self.default_delay

    def sitemaps(self, url: str) -> list[str]:
        parser = self._parser_for(url)
        return list(parser.site_maps() or []) if parser else []

    def _parser_for(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        host = _host(url)
        if host in self._parsers:
            return self._parsers[host]

        robots_url = urljoin(f"{urlparse(url).scheme}://{host}", "/robots.txt")
        parser: urllib.robotparser.RobotFileParser | None = None
        try:
            with httpx.Client(
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
                follow_redirects=False,
            ) as client:
                response = safe_get(client, robots_url, self.policy or UrlPolicy())
            if response.status_code == 404:
                parser = None  # No policy: unrestricted.
            elif response.is_success:
                parser = urllib.robotparser.RobotFileParser()
                parser.parse(response.text.splitlines())
            else:
                self._unreachable.add(host)
        except (httpx.HTTPError, UrlPolicyError):
            # A host whose robots.txt we cannot read is treated as disallowed
            # by `can_fetch`, which is the safe default.
            self._unreachable.add(host)

        self._parsers[host] = parser
        return parser


def _host(url: str) -> str:
    return urlparse(url).netloc.lower()
