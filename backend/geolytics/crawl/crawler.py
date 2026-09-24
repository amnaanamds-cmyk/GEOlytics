"""A polite, same-host BFS crawler with an on-disk cache."""

from __future__ import annotations

import hashlib
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

import httpx

from geolytics.chunking.base import Document
from geolytics.config import Settings, get_settings
from geolytics.crawl.extract import extract_document
from geolytics.crawl.robots import RobotsPolicy

_HREF_RE = re.compile(r'<a\b[^>]*href=["\']([^"\']+)["\']', re.IGNORECASE)
_SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".zip", ".gz", ".mp4", ".mp3", ".css", ".js", ".xml", ".json",
)


@dataclass(frozen=True, slots=True)
class CrawlResult:
    url: str
    status: int
    html: str
    document: Document
    elapsed: float
    from_cache: bool = False


@dataclass
class CrawlStats:
    requested: int = 0
    fetched: int = 0
    cached: int = 0
    skipped_robots: int = 0
    skipped_type: int = 0
    skipped_duplicate: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.fetched} fetched ({self.cached} from cache), "
            f"{self.skipped_duplicate} duplicate content, "
            f"{self.skipped_robots} disallowed by robots.txt, "
            f"{self.skipped_type} non-HTML, {len(self.errors)} errors"
        )


class Crawler:
    """Breadth-first crawl of one host.

    Design choices worth defending:

    * **Same host only.** An audit is about one site; following outbound links
      would produce a corpus that is not the site under audit.
    * **Raw HTML cached to disk.** Re-running an experiment must not re-crawl.
      Without this, every re-run hits the site again (rude, slow) and silently
      changes the corpus underneath the results (unreproducible).
    * **Delay applied per request, from robots.txt where declared.**
    """

    def __init__(
        self,
        settings: Settings | None = None,
        cache_dir: str | Path | None = "data/cache",
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.robots = RobotsPolicy(
            user_agent=self.settings.crawl_user_agent,
            default_delay=self.settings.crawl_delay_seconds,
            respect=self.settings.crawl_respect_robots,
        )
        self._client = client or httpx.Client(
            timeout=self.settings.crawl_timeout_seconds,
            headers={"User-Agent": self.settings.crawl_user_agent},
            follow_redirects=True,
        )
        self.stats = CrawlStats()

    def crawl(self, start_url: str, max_pages: int | None = None) -> list[CrawlResult]:
        limit = max_pages or self.settings.crawl_max_pages
        host = urlparse(start_url).netloc.lower()

        queue: deque[str] = deque([_canonical(start_url)])
        seen: set[str] = {_canonical(start_url)}
        seen_content: set[str] = set()
        results: list[CrawlResult] = []

        while queue and len(results) < limit:
            url = queue.popleft()
            self.stats.requested += 1

            if not self.robots.can_fetch(url):
                self.stats.skipped_robots += 1
                continue

            result = self._fetch(url)
            if result is None:
                continue

            # URL canonicalisation cannot catch every alias: "/" and
            # "/index.html" are distinct URLs serving identical content, as are
            # most tracking-parameter variants. Indexing the same text under two
            # doc_ids would duplicate every chunk derived from it, which inflates
            # the corpus and gives a query two equally valid gold answers.
            fingerprint = _content_fingerprint(result.document.text)
            if fingerprint in seen_content:
                self.stats.skipped_duplicate += 1
                continue
            seen_content.add(fingerprint)

            results.append(result)

            for link in _links(result.html, url):
                if len(seen) >= limit * 4:
                    break  # Bound the frontier on a large site.
                if urlparse(link).netloc.lower() != host or link in seen:
                    continue
                if link.lower().endswith(_SKIP_EXTENSIONS):
                    continue
                seen.add(link)
                queue.append(link)

        return results

    def fetch_one(self, url: str) -> CrawlResult | None:
        if not self.robots.can_fetch(url):
            self.stats.skipped_robots += 1
            return None
        return self._fetch(_canonical(url))

    def _fetch(self, url: str) -> CrawlResult | None:
        cached = self._read_cache(url)
        if cached is not None:
            self.stats.fetched += 1
            self.stats.cached += 1
            return CrawlResult(
                url=url,
                status=200,
                html=cached,
                document=extract_document(cached, url),
                elapsed=0.0,
                from_cache=True,
            )

        time.sleep(self.robots.crawl_delay(url))
        started = time.perf_counter()
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            self.stats.errors.append((url, str(exc)))
            return None

        if not response.is_success:
            self.stats.errors.append((url, f"HTTP {response.status_code}"))
            return None

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type.lower():
            self.stats.skipped_type += 1
            return None

        html = response.text
        self._write_cache(url, html)
        self.stats.fetched += 1

        return CrawlResult(
            url=url,
            status=response.status_code,
            html=html,
            document=extract_document(
                html, url, metadata={"fetched_at": time.time(), "content_type": content_type}
            ),
            elapsed=time.perf_counter() - started,
        )

    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{hashlib.sha1(url.encode()).hexdigest()}.html"

    def _read_cache(self, url: str) -> str | None:
        path = self._cache_path(url)
        if path and path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        return None

    def _write_cache(self, url: str, html: str) -> None:
        path = self._cache_path(url)
        if path:
            path.write_text(html, encoding="utf-8")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Crawler:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _content_fingerprint(text: str) -> str:
    """Hash of the extracted text, whitespace-normalised."""
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()


def _canonical(url: str) -> str:
    """Strip the fragment and a trailing slash so one page is crawled once."""
    clean, _ = urldefrag(url)
    parsed = urlparse(clean)
    path = parsed.path.rstrip("/") or "/"
    rebuilt = f"{parsed.scheme}://{parsed.netloc.lower()}{path}"
    return f"{rebuilt}?{parsed.query}" if parsed.query else rebuilt


def _links(html: str, base_url: str) -> list[str]:
    out: list[str] = []
    for href in _HREF_RE.findall(html):
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        out.append(_canonical(urljoin(base_url, href)))
    return out
