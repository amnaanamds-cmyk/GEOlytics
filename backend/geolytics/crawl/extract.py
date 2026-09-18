"""HTML to `Document`, preserving heading structure.

Heading structure is kept rather than flattened because two downstream
consumers need it: the sentence and semantic chunkers never span a heading
boundary (so every chunk has one topic), and the GEO layer scores heading
hierarchy directly.

Extractor precedence: trafilatura (best boilerplate removal) -> BeautifulSoup
-> a regex fallback that keeps the test suite dependency-free. Record which one
ran -- extractors disagree, and a corpus built with two of them is not one
corpus.
"""

from __future__ import annotations

import hashlib
import html as html_module
import re
from typing import Any

from geolytics.chunking.base import Document, Section

_HEADING_RE = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_DROP_RE = re.compile(
    r"<(script|style|nav|footer|header|aside|noscript|form)\b.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANKS_RE = re.compile(r"\n{3,}")


def doc_id_for(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def extract_document(
    html: str,
    url: str,
    prefer: str = "auto",
    metadata: dict[str, Any] | None = None,
) -> Document:
    """Extract main content and heading sections from raw HTML."""
    title = _extract_title(html)
    text, extractor = _extract_text(html, url, prefer)
    sections = extract_sections(html) or (
        Section(text=text, heading_path=(), start_char=0),
    )

    return Document(
        doc_id=doc_id_for(url),
        url=url,
        title=title,
        text=text,
        sections=sections,
        metadata={
            "extractor": extractor,
            "html_length": len(html),
            "text_length": len(text),
            **(metadata or {}),
        },
    )


def extract_sections(html: str) -> tuple[Section, ...]:
    """Split the body into heading-delimited sections with ancestor paths.

    Text appearing before the first heading becomes a section with an empty
    heading path, so no content is lost.
    """
    cleaned = _DROP_RE.sub(" ", html)
    matches = list(_HEADING_RE.finditer(cleaned))
    if not matches:
        return ()

    sections: list[Section] = []
    cursor = 0
    stack: list[tuple[int, str]] = []

    preamble = _plain_text(cleaned[: matches[0].start()])
    if preamble:
        sections.append(Section(text=preamble, heading_path=(), start_char=0))
    cursor = len(preamble)

    for i, match in enumerate(matches):
        level = int(match.group(1))
        heading = _plain_text(match.group(2))

        while stack and stack[-1][0] >= level:
            stack.pop()
        if heading:
            stack.append((level, heading))

        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        body = _plain_text(cleaned[match.end() : end])
        if not body:
            continue

        sections.append(
            Section(
                text=body,
                heading_path=tuple(h for _, h in stack),
                start_char=cursor,
            )
        )
        cursor += len(body) + 1

    return tuple(sections)


def _extract_text(html: str, url: str, prefer: str) -> tuple[str, str]:
    if prefer in ("auto", "trafilatura"):
        try:
            import trafilatura

            extracted = trafilatura.extract(html, url=url, include_comments=False)
            if extracted:
                return _normalize(extracted), "trafilatura"
        except ImportError:
            if prefer == "trafilatura":
                raise

    if prefer in ("auto", "bs4"):
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()
            return _normalize(soup.get_text("\n")), "bs4"
        except ImportError:
            if prefer == "bs4":
                raise

    return _normalize(_plain_text(_DROP_RE.sub(" ", html))), "regex"


def _extract_title(html: str) -> str:
    match = _TITLE_RE.search(html)
    if match:
        return _plain_text(match.group(1))
    heading = _HEADING_RE.search(html)
    return _plain_text(heading.group(2)) if heading else ""


def _plain_text(fragment: str) -> str:
    return _normalize(html_module.unescape(_TAG_RE.sub(" ", fragment)))


def _normalize(text: str) -> str:
    return _BLANKS_RE.sub("\n\n", _WS_RE.sub(" ", text)).strip()
