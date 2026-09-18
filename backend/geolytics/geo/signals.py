"""Measurable content signals that plausibly affect generative-engine visibility.

Every signal here is computed from the page itself, is defined in one place,
and is reproducible. None of them is a claim about what any particular engine
does -- that claim cannot be made without access to the engine's retrieval and
ranking internals, and this system does not have it.

The signal *set* is chosen from two sources:

* Aggarwal et al., "GEO: Generative Engine Optimization" (KDD 2024), which
  experimentally tested content modifications -- adding citations, quotations
  and statistics, and authoritative phrasing -- and measured visibility change
  in generative engines. Those become `citation_density`, `quotation_density`,
  `statistic_density` and `authority_markers`. Verify the paper's exact
  definitions and reported effect sizes before citing figures.
* Properties that follow from how RAG pipelines actually work, independent of
  any engine: a chunk that cannot be understood in isolation retrieves badly
  and quotes badly, structure survives extraction, and boilerplate dilutes
  every embedding computed over the page.

Weights are NOT set here. Asserting "schema markup is worth 15%" is the single
most attackable thing a project like this can do. `geolytics.geo.scoring` fits
the weights against measured citation rates from the simulation instead.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from geolytics.chunking.base import Chunk, Document, split_sentences
from geolytics.retrieval.bm25 import tokenize

SIGNAL_NAMES = (
    "statistic_density",
    "citation_density",
    "quotation_density",
    "authority_markers",
    "schema_coverage",
    "heading_structure",
    "self_containedness",
    "extractability",
    "answer_directness",
    "freshness",
)

_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")
_PERCENT_OR_UNIT_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent|million|billion|thousand|k|x|hours?|days?|years?)\b",
    re.IGNORECASE,
)
_QUOTE_RE = re.compile(r"[\"“][^\"“”]{20,}[\"”]")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Opening tokens that make a sentence depend on text that is no longer nearby
# once the chunk is retrieved in isolation.
_DANGLING_OPENERS = frozenset(
    [
        "it", "this", "that", "these", "those", "they", "them", "he", "she",
        "we", "our", "its", "their", "his", "her", "there", "here", "such",
        "also", "however", "therefore", "moreover", "furthermore",
        "additionally", "consequently", "thus", "instead", "otherwise",
        "meanwhile", "besides",
    ]
)

_AUTHORITY_MARKERS = (
    "according to", "research shows", "study found", "data from", "reported by",
    "survey of", "published in", "certified", "accredited", "peer-reviewed",
    "as of", "source:",
)

_DIRECT_ANSWER_MARKERS = (
    " is ", " are ", " means ", " refers to ", " is defined as ", " works by ",
    " you can ", " to do this", " the answer is ",
)


@dataclass(frozen=True, slots=True)
class SignalReport:
    """Signal values for one page, each normalised to [0, 1].

    Normalisation is intentional: the signals feed a fitted linear model, and
    un-normalised features would make the fitted coefficients incomparable.
    `raw` keeps the pre-normalisation measurements for the report's appendix.
    """

    doc_id: str
    url: str
    values: dict[str, float]
    raw: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def vector(self, names: Sequence[str] = SIGNAL_NAMES) -> list[float]:
        return [self.values.get(n, 0.0) for n in names]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_signals(
    document: Document,
    chunks: Sequence[Chunk] | None = None,
    html: str | None = None,
) -> SignalReport:
    """Compute every signal for one page.

    `chunks` enables the chunk-level signals (self-containedness). `html`
    enables the markup signals (schema.org, heading hierarchy). Both are
    optional; missing inputs produce a recorded warning and a neutral 0.0
    rather than a silent default that would look like a real measurement.
    """
    text = document.text
    words = tokenize(text, remove_stopwords=False)
    n_words = max(1, len(words))
    sentences = split_sentences(text)
    warnings: list[str] = []

    numbers = _NUMBER_RE.findall(text)
    qualified = _PERCENT_OR_UNIT_RE.findall(text)
    quotes = _QUOTE_RE.findall(text)
    authority_hits = sum(text.lower().count(m) for m in _AUTHORITY_MARKERS)
    citations = _count_citations(document, html)

    values: dict[str, float] = {
        # Per-100-word densities, squashed so a page stuffed with numbers does
        # not dominate the fitted model.
        "statistic_density": _saturate(len(qualified) * 100.0 / n_words, ceiling=4.0),
        "citation_density": _saturate(citations * 100.0 / n_words, ceiling=2.0),
        "quotation_density": _saturate(len(quotes) * 100.0 / n_words, ceiling=1.5),
        "authority_markers": _saturate(authority_hits * 100.0 / n_words, ceiling=2.0),
        "answer_directness": _answer_directness(sentences),
        "freshness": _freshness(document, text),
    }

    if html is not None:
        values["schema_coverage"] = _schema_coverage(html)
        values["heading_structure"] = _heading_structure(html)
        values["extractability"] = _extractability(text, html)
    else:
        warnings.append("no HTML supplied: schema, heading and extractability signals are 0")
        values.update(schema_coverage=0.0, heading_structure=0.0, extractability=0.0)

    if chunks:
        values["self_containedness"] = _self_containedness(chunks)
    else:
        warnings.append("no chunks supplied: self-containedness signal is 0")
        values["self_containedness"] = 0.0

    return SignalReport(
        doc_id=document.doc_id,
        url=document.url,
        values=values,
        raw={
            "n_words": len(words),
            "n_sentences": len(sentences),
            "n_numbers": len(numbers),
            "n_qualified_statistics": len(qualified),
            "n_quotations": len(quotes),
            "n_citations": citations,
            "n_authority_markers": authority_hits,
        },
        warnings=warnings,
    )


def _saturate(value: float, ceiling: float) -> float:
    """Map [0, inf) to [0, 1], linear up to `ceiling` then flat."""
    if ceiling <= 0:
        raise ValueError("ceiling must be positive")
    return min(1.0, max(0.0, value / ceiling))


def _count_citations(document: Document, html: str | None) -> int:
    """Outbound links to a different host, plus explicit source attributions."""
    count = int(document.metadata.get("outbound_links", 0))
    if html is not None:
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
        count = sum(1 for h in hrefs if h.startswith(("http://", "https://")))
    return count + sum(document.text.lower().count(m) for m in ("source:", "according to"))


def _answer_directness(sentences: Sequence[str]) -> float:
    """Fraction of sentences that state something directly rather than hedge.

    A generative engine quotes sentences that answer; a page of marketing
    throat-clearing gives it nothing to lift.
    """
    if not sentences:
        return 0.0
    hits = sum(
        1 for s in sentences if any(m in f" {s.lower()} " for m in _DIRECT_ANSWER_MARKERS)
    )
    return hits / len(sentences)


def _freshness(document: Document, text: str) -> float:
    """Whether the page carries a usable date signal.

    Binary-ish by design: this measures whether a date is *stated*, not whether
    the content is actually current, which the crawler cannot know.
    """
    if document.metadata.get("published_at") or document.metadata.get("modified_at"):
        return 1.0
    years = _YEAR_RE.findall(text)
    return 0.5 if years else 0.0


def _schema_coverage(html: str) -> float:
    """Presence and richness of JSON-LD structured data.

    Scored on the count of distinct schema.org @type values, saturating at 4:
    a page with Organization + LocalBusiness + FAQPage + BreadcrumbList is
    doing everything a small-business site reasonably can.
    """
    blocks = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    types: set[str] = set()
    for block in blocks:
        try:
            parsed = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        types.update(_collect_types(parsed))
    return _saturate(float(len(types)), ceiling=4.0)


def _collect_types(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        value = node.get("@type")
        if isinstance(value, str):
            found.add(value)
        elif isinstance(value, list):
            found.update(str(v) for v in value)
        for child in node.values():
            found |= _collect_types(child)
    elif isinstance(node, list):
        for child in node:
            found |= _collect_types(child)
    return found


def _heading_structure(html: str) -> float:
    """Heading hierarchy validity: exactly one h1, no skipped levels, some h2s.

    Extractors use heading levels to segment a page; a page with six h1s and no
    h2s segments into one undifferentiated block.
    """
    levels = [int(m) for m in re.findall(r"<h([1-6])\b", html, flags=re.IGNORECASE)]
    if not levels:
        return 0.0

    score = 0.0
    if levels.count(1) == 1:
        score += 0.4
    elif levels.count(1) > 1:
        score += 0.1

    if any(level == 2 for level in levels):
        score += 0.3

    skips = sum(1 for a, b in zip(levels, levels[1:], strict=False) if b - a > 1)
    score += 0.3 * max(0.0, 1.0 - skips / max(1, len(levels) - 1))
    return min(1.0, score)


def _extractability(text: str, html: str) -> float:
    """Main-content share of the page's total text.

    A page whose visible text is mostly navigation, cookie banners and footer
    links dilutes every embedding computed over it. Approximated as extracted
    text length over all stripped HTML text length.
    """
    stripped = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.DOTALL | re.I)
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    total = len(re.sub(r"\s+", " ", stripped).strip())
    if total == 0:
        return 0.0
    return min(1.0, len(text) / total)


def _self_containedness(chunks: Sequence[Chunk]) -> float:
    """Fraction of chunks that open without an unresolved reference.

    A chunk starting "It also includes 24/7 support" is useless once retrieved
    alone: neither the embedder nor the generator can tell what "it" is. This
    is the signal most directly under the site owner's control and the one most
    directly tied to RAG mechanics, which is why it is measured rather than
    assumed.
    """
    if not chunks:
        return 0.0
    good = 0
    for chunk in chunks:
        sentences = split_sentences(chunk.text)
        if not sentences:
            continue
        first = sentences[0].strip()
        opener = tokenize(first.split()[0], remove_stopwords=False) if first.split() else []
        if not opener or opener[0] not in _DANGLING_OPENERS:
            good += 1
    return good / len(chunks)
