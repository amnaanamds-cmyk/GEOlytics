"""Project document-level gold spans onto a specific chunking.

Called once per (chunking strategy, query set) pair, immediately before
scoring. This is what makes a single authored judgment set comparable across
chunkers.

Matching is deliberately two-sided:

* **Offset overlap** is the primary signal -- exact, and cheap.
* **Normalised text containment** is the fallback, because not every chunker
  can report faithful character offsets (the fixed-size chunker windows over
  tokens and re-joins, normalising whitespace).

A chunk is judged relevant when it carries enough of the gold passage to
actually answer the query. "Enough" is `min_coverage`, defaulting to 0.5. Two
things follow that must be stated in the report:

1. A large chunk can satisfy the threshold while burying the answer in
   unrelated text. That is a real property of coarse chunking, not a bug in
   the measurement -- but it means coarse strategies get a mild advantage on
   recall. Report mean chunk length beside every result table so a reader can
   see it.
2. The threshold is a free parameter. Fix it once, before running the
   experiments, and re-run the headline comparison at 0.3 and 0.7 as a
   sensitivity check.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from geolytics.chunking.base import Chunk
from geolytics.evaluation.qrels import Query, QuerySet, Relevance, SpanRelevance

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace, for offset-free containment matching."""
    return _WS_RE.sub(" ", text).strip().lower()


@dataclass(frozen=True, slots=True)
class ProjectionStats:
    """Diagnostics for one projection -- report these, do not discard them."""

    n_queries: int
    n_projected: int
    n_unmatched: int
    n_judgments: int
    mean_judgments_per_query: float

    @property
    def match_rate(self) -> float:
        return self.n_projected / self.n_queries if self.n_queries else 0.0


def project_query(
    query: Query,
    chunks: Sequence[Chunk],
    min_coverage: float = 0.5,
) -> Query:
    """Return `query` with `judgments` computed against `chunks`."""
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")

    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    best: dict[str, int] = {}
    for span in query.spans:
        for chunk in by_doc.get(span.doc_id, ()):
            # A chunk matching several spans keeps the highest grade.
            if _covers(chunk, span, min_coverage) and span.grade > best.get(chunk.chunk_id, 0):
                best[chunk.chunk_id] = span.grade

    return query.with_judgments(
        Relevance(chunk_id=cid, grade=grade) for cid, grade in sorted(best.items())
    )


def project_query_set(
    query_set: QuerySet,
    chunks: Sequence[Chunk],
    min_coverage: float = 0.5,
) -> tuple[QuerySet, ProjectionStats]:
    """Project every query, returning the new set and its diagnostics."""
    projected = [project_query(q, chunks, min_coverage) for q in query_set]
    n_judgments = sum(q.n_relevant for q in projected)
    matched = sum(1 for q in projected if q.n_relevant > 0)

    stats = ProjectionStats(
        n_queries=len(projected),
        n_projected=matched,
        n_unmatched=len(projected) - matched,
        n_judgments=n_judgments,
        mean_judgments_per_query=(n_judgments / matched) if matched else 0.0,
    )
    return (
        QuerySet(
            name=query_set.name,
            queries=projected,
            metadata={**query_set.metadata, "min_coverage": min_coverage},
        ),
        stats,
    )


def _covers(chunk: Chunk, span: SpanRelevance, min_coverage: float) -> bool:
    """Does `chunk` carry at least `min_coverage` of `span`?"""
    if span.length > 0:
        overlap = min(chunk.end_char, span.end_char) - max(chunk.start_char, span.start_char)
        if overlap > 0 and overlap / span.length >= min_coverage:
            return True

    gold = span.text
    if not gold:
        return False

    # Offsets disagreed (or were never reliable); fall back to text.
    haystack = normalize(chunk.text)
    needle = normalize(gold)
    if not needle:
        return False
    if needle in haystack:
        return True

    # Partial containment: the longest prefix of the gold passage present in
    # the chunk, which is what a chunk boundary cutting mid-span produces.
    return _prefix_coverage(haystack, needle) >= min_coverage


def _prefix_coverage(haystack: str, needle: str) -> float:
    """Fraction of `needle` present as a contiguous prefix or suffix of a match."""
    words = needle.split()
    if not words:
        return 0.0

    # Longest prefix of the gold passage that appears in the chunk.
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if " ".join(words[:mid]) in haystack:
            lo = mid
        else:
            hi = mid - 1
    prefix = lo

    # Longest suffix, for a chunk holding the tail of the passage.
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if " ".join(words[len(words) - mid :]) in haystack:
            lo = mid
        else:
            hi = mid - 1
    suffix = lo

    return max(prefix, suffix) / len(words)
