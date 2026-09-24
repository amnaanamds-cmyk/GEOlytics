"""Query generation without an LLM.

Purpose: make the whole research pipeline runnable from a clean clone, in CI,
and on a machine with no model server. `geolytics experiment` should not fail
because Ollama is not installed.

**This is a fallback, not the method.** Template-generated questions reuse the
passage's own vocabulary, so their lexical overlap with the gold span is high
by construction — which flatters BM25 and any hybrid retriever containing it.
Queries produced here are marked `provenance="heuristic"` so they can never be
silently mixed into a reported result, and `generate_heuristic_query_set`
returns the same `GenerationReport` as the LLM path, including the measured
overlap distribution.

Use it to develop, to smoke-test a pipeline change, and as a deterministic
regression baseline. Use the LLM path (and the human-labelled validation in
`agreement.py`) for anything that goes in the report.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence

from geolytics.chunking.base import Chunk, ChunkingStrategy, Document, split_sentences
from geolytics.evaluation.qagen import (
    DEFAULT_GENERATION_CHUNKER,
    GenerationReport,
    QAGenerationConfig,
    eligible_units,
    lexical_overlap,
)
from geolytics.evaluation.qrels import Query, QuerySet, SpanRelevance
from geolytics.retrieval.bm25 import tokenize

# Cues that decide which question shape fits a passage. Ordered by specificity:
# the first matching template wins, so "how much does X cost" beats "what is X"
# on a pricing passage.
_COST_CUES = ("cost", "costs", "price", "prices", "priced", "fee", "fees", "pkr",
              "usd", "rate", "rates", "charge", "charges", "billed", "quote")
_DURATION_CUES = ("minutes", "hours", "days", "weeks", "months", "takes", "within",
                  "turnaround", "lead", "response")
_COVERAGE_CUES = ("covers", "covering", "includes", "included", "including",
                  "offers", "provides", "available", "serves", "supports")
_PROCESS_CUES = ("uses", "using", "process", "step", "steps", "works", "performed",
                 "carried", "installed", "fitted", "measured")

_TEMPLATES: tuple[tuple[tuple[str, ...], str], ...] = (
    (_COST_CUES, "how much does {focus} cost"),
    (_DURATION_CUES, "how long does {focus} take"),
    (_COVERAGE_CUES, "what does {focus} include"),
    (_PROCESS_CUES, "how is {focus} done"),
)
_FALLBACK_TEMPLATE = "what is {focus}"

_NUMERIC_RE = re.compile(r"\d")


def generate_heuristic_query_set(
    documents: Sequence[Document],
    config: QAGenerationConfig | None = None,
    generation_chunker: ChunkingStrategy | None = None,
    name: str = "heuristic",
) -> tuple[QuerySet, GenerationReport]:
    """Build an evaluation query set from templates and corpus salience.

    Focus terms are chosen by a TF-IDF-style score over the generation units:
    a term that is frequent in this unit but rare across the corpus identifies
    the unit, which is exactly what a query needs to do. Using corpus-wide
    document frequency is what stops every question becoming "what is Acme".
    """
    config = config or QAGenerationConfig()
    chunker = generation_chunker or DEFAULT_GENERATION_CHUNKER

    units, too_short = eligible_units(documents, chunker, config)

    if not units:
        return (
            QuerySet(name=name, queries=[], metadata={"generator": "heuristic"}),
            GenerationReport(
                n_units=0,
                n_generated=0,
                n_dropped_overlap=0,
                n_failed_units=0,
                mean_lexical_overlap=0.0,
                config={**config.to_dict(), "generator": "heuristic"},
                n_units_too_short=too_short,
            ),
        )

    idf = _corpus_idf([unit.text for _, unit in units])

    queries: list[Query] = []
    overlaps: list[float] = []
    dropped = 0
    failed = 0
    seen: set[str] = set()

    for document, unit in units:
        questions = _questions_for(unit, idf, config.questions_per_unit)
        if not questions:
            failed += 1
            continue

        for question in questions:
            if question in seen:
                # Two units can yield the same template question; a duplicate
                # query with a different gold span would make the judgment set
                # self-contradictory.
                continue
            seen.add(question)

            overlap = lexical_overlap(question, unit.text)
            overlaps.append(overlap)
            if (
                config.max_lexical_overlap is not None
                and overlap > config.max_lexical_overlap
            ):
                dropped += 1
                continue

            queries.append(
                Query(
                    query_id=_query_id(unit.chunk_id, question),
                    text=question,
                    spans=(
                        SpanRelevance(
                            doc_id=document.doc_id,
                            start_char=unit.start_char,
                            end_char=unit.end_char,
                            grade=1,
                            text=unit.text,
                        ),
                    ),
                    provenance="heuristic",
                    source_doc_id=document.doc_id,
                    metadata={
                        "lexical_overlap": round(overlap, 4),
                        "generation_unit": unit.chunk_id,
                        "url": document.url,
                    },
                )
            )

    report = GenerationReport(
        n_units=len(units),
        n_generated=len(queries),
        n_dropped_overlap=dropped,
        n_failed_units=failed,
        mean_lexical_overlap=(sum(overlaps) / len(overlaps)) if overlaps else 0.0,
        config={**config.to_dict(), "generator": "heuristic"},
        n_units_too_short=too_short,
    )
    query_set = QuerySet(
        name=name,
        queries=queries,
        metadata={
            "generation_chunker": chunker.describe(),
            "generator": "heuristic",
            "warning": (
                "Template-generated queries. Lexically biased toward the source "
                "passage by construction; not suitable for reported results."
            ),
            **config.to_dict(),
        },
    )
    return query_set, report


def _corpus_idf(texts: Sequence[str]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for text in texts:
        document_frequency.update(set(tokenize(text)))
    n = len(texts)
    return {
        term: math.log((n + 1) / (freq + 0.5))
        for term, freq in document_frequency.items()
    }


def _questions_for(unit: Chunk, idf: dict[str, float], limit: int) -> list[str]:
    """One question per distinct template that the passage's cues support."""
    focus = _focus_phrase(unit, idf)
    if not focus:
        return []

    lowered = unit.text.lower()
    questions: list[str] = []
    for cues, template in _TEMPLATES:
        if any(cue in lowered for cue in cues):
            # A cost template on a passage with no digits would ask about a
            # price the passage never states.
            if cues is _COST_CUES and not _NUMERIC_RE.search(unit.text):
                continue
            questions.append(template.format(focus=focus))
        if len(questions) >= limit:
            break

    if not questions:
        questions.append(_FALLBACK_TEMPLATE.format(focus=focus))
    return questions[:limit]


def _focus_phrase(unit: Chunk, idf: dict[str, float], max_terms: int = 3) -> str:
    """The most identifying terms in this unit, in their original order.

    Preserving reading order matters: "emergency bathroom callout" is not a
    phrase anyone would search, whereas the source order usually is.
    """
    if unit.heading_path:
        # A heading is an author-written topic label — better than anything
        # term statistics will recover.
        return unit.heading_path[-1].lower()

    sentences = split_sentences(unit.text)
    head = sentences[0] if sentences else unit.text
    terms = tokenize(head)
    if not terms:
        return ""

    counts = Counter(terms)
    scored = {t: counts[t] * idf.get(t, 0.0) for t in set(terms)}
    best = {t for t, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:max_terms]}

    ordered = [t for t in dict.fromkeys(terms) if t in best]
    return " ".join(ordered)


def _query_id(unit_id: str, question: str) -> str:
    digest = hashlib.sha1(f"{unit_id}|{question}".encode()).hexdigest()[:12]
    return f"h_{digest}"
