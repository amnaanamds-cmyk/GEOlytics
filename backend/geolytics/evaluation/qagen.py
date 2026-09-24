"""Synthetic query generation -- how the ground truth gets built.

The method: split each document into *generation units* with a neutral
segmentation, ask an LLM for questions answerable only from that unit, and
record the unit's span as the gold passage. The result is a judgment set
anchored to the document, not to any one chunking.

Two design choices that must survive questioning:

**The generation segmentation is fixed and separate from every chunker under
test.** If gold spans were derived from, say, the semantic chunker's output,
the semantic condition would be scored against boundaries it chose itself.
`DEFAULT_GENERATION_CHUNKER` is deliberately a plain sentence packer that is
not one of the evaluated configurations.

**Synthetic queries are lexically biased toward their source passage.** The
model paraphrases the text it was shown, so its questions share vocabulary
with the gold span, which flatters lexical retrieval (BM25) relative to what
real user queries would show. This is the main threat to validity of the whole
evaluation. Three mitigations, all implemented or supported here:

1. A paraphrase instruction in the prompt that forbids reusing distinctive
   wording (`QAGenerationConfig.avoid_verbatim`).
2. `lexical_overlap` measures the residual bias per query, so it can be
   reported rather than hoped away -- and queries above a chosen threshold can
   be dropped.
3. A human-labelled subset scored against the synthetic labels
   (`geolytics.evaluation.agreement`), which is the number that actually
   defends the method.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from geolytics.chunking.base import Chunk, ChunkingStrategy, Document
from geolytics.chunking.sentence import SentenceChunker
from geolytics.evaluation.qrels import Query, QuerySet, SpanRelevance
from geolytics.llm import LLMClient
from geolytics.retrieval.bm25 import tokenize

SYSTEM_PROMPT = (
    "You write evaluation questions for a search system. "
    "You only use the passage you are given. You never invent facts."
)

PROMPT_TEMPLATE = """\
Read the passage below and write {n} question(s) that it answers.

Rules:
- Each question must be answerable using ONLY this passage.
- Each question must make sense on its own, with no reference to "the passage",
  "this text", "above", or the document.
- Ask what a real person searching the web would ask.
{extra_rules}

Passage (from the page titled "{title}"):
\"\"\"
{passage}
\"\"\"

Return a JSON array of strings and nothing else."""

AVOID_VERBATIM_RULE = (
    "- Do NOT reuse the passage's distinctive wording. Paraphrase. If the passage\n"
    '  says "24/7 emergency plumbing", ask about "round-the-clock pipe repair".'
)

MULTI_HOP_RULE = (
    "- Make at least one question require combining two separate facts from the\n"
    "  passage, rather than reading a single sentence."
)

DEFAULT_GENERATION_CHUNKER = SentenceChunker(max_tokens=180, overlap_sentences=0, min_tokens=25)


@dataclass(slots=True)
class QAGenerationConfig:
    """Recorded with the generated set so it can be regenerated identically."""

    questions_per_unit: int = 2
    avoid_verbatim: bool = True
    multi_hop: bool = True
    temperature: float = 0.3
    max_units: int | None = None
    min_unit_tokens: int = 15
    max_lexical_overlap: float | None = 0.8

    def extra_rules(self) -> str:
        rules = []
        if self.avoid_verbatim:
            rules.append(AVOID_VERBATIM_RULE)
        if self.multi_hop and self.questions_per_unit > 1:
            rules.append(MULTI_HOP_RULE)
        return "\n".join(rules)

    def to_dict(self) -> dict[str, Any]:
        return {
            "questions_per_unit": self.questions_per_unit,
            "avoid_verbatim": self.avoid_verbatim,
            "multi_hop": self.multi_hop,
            "temperature": self.temperature,
            "max_units": self.max_units,
            "min_unit_tokens": self.min_unit_tokens,
            "max_lexical_overlap": self.max_lexical_overlap,
        }


@dataclass(slots=True)
class GenerationReport:
    """Diagnostics for one generation pass.

    `n_units_too_short` is reported because it is otherwise invisible: a site
    of short sections can have most of its content silently excluded from the
    evaluation set, leaving a query count too small to test anything. Seeing
    the number is what prompts lowering `min_unit_tokens` or merging sections.
    """

    n_units: int
    n_generated: int
    n_dropped_overlap: int
    n_failed_units: int
    mean_lexical_overlap: float
    config: dict[str, Any] = field(default_factory=dict)
    n_units_too_short: int = 0

    def summary(self) -> str:
        generator = self.config.get("generator", "llm")
        return (
            f"{self.n_generated} queries from {self.n_units} units via the "
            f"{generator} generator (mean lexical overlap "
            f"{self.mean_lexical_overlap:.2f}; {self.n_units_too_short} units below the "
            f"length threshold, {self.n_dropped_overlap} queries dropped above the "
            f"overlap threshold, {self.n_failed_units} units failed)"
        )


def lexical_overlap(query_text: str, passage_text: str) -> float:
    """Fraction of the query's content terms that also appear in the gold passage.

    1.0 means every content word was lifted from the passage -- a question BM25
    can answer without understanding anything. Report the distribution of this
    statistic; a set whose mean sits near 1.0 cannot support a claim that dense
    retrieval beats lexical retrieval.
    """
    query_terms = set(tokenize(query_text))
    if not query_terms:
        return 0.0
    passage_terms = set(tokenize(passage_text))
    return len(query_terms & passage_terms) / len(query_terms)


def eligible_units(
    documents: Sequence[Document],
    chunker: ChunkingStrategy,
    config: QAGenerationConfig,
) -> tuple[list[tuple[Document, Chunk]], int]:
    """Generation units long enough to carry an answer, plus the rejected count."""
    units: list[tuple[Document, Chunk]] = []
    too_short = 0
    for document in documents:
        for unit in chunker.chunk(document):
            if len(unit.text.split()) >= config.min_unit_tokens:
                units.append((document, unit))
            else:
                too_short += 1
    if config.max_units is not None:
        units = units[: config.max_units]
    return units, too_short


def generate_query_set(
    documents: Sequence[Document],
    llm: LLMClient,
    config: QAGenerationConfig | None = None,
    generation_chunker: ChunkingStrategy | None = None,
    name: str = "synthetic",
) -> tuple[QuerySet, GenerationReport]:
    """Generate an evaluation query set with document-anchored gold spans."""
    config = config or QAGenerationConfig()
    chunker = generation_chunker or DEFAULT_GENERATION_CHUNKER

    units, too_short = eligible_units(documents, chunker, config)

    queries: list[Query] = []
    overlaps: list[float] = []
    dropped = 0
    failed = 0

    for document, unit in units:
        try:
            questions = _ask(llm, document, unit, config)
        except Exception:  # noqa: BLE001 - one bad unit must not kill the pass
            failed += 1
            continue

        for question in questions:
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
                    provenance="synthetic",
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
        config={**config.to_dict(), "generator": "llm"},
        n_units_too_short=too_short,
    )
    query_set = QuerySet(
        name=name,
        queries=queries,
        metadata={
            "generation_chunker": chunker.describe(),
            "llm": llm.describe(),
            **config.to_dict(),
        },
    )
    return query_set, report


def _ask(
    llm: LLMClient, document: Document, unit: Chunk, config: QAGenerationConfig
) -> list[str]:
    prompt = PROMPT_TEMPLATE.format(
        n=config.questions_per_unit,
        title=document.title or document.url,
        passage=unit.text,
        extra_rules=config.extra_rules(),
    )
    parsed = llm.complete_json(prompt, system=SYSTEM_PROMPT, temperature=config.temperature)
    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON array of questions, got {type(parsed).__name__}")

    out: list[str] = []
    for item in parsed[: config.questions_per_unit]:
        text = item.strip() if isinstance(item, str) else str(item.get("question", "")).strip()
        if text:
            out.append(text)
    return out


def _query_id(unit_id: str, question: str) -> str:
    digest = hashlib.sha1(f"{unit_id}|{question}".encode()).hexdigest()[:12]
    return f"q_{digest}"
