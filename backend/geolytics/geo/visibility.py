"""Visibility metrics over a generated answer.

The unit of GEO success is not "did the page rank" but "did the generated
answer use this page, and how prominently". These metrics follow the impression
framework of Aggarwal et al. (KDD 2024): measure the *share of the answer*
attributable to a source, weighting earlier material more heavily because a
reader's attention decays down the answer.

Scope: these measure the answer produced by *this system's* simulated
generative engine. They say nothing directly about ChatGPT, Perplexity or AI
Overviews, whose retrieval and ranking are not observable from outside. The
external-validity step is separate: run the same queries against real engines,
record whether the site appears, and correlate that against the score predicted
here. Report a correlation, never a causal attribution.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from geolytics.retrieval.bm25 import tokenize

Decay = Literal["linear", "exponential", "none"]


@dataclass(frozen=True, slots=True)
class AnswerSentence:
    """One sentence of a generated answer and the chunks it cited."""

    text: str
    cited_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ImpressionMetrics:
    """How much of one generated answer is attributable to the target site."""

    n_sentences: int
    n_words: int
    word_count_share: float
    position_adjusted_share: float
    citation_count_share: float
    first_cited_position: int | None
    cited_chunk_ids: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def cited(self) -> bool:
        return self.first_cited_position is not None

    def summary(self) -> str:
        if not self.cited:
            return "not cited in the generated answer"
        return (
            f"cited from sentence {self.first_cited_position}; "
            f"{self.word_count_share:.1%} of answer words, "
            f"{self.position_adjusted_share:.1%} position-adjusted, "
            f"{self.citation_count_share:.1%} of citations"
        )


def citation_visibility(
    sentences: Sequence[AnswerSentence],
    target_chunk_ids: Sequence[str] | set[str],
    decay: Decay = "exponential",
    half_life: float = 5.0,
) -> ImpressionMetrics:
    """Compute impression metrics for one answer.

    `decay` controls the position weighting. "exponential" halves a sentence's
    weight every `half_life` sentences, which matches the way answer text is
    read and skimmed better than a linear ramp; "none" reduces the metric to a
    plain word-count share, which is the right ablation to report alongside it.
    """
    target = set(target_chunk_ids)
    if not sentences:
        return ImpressionMetrics(0, 0, 0.0, 0.0, 0.0, None, notes=["empty answer"])

    word_counts = [len(tokenize(s.text, remove_stopwords=False)) for s in sentences]
    weights = [_weight(i, decay, half_life) for i in range(len(sentences))]

    total_words = sum(word_counts)
    total_weighted = sum(w * c for w, c in zip(weights, word_counts, strict=True))
    total_citations = sum(len(s.cited_chunk_ids) for s in sentences)

    hit_words = 0
    hit_weighted = 0.0
    hit_citations = 0
    first_position: int | None = None
    cited: list[str] = []

    for i, (sentence, count, weight) in enumerate(
        zip(sentences, word_counts, weights, strict=True), start=1
    ):
        matched = [cid for cid in sentence.cited_chunk_ids if cid in target]
        hit_citations += len(matched)
        if matched:
            cited.extend(matched)
            hit_words += count
            hit_weighted += weight * count
            if first_position is None:
                first_position = i

    return ImpressionMetrics(
        n_sentences=len(sentences),
        n_words=total_words,
        word_count_share=(hit_words / total_words) if total_words else 0.0,
        position_adjusted_share=(hit_weighted / total_weighted) if total_weighted else 0.0,
        citation_count_share=(hit_citations / total_citations) if total_citations else 0.0,
        first_cited_position=first_position,
        cited_chunk_ids=tuple(dict.fromkeys(cited)),
    )


def _weight(index: int, decay: Decay, half_life: float) -> float:
    if decay == "none":
        return 1.0
    if decay == "linear":
        return 1.0 / (index + 1)
    if decay == "exponential":
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        return math.exp(-math.log(2.0) * index / half_life)
    raise ValueError(f"unknown decay {decay!r}")
