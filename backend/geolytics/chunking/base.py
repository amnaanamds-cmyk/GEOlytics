"""Core chunking types and the strategy interface."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol


class TokenCounter(Protocol):
    """Counts tokens in a string.

    The default implementation counts whitespace-delimited words, which is
    dependency-free but only approximates a model tokenizer (roughly 0.75
    words per token for English). Plug in a real tokenizer before quoting
    absolute token budgets in the report; relative comparisons between
    chunkers are unaffected as long as the same counter is used throughout.
    """

    def count(self, text: str) -> int: ...

    def split(self, text: str) -> list[str]: ...


class WhitespaceTokenCounter:
    """Approximate token counter with no external dependency."""

    _WORD_RE = re.compile(r"\S+")

    def count(self, text: str) -> int:
        return len(self._WORD_RE.findall(text))

    def split(self, text: str) -> list[str]:
        return self._WORD_RE.findall(text)


@dataclass(frozen=True, slots=True)
class Section:
    """A heading-delimited block of a page.

    `heading_path` is the chain of ancestor headings (e.g.
    ``["Pricing", "Enterprise plan"]``). Chunkers propagate it into chunk
    metadata so that a retrieved chunk can be rendered with context and so
    that the GEO layer can score heading hierarchy.
    """

    text: str
    heading_path: tuple[str, ...] = ()
    start_char: int = 0

    @property
    def end_char(self) -> int:
        return self.start_char + len(self.text)


@dataclass(frozen=True, slots=True)
class Document:
    """An extracted page, ready to be chunked."""

    doc_id: str
    url: str
    title: str
    text: str
    sections: tuple[Section, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def effective_sections(self) -> tuple[Section, ...]:
        """Sections if the extractor produced them, else the whole document."""
        return self.sections or (Section(text=self.text, heading_path=(), start_char=0),)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A unit of retrieval.

    `parent_id` is set only by hierarchical strategies: the chunk is *indexed*
    by its own text but *returned* as its parent's text (see
    `ParentChildChunker`). Evaluation resolves relevance against `chunk_id`,
    so a parent-child run judges the child that was matched.
    """

    chunk_id: str
    doc_id: str
    text: str
    ordinal: int
    start_char: int
    end_char: int
    heading_path: tuple[str, ...] = ()
    parent_id: str | None = None
    parent_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context_text(self) -> str:
        """Text handed to the generator: the parent when one exists."""
        return self.parent_text if self.parent_text is not None else self.text


def make_chunk_id(doc_id: str, strategy: str, ordinal: int, text: str) -> str:
    """Deterministic chunk id.

    Includes the strategy name so that two strategies producing byte-identical
    text on the same document still get distinct ids -- otherwise a shared
    Qdrant collection would silently collapse them.
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}:{strategy}:{ordinal:04d}:{digest}"


class ChunkingStrategy(ABC):
    """Interface every chunking strategy implements."""

    name: str = "base"

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split `document` into retrieval units, in reading order."""

    @property
    def params(self) -> dict[str, Any]:
        """Configuration recorded alongside every experimental result."""
        return {}

    def describe(self) -> dict[str, Any]:
        return {"strategy": self.name, "params": self.params}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"


# --- Sentence segmentation -------------------------------------------------

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "inc", "ltd", "co", "corp", "dept", "est", "fig", "no", "vol",
}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    A regex segmenter, chosen so the chunking layer has no NLP dependency.
    It handles the common abbreviation and decimal-number false positives. For
    the final report, swap in a trained segmenter (spaCy / PySBD) and note the
    change -- sentence boundaries shift the sentence and semantic chunkers, so
    the swap must happen before the experiments are run, not between them.
    """
    text = text.strip()
    if not text:
        return []

    pieces = _SENTENCE_BOUNDARY.split(text)
    sentences: list[str] = []
    buffer = ""

    for piece in pieces:
        candidate = f"{buffer} {piece}".strip() if buffer else piece.strip()
        if not candidate:
            continue
        if _ends_on_false_boundary(candidate):
            buffer = candidate
            continue
        sentences.append(candidate)
        buffer = ""

    if buffer:
        sentences.append(buffer)
    return sentences


def _ends_on_false_boundary(candidate: str) -> bool:
    """True when a '.' ends an abbreviation or a decimal, not a sentence."""
    stripped = candidate.rstrip("\"')]")
    if not stripped.endswith("."):
        return False
    last = stripped[:-1].split()[-1].lower() if stripped[:-1].split() else ""
    if last in _ABBREVIATIONS:
        return True
    # A single capital letter ("J. Smith") or a bare number ("Section 3.").
    return len(last) == 1 and last.isalpha()
