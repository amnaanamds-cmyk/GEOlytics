"""Sentence-aware chunking: pack whole sentences up to a token budget."""

from __future__ import annotations

from typing import Any

from geolytics.chunking.base import (
    Chunk,
    ChunkingStrategy,
    Document,
    TokenCounter,
    WhitespaceTokenCounter,
    make_chunk_id,
    split_sentences,
)


class SentenceChunker(ChunkingStrategy):
    """Greedily pack sentences into chunks, never splitting a sentence.

    Respects section boundaries: a chunk never spans two headings, so every
    chunk carries exactly one `heading_path`. A sentence longer than the token
    budget becomes its own oversized chunk rather than being truncated --
    silently dropping text would corrupt recall measurements.
    """

    name = "sentence"

    def __init__(
        self,
        max_tokens: int = 256,
        overlap_sentences: int = 1,
        min_tokens: int = 16,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be non-negative")
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.min_tokens = min_tokens
        self.tokens = token_counter or WhitespaceTokenCounter()

    @property
    def params(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "overlap_sentences": self.overlap_sentences,
            "min_tokens": self.min_tokens,
        }

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        ordinal = 0

        for section in document.effective_sections():
            sentences = split_sentences(section.text)
            if not sentences:
                continue

            for group in self._group(sentences):
                text = " ".join(group)
                offset = section.text.find(group[0])
                start_char = section.start_char + (offset if offset >= 0 else 0)
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(document.doc_id, self.name, ordinal, text),
                        doc_id=document.doc_id,
                        text=text,
                        ordinal=ordinal,
                        start_char=start_char,
                        end_char=start_char + len(text),
                        heading_path=section.heading_path,
                        metadata={
                            "url": document.url,
                            "title": document.title,
                            "sentences": len(group),
                        },
                    )
                )
                ordinal += 1

        return chunks

    def _group(self, sentences: list[str]) -> list[list[str]]:
        """Pack sentences into token-budgeted groups with sentence overlap."""
        groups: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            n = self.tokens.count(sentence)
            if current and current_tokens + n > self.max_tokens:
                groups.append(current)
                current = current[-self.overlap_sentences :] if self.overlap_sentences else []
                current_tokens = sum(self.tokens.count(s) for s in current)
            current.append(sentence)
            current_tokens += n

        if current:
            # Fold a trailing scrap into the previous group rather than emitting
            # a chunk too short to carry meaning.
            if groups and current_tokens < self.min_tokens:
                tail = current[self.overlap_sentences :] if self.overlap_sentences else current
                groups[-1].extend(tail)
            else:
                groups.append(current)

        return groups
