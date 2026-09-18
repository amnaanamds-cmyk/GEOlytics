"""Fixed-size sliding-window chunking -- the baseline strategy."""

from __future__ import annotations

from typing import Any

from geolytics.chunking.base import (
    Chunk,
    ChunkingStrategy,
    Document,
    TokenCounter,
    WhitespaceTokenCounter,
    make_chunk_id,
)


class FixedSizeChunker(ChunkingStrategy):
    """Slice the document into equal token windows with a fixed overlap.

    This is the baseline every other strategy is compared against. It ignores
    sentence and section boundaries entirely, which is exactly the property
    under test: a chunk that starts or ends mid-sentence is less
    self-contained, and the hypothesis is that this costs retrieval quality.
    """

    name = "fixed"

    def __init__(
        self,
        chunk_tokens: int = 256,
        overlap_tokens: int = 32,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if not 0 <= overlap_tokens < chunk_tokens:
            raise ValueError("overlap_tokens must satisfy 0 <= overlap < chunk_tokens")
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.tokens = token_counter or WhitespaceTokenCounter()

    @property
    def params(self) -> dict[str, Any]:
        return {"chunk_tokens": self.chunk_tokens, "overlap_tokens": self.overlap_tokens}

    def chunk(self, document: Document) -> list[Chunk]:
        words = self.tokens.split(document.text)
        if not words:
            return []

        stride = self.chunk_tokens - self.overlap_tokens
        chunks: list[Chunk] = []

        for ordinal, start in enumerate(range(0, len(words), stride)):
            window = words[start : start + self.chunk_tokens]
            if not window:
                break
            text = " ".join(window)
            # Character offsets are approximate for this strategy: the window is
            # defined over tokens, and re-joining normalises internal whitespace.
            start_char = len(" ".join(words[:start])) + (1 if start else 0)
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.doc_id, self.name, ordinal, text),
                    doc_id=document.doc_id,
                    text=text,
                    ordinal=ordinal,
                    start_char=start_char,
                    end_char=start_char + len(text),
                    heading_path=(),
                    metadata={"url": document.url, "title": document.title},
                )
            )
            if start + self.chunk_tokens >= len(words):
                break

        return chunks
