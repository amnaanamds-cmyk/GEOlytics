"""Semantic chunking: cut where consecutive sentences stop being similar."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from geolytics.chunking.base import (
    Chunk,
    ChunkingStrategy,
    Document,
    TokenCounter,
    WhitespaceTokenCounter,
    make_chunk_id,
    split_sentences,
)

if TYPE_CHECKING:  # pragma: no cover
    from geolytics.embedding.base import Embedder


class SemanticChunker(ChunkingStrategy):
    """Breakpoint-percentile semantic chunking.

    Embeds each sentence in the context of its neighbours, measures the cosine
    distance between consecutive sentence embeddings, and cuts at the distances
    above the `breakpoint_percentile`th percentile -- i.e. where the topic
    shifts most sharply.

    Two properties matter for the experiment:

    * The threshold is a *percentile of this document's own distances*, not an
      absolute value, so a uniformly on-topic page is not shredded and a
      wide-ranging page is not left as one chunk.
    * Chunk sizes are data-dependent, so a semantic run is not size-matched to
      the fixed baseline. Report mean chunk length alongside the retrieval
      metrics -- otherwise a quality difference could simply be a size effect,
      and a reviewer will ask.
    """

    name = "semantic"

    def __init__(
        self,
        embedder: Embedder,
        breakpoint_percentile: float = 90.0,
        buffer_sentences: int = 1,
        max_tokens: int = 512,
        min_sentences: int = 2,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if not 0 < breakpoint_percentile < 100:
            raise ValueError("breakpoint_percentile must be in (0, 100)")
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_sentences = max(0, buffer_sentences)
        self.max_tokens = max_tokens
        self.min_sentences = max(1, min_sentences)
        self.tokens = token_counter or WhitespaceTokenCounter()

    @property
    def params(self) -> dict[str, Any]:
        return {
            "breakpoint_percentile": self.breakpoint_percentile,
            "buffer_sentences": self.buffer_sentences,
            "max_tokens": self.max_tokens,
            "min_sentences": self.min_sentences,
            "embedder": self.embedder.describe(),
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
        if len(sentences) <= self.min_sentences:
            return [sentences]

        distances = self._consecutive_distances(sentences)
        if distances.size == 0:
            return [sentences]

        threshold = float(np.percentile(distances, self.breakpoint_percentile))
        # Index i in `distances` is the gap between sentence i and i+1, so a
        # breakpoint at i means the next group starts at i+1.
        breakpoints = [i + 1 for i, d in enumerate(distances) if d > threshold]

        groups: list[list[str]] = []
        start = 0
        for cut in [*breakpoints, len(sentences)]:
            group = sentences[start:cut]
            if group:
                groups.append(group)
            start = cut

        return [split for group in groups for split in self._enforce_max_tokens(group)]

    def _consecutive_distances(self, sentences: list[str]) -> np.ndarray:
        """Cosine distance between each adjacent pair of context windows."""
        windows = [self._window(sentences, i) for i in range(len(sentences))]
        vectors = np.asarray(self.embedder.embed(windows), dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # A zero vector has no direction; treat it as maximally distant rather
        # than dividing by zero.
        safe = np.where(norms == 0, 1.0, norms)
        unit = vectors / safe
        similarities = np.sum(unit[:-1] * unit[1:], axis=1)
        degenerate = (norms[:-1, 0] == 0) | (norms[1:, 0] == 0)
        similarities = np.where(degenerate, -1.0, similarities)
        return 1.0 - similarities

    def _window(self, sentences: list[str], i: int) -> str:
        lo = max(0, i - self.buffer_sentences)
        hi = min(len(sentences), i + self.buffer_sentences + 1)
        return " ".join(sentences[lo:hi])

    def _enforce_max_tokens(self, group: list[str]) -> list[list[str]]:
        """Hard-split a semantically coherent but over-budget group."""
        out: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in group:
            n = self.tokens.count(sentence)
            if current and current_tokens + n > self.max_tokens:
                out.append(current)
                current, current_tokens = [], 0
            current.append(sentence)
            current_tokens += n
        if current:
            out.append(current)
        return out
