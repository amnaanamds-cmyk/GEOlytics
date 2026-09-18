"""Retriever interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from geolytics.index.base import ScoredChunk


class Retriever(ABC):
    """Maps a query string to a ranked list of chunks."""

    name: str = "base"

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        """Return up to `top_k` chunks, best first, ranks starting at 1."""

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def describe(self) -> dict[str, Any]:
        return {"retriever": self.name, "params": self.params}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"


def renumber(results: list[ScoredChunk]) -> list[ScoredChunk]:
    """Reassign 1-based ranks after a re-sort or a fusion step."""
    return [
        ScoredChunk(chunk=r.chunk, score=r.score, rank=i)
        for i, r in enumerate(results, start=1)
    ]
