"""Vector store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from geolytics.chunking.base import Chunk

Metric = Literal["cosine", "dot", "euclidean"]


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A retrieval result. Higher `score` is always better.

    Distance-based metrics are negated on the way out so that every retriever,
    fusion step and metric in the system can assume descending-by-score order.
    """

    chunk: Chunk
    score: float
    rank: int

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class VectorStore(ABC):
    """Stores chunk vectors and answers nearest-neighbour queries."""

    @abstractmethod
    def create_collection(self, name: str, dim: int, metric: Metric = "cosine") -> None: ...

    @abstractmethod
    def upsert(self, name: str, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(
        self,
        name: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        metric: Metric | None = None,
    ) -> list[ScoredChunk]: ...

    @abstractmethod
    def drop_collection(self, name: str) -> None: ...

    def describe(self) -> dict[str, Any]:
        return {"store": type(self).__name__}
