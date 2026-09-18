"""NumPy-backed vector store.

Exact brute-force search over a few thousand chunks, which is the regime a
single-site audit lives in. Two reasons it is the default for experiments:

1. No approximate-nearest-neighbour recall loss. HNSW is a *second* source of
   variance on top of the factor under test; measuring chunking strategies
   against an exact index isolates the effect being studied.
2. No running service, so the evaluation suite is reproducible from a clone.

Switch to `QdrantVectorStore` for the deployed application, and -- if the
report claims the two agree -- measure the ANN recall gap rather than assuming
it away.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from geolytics.chunking.base import Chunk
from geolytics.index.base import Metric, ScoredChunk, VectorStore


@dataclass
class _Collection:
    dim: int
    metric: Metric
    chunks: list[Chunk] = field(default_factory=list)
    vectors: np.ndarray | None = None
    _index: dict[str, int] = field(default_factory=dict)


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._collections: dict[str, _Collection] = {}

    def create_collection(self, name: str, dim: int, metric: Metric = "cosine") -> None:
        self._collections[name] = _Collection(dim=dim, metric=metric)

    def drop_collection(self, name: str) -> None:
        self._collections.pop(name, None)

    def upsert(self, name: str, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        collection = self._require(name)
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"got {len(chunks)} chunks but {vectors.shape[0]} vectors"
            )
        if vectors.shape[1] != collection.dim:
            raise ValueError(
                f"collection {name!r} has dim {collection.dim}, got {vectors.shape[1]}"
            )

        for chunk, vector in zip(chunks, vectors, strict=True):
            existing = collection._index.get(chunk.chunk_id)
            if existing is None:
                collection._index[chunk.chunk_id] = len(collection.chunks)
                collection.chunks.append(chunk)
                collection.vectors = (
                    vector[None, :]
                    if collection.vectors is None
                    else np.vstack([collection.vectors, vector[None, :]])
                )
            else:
                collection.chunks[existing] = chunk
                assert collection.vectors is not None
                collection.vectors[existing] = vector

    def search(
        self,
        name: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        metric: Metric | None = None,
    ) -> list[ScoredChunk]:
        collection = self._require(name)
        if collection.vectors is None or not collection.chunks:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        scores = score_matrix(collection.vectors, query, metric or collection.metric)

        k = min(top_k, scores.shape[0])
        # argpartition is O(n); the slice is then sorted exactly.
        candidates = np.argpartition(-scores, k - 1)[:k]
        order = candidates[np.argsort(-scores[candidates], kind="stable")]
        return [
            ScoredChunk(chunk=collection.chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]

    def count(self, name: str) -> int:
        return len(self._require(name).chunks)

    def _require(self, name: str) -> _Collection:
        try:
            return self._collections[name]
        except KeyError:
            raise KeyError(f"collection {name!r} does not exist") from None


def score_matrix(vectors: np.ndarray, query: np.ndarray, metric: Metric) -> np.ndarray:
    """Similarity of every row of `vectors` to `query`, higher is better."""
    if metric == "dot":
        return vectors @ query
    if metric == "cosine":
        norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
        return np.divide(
            vectors @ query, norms, out=np.zeros(vectors.shape[0]), where=norms > 0
        )
    if metric == "euclidean":
        # Negated: callers sort descending by score.
        return -np.linalg.norm(vectors - query, axis=1)
    raise ValueError(f"unknown metric {metric!r}")
