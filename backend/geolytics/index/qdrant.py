"""Qdrant-backed vector store (the deployed application's index)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

import numpy as np

from geolytics.chunking.base import Chunk
from geolytics.index.base import Metric, ScoredChunk, VectorStore

_DISTANCE = {"cosine": "Cosine", "dot": "Dot", "euclidean": "Euclid"}


class QdrantVectorStore(VectorStore):
    """Wraps `qdrant-client`.

    Note for the similarity-metric discussion in the report: Qdrant normalises
    vectors on upsert when a collection is created with `Cosine` distance. A
    `Cosine` collection and a `Dot` collection built from the same normalised
    embeddings therefore return identical rankings -- this is a property of the
    store as well as of the mathematics.
    """

    def __init__(self, url: str, api_key: str | None = None, timeout: float = 30.0) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "QdrantVectorStore requires the 'vector' extra: pip install -e '.[vector]'"
            ) from exc
        self._client = QdrantClient(url=url, api_key=api_key, timeout=int(timeout))

    def create_collection(self, name: str, dim: int, metric: Metric = "cosine") -> None:
        from qdrant_client.models import Distance, VectorParams

        self._client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance(_DISTANCE[metric])),
        )

    def drop_collection(self, name: str) -> None:
        self._client.delete_collection(collection_name=name)

    def upsert(self, name: str, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        vectors = np.asarray(vectors, dtype=np.float32)
        points = [
            PointStruct(
                # Qdrant ids must be UUID or unsigned int; the human-readable
                # chunk id is kept in the payload and restored on read.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=vector.tolist(),
                payload=_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=name, points=points, wait=True)

    def search(
        self,
        name: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        metric: Metric | None = None,
    ) -> list[ScoredChunk]:
        # `metric` is fixed at collection creation in Qdrant; a per-query
        # override would silently be ignored, so reject it loudly instead.
        if metric is not None:
            raise ValueError(
                "Qdrant fixes the distance metric per collection; "
                "create one collection per metric instead of overriding at query time"
            )
        hits = self._client.search(
            collection_name=name,
            query_vector=np.asarray(query_vector, dtype=np.float32).reshape(-1).tolist(),
            limit=top_k,
            with_payload=True,
        )
        return [
            ScoredChunk(chunk=_chunk_from_payload(hit.payload or {}), score=float(hit.score),
                        rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]

    def describe(self) -> dict[str, Any]:
        return {"store": "qdrant"}


def _payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "ordinal": chunk.ordinal,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "heading_path": list(chunk.heading_path),
        "parent_id": chunk.parent_id,
        "parent_text": chunk.parent_text,
        "metadata": chunk.metadata,
    }


def _chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=payload.get("chunk_id", ""),
        doc_id=payload.get("doc_id", ""),
        text=payload.get("text", ""),
        ordinal=int(payload.get("ordinal", 0)),
        start_char=int(payload.get("start_char", 0)),
        end_char=int(payload.get("end_char", 0)),
        heading_path=tuple(payload.get("heading_path") or ()),
        parent_id=payload.get("parent_id"),
        parent_text=payload.get("parent_text"),
        metadata=payload.get("metadata") or {},
    )
