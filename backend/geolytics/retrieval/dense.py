"""Dense vector retrieval."""

from __future__ import annotations

from typing import Any

from geolytics.embedding.base import Embedder
from geolytics.index.base import Metric, ScoredChunk, VectorStore
from geolytics.retrieval.base import Retriever


class DenseRetriever(Retriever):
    """Embed the query, nearest-neighbour search the collection."""

    name = "dense"

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        collection: str,
        metric: Metric | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.collection = collection
        self.metric = metric

    @property
    def params(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "metric": self.metric,
            "embedder": self.embedder.describe(),
        }

    def retrieve(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        vector = self.embedder.embed_query([query])[0]
        return self.store.search(self.collection, vector, top_k=top_k, metric=self.metric)
