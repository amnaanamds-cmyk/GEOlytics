"""Cross-encoder reranking."""

from __future__ import annotations

from typing import Any, Protocol

from geolytics.index.base import ScoredChunk
from geolytics.retrieval.base import Retriever, renumber


class CrossEncoderScorer(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class SentenceTransformerCrossEncoder:
    """Wraps a `sentence-transformers` CrossEncoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", batch_size: int = 16) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "reranking requires the 'embed' extra: pip install -e '.[embed]'"
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, t) for t in texts]
        return [float(s) for s in self._model.predict(pairs, batch_size=self.batch_size)]


class RerankingRetriever(Retriever):
    """Retrieve `candidate_k`, rescore with a cross-encoder, keep `top_k`.

    The cross-encoder sees the query and the passage jointly, so it can model
    interactions a bi-encoder cannot -- at the cost of a forward pass per
    candidate. `candidate_k` is the accuracy/latency dial and should be
    reported: a reranker over 50 candidates is a different system from one
    over 10, and the recall ceiling is whatever the first stage achieved at
    `candidate_k`.
    """

    name = "reranked"

    def __init__(
        self,
        base: Retriever,
        scorer: CrossEncoderScorer,
        candidate_k: int = 50,
    ) -> None:
        self.base = base
        self.scorer = scorer
        self.candidate_k = candidate_k

    @property
    def params(self) -> dict[str, Any]:
        return {"candidate_k": self.candidate_k, "base": self.base.describe()}

    def retrieve(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        candidates = self.base.retrieve(query, top_k=max(self.candidate_k, top_k))
        if not candidates:
            return []

        scores = self.scorer.score(query, [c.chunk.text for c in candidates])
        rescored = sorted(
            (
                ScoredChunk(chunk=c.chunk, score=float(s), rank=0)
                for c, s in zip(candidates, scores, strict=True)
            ),
            key=lambda r: (-r.score, r.chunk_id),
        )
        return renumber(rescored)[:top_k]
