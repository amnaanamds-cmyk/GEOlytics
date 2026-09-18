"""Sentence-Transformers embedding backend (the one used for real results)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from geolytics.embedding.base import Embedder

# Query/passage prefixes for asymmetric models. Using the wrong prefix -- or
# none -- measurably degrades retrieval, so the mapping is explicit rather
# than inferred.
_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/e5-base-v2": ("query: ", "passage: "),
    "intfloat/e5-large-v2": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-large-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
}


class SentenceTransformerEmbedder(Embedder):
    """Wraps a `sentence-transformers` model.

    `normalize` defaults to True, matching the default of most models on the
    MTEB leaderboard. Set it to False only with a model trained for dot-product
    retrieval on unnormalised vectors (the `multi-qa-*-dot-v1` family), where
    vector magnitude carries information.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "SentenceTransformerEmbedder requires the 'embed' extra: "
                "pip install -e '.[embed]'"
            ) from exc

        self.name = model_name
        self.model_name = model_name
        self._normalize = normalize
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self._query_prefix, self._passage_prefix = _PREFIXES.get(model_name, ("", ""))

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def normalized(self) -> bool:
        return self._normalize

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self._passage_prefix)

    def embed_query(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self._query_prefix)

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        prepared = [prefix + t for t in texts] if prefix else list(texts)
        vectors = self._model.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def describe(self) -> dict[str, Any]:
        return {
            "embedder": "sentence-transformers",
            "model": self.model_name,
            "dim": self.dim,
            "normalized": self._normalize,
        }
