"""Embedder construction from settings."""

from __future__ import annotations

from typing import Any

from geolytics.config import Settings, get_settings
from geolytics.embedding.base import Embedder
from geolytics.embedding.hashing import HashingEmbedder


def build_embedder(settings: Settings | None = None, **overrides: Any) -> Embedder:
    settings = settings or get_settings()
    backend = overrides.pop("backend", settings.embedding_backend)

    if backend == "hashing":
        if settings.env == "production":
            raise RuntimeError(
                "the hashing embedder is a test fixture and carries no semantics; "
                "it must not be used for reported results"
            )
        return HashingEmbedder(dim=overrides.pop("dim", settings.embedding_dim), **overrides)

    if backend == "sentence-transformers":
        from geolytics.embedding.sentence_transformers_backend import (
            SentenceTransformerEmbedder,
        )

        return SentenceTransformerEmbedder(
            model_name=overrides.pop("model_name", settings.embedding_model), **overrides
        )

    raise ValueError(f"unknown embedding backend {backend!r}")
