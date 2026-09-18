"""Embedding backends."""

from geolytics.embedding.base import Embedder, embed_queries
from geolytics.embedding.hashing import HashingEmbedder
from geolytics.embedding.registry import build_embedder

__all__ = ["Embedder", "HashingEmbedder", "build_embedder", "embed_queries"]
