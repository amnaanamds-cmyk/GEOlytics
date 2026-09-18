"""Name -> chunker construction, so experiment grids can be declared as data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geolytics.chunking.base import ChunkingStrategy
from geolytics.chunking.fixed import FixedSizeChunker
from geolytics.chunking.parent_child import ParentChildChunker
from geolytics.chunking.semantic import SemanticChunker
from geolytics.chunking.sentence import SentenceChunker

if TYPE_CHECKING:  # pragma: no cover
    from geolytics.embedding.base import Embedder

CHUNKERS = ("fixed", "sentence", "semantic", "parent_child")


def build_chunker(
    name: str,
    *,
    embedder: Embedder | None = None,
    **params: Any,
) -> ChunkingStrategy:
    """Construct a chunker by name.

    `parent_child` takes nested specs, e.g.::

        build_chunker(
            "parent_child",
            parent={"name": "sentence", "max_tokens": 1024},
            child={"name": "sentence", "max_tokens": 128},
        )
    """
    if name == "fixed":
        return FixedSizeChunker(**params)
    if name == "sentence":
        return SentenceChunker(**params)
    if name == "semantic":
        if embedder is None:
            raise ValueError("the 'semantic' chunker requires an embedder")
        return SemanticChunker(embedder=embedder, **params)
    if name == "parent_child":
        parent_spec = dict(params.pop("parent", {"name": "sentence", "max_tokens": 1024}))
        child_spec = dict(params.pop("child", {"name": "sentence", "max_tokens": 128}))
        if params:
            raise TypeError(f"unexpected parent_child params: {sorted(params)}")
        return ParentChildChunker(
            parent=build_chunker(parent_spec.pop("name"), embedder=embedder, **parent_spec),
            child=build_chunker(child_spec.pop("name"), embedder=embedder, **child_spec),
        )
    raise ValueError(f"unknown chunker {name!r}; expected one of {CHUNKERS}")
