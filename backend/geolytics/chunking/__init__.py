"""Chunking strategies -- experimental factor #1.

All strategies implement `ChunkingStrategy` so the evaluation harness can swap
them without touching any other layer. Each strategy reports `params`, which is
recorded with every run so a result table can be traced back to an exact
configuration.
"""

from geolytics.chunking.base import Chunk, ChunkingStrategy, Document, Section
from geolytics.chunking.fixed import FixedSizeChunker
from geolytics.chunking.parent_child import ParentChildChunker
from geolytics.chunking.registry import CHUNKERS, build_chunker
from geolytics.chunking.semantic import SemanticChunker
from geolytics.chunking.sentence import SentenceChunker

__all__ = [
    "CHUNKERS",
    "Chunk",
    "ChunkingStrategy",
    "Document",
    "FixedSizeChunker",
    "ParentChildChunker",
    "Section",
    "SemanticChunker",
    "SentenceChunker",
    "build_chunker",
]
