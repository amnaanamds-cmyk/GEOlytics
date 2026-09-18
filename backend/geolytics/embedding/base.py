"""The embedder interface.

`normalized` is not cosmetic -- it determines whether the similarity-metric
comparison in `evaluation.rank_equivalence` is a real experimental factor or a
mathematical identity. See that module for the argument.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np


class Embedder(ABC):
    """Turns text into dense vectors."""

    name: str = "base"

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimensionality."""

    @property
    def normalized(self) -> bool:
        """Whether `embed` returns unit-length vectors.

        When True, cosine similarity, dot product and (negative) Euclidean
        distance induce *identical* rankings, so those three cannot be
        compared as retrieval variants. Assert this rather than assume it.
        """
        return True

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents. Returns shape (len(texts), dim)."""

    def embed_query(self, texts: Sequence[str]) -> np.ndarray:
        """Embed queries.

        Separate from `embed` because asymmetric models (E5, BGE, GTR) expect
        a different prefix on queries than on passages. Getting this wrong is
        a silent ~10-point nDCG loss, so the split is mandatory in the
        interface even when a backend treats both the same.
        """
        return self.embed(texts)

    def describe(self) -> dict[str, Any]:
        return {"embedder": self.name, "dim": self.dim, "normalized": self.normalized}


def embed_queries(embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
    return embedder.embed_query(texts)
