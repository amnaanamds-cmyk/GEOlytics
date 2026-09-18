"""A deterministic, dependency-free embedder.

Purpose: the test suite and CI must exercise chunking, indexing, retrieval and
the evaluation harness without downloading a transformer or requiring a GPU.
This backend hashes character n-grams into a fixed-width vector -- essentially
a signed hashing-trick bag-of-ngrams.

It is a *test fixture, not a research baseline*. It carries no semantics
beyond lexical overlap, so never report experimental numbers produced with it.
`build_embedder` refuses it outside test/development settings for that reason.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from geolytics.embedding.base import Embedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder(Embedder):
    name = "hashing"

    def __init__(self, dim: int = 384, ngram: int = 3, seed: int = 0) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self.ngram = ngram
        self.seed = seed

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def normalized(self) -> bool:
        return True

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(
                    feature.encode("utf-8"), digest_size=8, key=str(self.seed).encode()
                ).digest()
                value = int.from_bytes(digest, "big")
                out[row, value % self._dim] += 1.0 if (value >> 63) & 1 else -1.0
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        features = list(tokens)
        for n in range(2, self.ngram + 1):
            features.extend(
                " ".join(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))
            )
        return features
