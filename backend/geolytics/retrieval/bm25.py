"""Okapi BM25 lexical retrieval.

Implemented here rather than pulled from a package so that the report can state
the exact scoring function and parameters, and so the sparse side of the hybrid
retriever has no external dependency.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from geolytics.chunking.base import Chunk
from geolytics.index.base import ScoredChunk
from geolytics.retrieval.base import Retriever

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A short, explicit stop list. Deliberately not the full NLTK list: aggressive
# stopping hurts BM25 on the short, entity-heavy queries this system generates.
_STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
        "was", "were", "with",
    ]
)


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


class BM25Retriever(Retriever):
    """Okapi BM25 with the standard k1/b parameterisation.

    idf uses the Robertson/Sparck-Jones form with the +1 smoothing that keeps
    the value positive for terms appearing in more than half the corpus --
    without it, a common term contributes a *negative* score and can push a
    genuinely relevant chunk down the ranking.
    """

    name = "bm25"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
        remove_stopwords: bool = True,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords
        self.chunks = list(chunks)

        self._docs = [tokenize(c.text, remove_stopwords) for c in self.chunks]
        self._lengths = np.array([len(d) for d in self._docs], dtype=np.float64)
        self._avg_len = float(self._lengths.mean()) if len(self._lengths) else 0.0

        self._tf: list[Counter[str]] = [Counter(d) for d in self._docs]
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(set(doc))
        n = len(self._docs)
        self._idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    @property
    def params(self) -> dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "remove_stopwords": self.remove_stopwords,
            "n_docs": len(self.chunks),
        }

    def scores(self, query: str) -> np.ndarray:
        terms = tokenize(query, self.remove_stopwords)
        out = np.zeros(len(self._docs), dtype=np.float64)
        if not terms or self._avg_len == 0:
            return out

        for term in terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            tf = np.array([counts.get(term, 0) for counts in self._tf], dtype=np.float64)
            denom = tf + self.k1 * (1.0 - self.b + self.b * self._lengths / self._avg_len)
            out += idf * (tf * (self.k1 + 1.0)) / np.where(denom == 0, 1.0, denom)
        return out

    def retrieve(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        scores = self.scores(query)
        if scores.size == 0:
            return []
        k = min(top_k, scores.size)
        candidates = np.argpartition(-scores, k - 1)[:k]
        order = candidates[np.argsort(-scores[candidates], kind="stable")]
        return [
            ScoredChunk(chunk=self.chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
            if scores[i] > 0.0
        ]
