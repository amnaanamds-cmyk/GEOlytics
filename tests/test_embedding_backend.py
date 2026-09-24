"""The Sentence-Transformers wrapper.

The model itself is stubbed. That is not a shortcut around testing: what this
wrapper is responsible for is the query/passage prefix split, the normalisation
flag, the dtype and the reported dimension -- and getting any of those wrong is
a silent quality loss, not a crash. Downloading a real transformer would test
Sentence-Transformers, not this code.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

sentence_transformers = pytest.importorskip(
    "sentence_transformers", reason="install the 'embed' extra"
)


class StubModel:
    """Records what it was asked to encode and returns deterministic vectors."""

    def __init__(self, dim: int | None = 8) -> None:
        self.dim = dim
        self.encoded: list[list[str]] = []
        self.kwargs: list[dict] = []

    def get_sentence_embedding_dimension(self) -> int | None:
        return self.dim

    def encode(self, texts, **kwargs):
        self.encoded.append(list(texts))
        self.kwargs.append(kwargs)
        rows = []
        for text in texts:
            # Seeded on the text so different input gives a different
            # *direction*, not just a different magnitude -- a magnitude-only
            # stub would survive normalisation and hide a dropped prefix.
            seed = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), "big")
            vector = np.random.default_rng(seed).normal(size=self.dim)
            if kwargs.get("normalize_embeddings"):
                vector = vector / np.linalg.norm(vector)
            rows.append(vector)
        return np.asarray(rows)


@pytest.fixture
def make_embedder(monkeypatch):
    """Build the wrapper around a StubModel instead of a downloaded model."""

    def factory(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        stub = StubModel(dim=kwargs.pop("stub_dim", 8))
        monkeypatch.setattr(
            sentence_transformers, "SentenceTransformer", lambda *a, **k: stub
        )
        from geolytics.embedding.sentence_transformers_backend import (
            SentenceTransformerEmbedder,
        )

        return SentenceTransformerEmbedder(model_name=model_name, **kwargs), stub

    return factory


class TestDimensions:
    def test_reports_the_model_dimension(self, make_embedder):
        embedder, _ = make_embedder(stub_dim=384)
        assert embedder.dim == 384

    def test_a_model_without_a_fixed_width_is_rejected(self, make_embedder):
        embedder, _ = make_embedder(stub_dim=None)
        with pytest.raises(RuntimeError, match="does not report a sentence embedding"):
            _ = embedder.dim


class TestNormalisation:
    def test_normalises_by_default(self, make_embedder):
        embedder, stub = make_embedder()
        vectors = embedder.embed(["one", "two"])
        assert embedder.normalized
        assert stub.kwargs[0]["normalize_embeddings"] is True
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_can_be_disabled_for_dot_trained_models(self, make_embedder):
        embedder, stub = make_embedder(
            model_name="sentence-transformers/multi-qa-distilbert-dot-v1", normalize=False
        )
        embedder.embed(["one"])
        assert not embedder.normalized
        assert stub.kwargs[0]["normalize_embeddings"] is False

    def test_returns_float32(self, make_embedder):
        embedder, _ = make_embedder()
        assert embedder.embed(["one"]).dtype == np.float32


class TestAsymmetricPrefixes:
    def test_e5_gets_query_and_passage_prefixes(self, make_embedder):
        embedder, stub = make_embedder(model_name="intfloat/e5-base-v2")
        embedder.embed(["a passage"])
        embedder.embed_query(["a question"])
        assert stub.encoded[0] == ["passage: a passage"]
        assert stub.encoded[1] == ["query: a question"]

    def test_bge_prefixes_only_the_query(self, make_embedder):
        embedder, stub = make_embedder(model_name="BAAI/bge-base-en-v1.5")
        embedder.embed(["a passage"])
        embedder.embed_query(["a question"])
        assert stub.encoded[0] == ["a passage"]
        assert stub.encoded[1][0].startswith("Represent this sentence")

    def test_symmetric_models_get_no_prefix(self, make_embedder):
        embedder, stub = make_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embedder.embed(["a passage"])
        embedder.embed_query(["a question"])
        assert stub.encoded == [["a passage"], ["a question"]]

    def test_prefixing_changes_the_vector(self, make_embedder):
        """A prefix that never reached the model would be a silent no-op."""
        embedder, _ = make_embedder(model_name="intfloat/e5-base-v2")
        assert not np.allclose(embedder.embed(["x"]), embedder.embed_query(["x"]))


class TestBatching:
    def test_passes_the_batch_size_through(self, make_embedder):
        embedder, stub = make_embedder(batch_size=7)
        embedder.embed(["a", "b"])
        assert stub.kwargs[0]["batch_size"] == 7

    def test_never_shows_a_progress_bar(self, make_embedder):
        embedder, stub = make_embedder()
        embedder.embed(["a"])
        assert stub.kwargs[0]["show_progress_bar"] is False


class TestDescribe:
    def test_records_the_model_and_normalisation(self, make_embedder):
        embedder, _ = make_embedder(model_name="intfloat/e5-base-v2", stub_dim=768)
        described = embedder.describe()
        assert described["model"] == "intfloat/e5-base-v2"
        assert described["dim"] == 768
        assert described["normalized"] is True
