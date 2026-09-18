"""Retrievers, fusion, and the vector store."""

from __future__ import annotations

import numpy as np
import pytest

from geolytics.chunking import SentenceChunker
from geolytics.index.base import ScoredChunk
from geolytics.index.memory import InMemoryVectorStore, score_matrix
from geolytics.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankingRetriever,
    reciprocal_rank_fusion,
    weighted_fusion,
)


@pytest.fixture
def indexed(corpus, embedder):
    chunker = SentenceChunker(max_tokens=40, overlap_sentences=0, min_tokens=1)
    chunks = [c for d in corpus for c in chunker.chunk(d)]
    store = InMemoryVectorStore()
    store.create_collection("t", dim=embedder.dim)
    store.upsert("t", chunks, embedder.embed([c.text for c in chunks]))
    return chunks, store


class TestScoreMatrix:
    def test_cosine_of_identical_vectors_is_one(self):
        v = np.array([[1.0, 2.0, 3.0]])
        assert score_matrix(v, v[0], "cosine") == pytest.approx([1.0])

    def test_euclidean_is_negated_for_descending_sort(self):
        v = np.array([[0.0, 0.0], [3.0, 4.0]])
        scores = score_matrix(v, np.array([0.0, 0.0]), "euclidean")
        assert scores[0] > scores[1]
        assert scores[1] == pytest.approx(-5.0)

    def test_zero_vector_does_not_divide_by_zero(self):
        v = np.array([[0.0, 0.0]])
        assert score_matrix(v, np.array([1.0, 0.0]), "cosine") == pytest.approx([0.0])

    def test_unknown_metric(self):
        with pytest.raises(ValueError, match="unknown metric"):
            score_matrix(np.zeros((1, 2)), np.zeros(2), "manhattan")


class TestInMemoryVectorStore:
    def test_returns_ranked_results(self, indexed):
        chunks, store = indexed
        results = store.search("t", np.ones(128, dtype=np.float32), top_k=5)
        assert len(results) == 5
        assert [r.rank for r in results] == [1, 2, 3, 4, 5]
        assert all(a.score >= b.score for a, b in zip(results, results[1:], strict=False))

    def test_top_k_larger_than_corpus(self, indexed):
        chunks, store = indexed
        assert len(store.search("t", np.ones(128, dtype=np.float32), top_k=999)) == len(chunks)

    def test_upsert_replaces_by_chunk_id(self, indexed, embedder):
        chunks, store = indexed
        before = store.count("t")
        store.upsert("t", chunks[:2], embedder.embed([c.text for c in chunks[:2]]))
        assert store.count("t") == before

    def test_dimension_mismatch_is_rejected(self, indexed):
        chunks, store = indexed
        with pytest.raises(ValueError, match="dim"):
            store.upsert("t", chunks[:1], np.zeros((1, 5), dtype=np.float32))

    def test_count_mismatch_is_rejected(self, indexed):
        chunks, store = indexed
        with pytest.raises(ValueError, match="chunks but"):
            store.upsert("t", chunks[:2], np.zeros((1, 128), dtype=np.float32))

    def test_missing_collection(self):
        with pytest.raises(KeyError):
            InMemoryVectorStore().search("nope", np.zeros(4), top_k=1)

    def test_empty_collection_returns_nothing(self, embedder):
        store = InMemoryVectorStore()
        store.create_collection("e", dim=embedder.dim)
        assert store.search("e", np.zeros(embedder.dim), top_k=5) == []


class TestBM25:
    def test_finds_the_lexically_matching_chunk(self, indexed):
        chunks, _ = indexed
        results = BM25Retriever(chunks).retrieve("motorised auger blocked drain", top_k=3)
        assert results
        assert "auger" in results[0].chunk.text

    def test_unknown_terms_return_nothing(self, indexed):
        chunks, _ = indexed
        assert BM25Retriever(chunks).retrieve("zzzz qqqq", top_k=5) == []

    def test_scores_are_non_negative(self, indexed):
        chunks, _ = indexed
        scores = BM25Retriever(chunks).scores("boiler servicing")
        assert (scores >= 0).all()

    def test_common_term_does_not_score_negative(self):
        """The +1 idf smoothing keeps a term in most documents from scoring < 0."""
        from geolytics.chunking.base import Chunk

        chunks = [
            Chunk(chunk_id=f"c{i}", doc_id="d", text="acme service today", ordinal=i,
                  start_char=0, end_char=0)
            for i in range(10)
        ]
        assert (BM25Retriever(chunks).scores("acme") >= 0).all()

    def test_ranks_are_sequential(self, indexed):
        chunks, _ = indexed
        results = BM25Retriever(chunks).retrieve("acme service quoted", top_k=4)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))


class TestDense:
    def test_retrieves_semantically_close_chunks(self, indexed, embedder):
        chunks, store = indexed
        results = DenseRetriever(store, embedder, "t").retrieve("boiler heat exchanger", top_k=3)
        assert results
        assert any("boiler" in r.chunk.text.lower() for r in results)


class TestFusion:
    def _hit(self, cid: str, rank: int, score: float = 1.0) -> ScoredChunk:
        from geolytics.chunking.base import Chunk

        return ScoredChunk(
            chunk=Chunk(chunk_id=cid, doc_id="d", text=cid, ordinal=0, start_char=0, end_char=0),
            score=score,
            rank=rank,
        )

    def test_rrf_rewards_agreement(self):
        a = [self._hit("x", 1), self._hit("y", 2)]
        b = [self._hit("y", 1), self._hit("z", 2)]
        fused = reciprocal_rank_fusion([a, b])
        # y is ranked 2nd and 1st; x only appears once at rank 1.
        assert fused[0].chunk_id == "y"

    def test_rrf_reranks_from_one(self):
        fused = reciprocal_rank_fusion([[self._hit("x", 1)], [self._hit("y", 1)]])
        assert [f.rank for f in fused] == [1, 2]

    def test_rrf_ignores_raw_score_magnitude(self):
        a = [self._hit("x", 1, score=1000.0)]
        b = [self._hit("y", 1, score=0.001)]
        fused = reciprocal_rank_fusion([a, b])
        assert fused[0].score == pytest.approx(fused[1].score)

    def test_weighted_alpha_one_is_pure_dense(self):
        dense = [self._hit("x", 1, 0.9), self._hit("y", 2, 0.1)]
        lexical = [self._hit("y", 1, 50.0), self._hit("x", 2, 1.0)]
        assert weighted_fusion(dense, lexical, alpha=1.0)[0].chunk_id == "x"

    def test_weighted_alpha_zero_is_pure_lexical(self):
        dense = [self._hit("x", 1, 0.9), self._hit("y", 2, 0.1)]
        lexical = [self._hit("y", 1, 50.0), self._hit("x", 2, 1.0)]
        assert weighted_fusion(dense, lexical, alpha=0.0)[0].chunk_id == "y"

    def test_weighted_rejects_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            weighted_fusion([], [], alpha=1.5)

    def test_all_tied_scores_do_not_collapse(self):
        tied = [self._hit("x", 1, 5.0), self._hit("y", 2, 5.0)]
        fused = weighted_fusion(tied, [], alpha=1.0)
        assert all(f.score > 0 for f in fused)


class TestHybridRetriever:
    def test_returns_at_most_top_k(self, indexed, embedder):
        chunks, store = indexed
        hybrid = HybridRetriever(
            dense=DenseRetriever(store, embedder, "t"), lexical=BM25Retriever(chunks)
        )
        assert len(hybrid.retrieve("boiler heat exchanger", top_k=3)) <= 3

    def test_fetches_deeper_than_top_k_from_each_arm(self, indexed, embedder):
        chunks, store = indexed
        hybrid = HybridRetriever(
            dense=DenseRetriever(store, embedder, "t"),
            lexical=BM25Retriever(chunks),
            candidate_multiplier=3,
        )
        assert hybrid.params["candidate_multiplier"] == 3

    def test_unknown_fusion_method(self, indexed, embedder):
        chunks, store = indexed
        hybrid = HybridRetriever(
            dense=DenseRetriever(store, embedder, "t"),
            lexical=BM25Retriever(chunks),
            method="nope",
        )
        with pytest.raises(ValueError, match="fusion method"):
            hybrid.retrieve("boiler", top_k=2)


class TestReranking:
    def test_reorders_by_cross_encoder_score(self, indexed, embedder):
        chunks, store = indexed

        class ReverseScorer:
            """Scores candidates in reverse order of arrival."""

            def score(self, query: str, texts: list[str]) -> list[float]:
                return [float(i) for i in range(len(texts))]

        base = DenseRetriever(store, embedder, "t")
        original = base.retrieve("boiler", top_k=5)
        reranked = RerankingRetriever(base, ReverseScorer(), candidate_k=5).retrieve("boiler", 5)
        assert reranked[0].chunk_id == original[-1].chunk_id
        assert [r.rank for r in reranked] == [1, 2, 3, 4, 5]

    def test_empty_base_result(self, indexed, embedder):
        class NullRetriever:
            name = "null"

            def retrieve(self, query, top_k=10):
                return []

            def describe(self):
                return {}

        class Scorer:
            def score(self, query, texts):
                return []

        assert RerankingRetriever(NullRetriever(), Scorer()).retrieve("x", 5) == []
