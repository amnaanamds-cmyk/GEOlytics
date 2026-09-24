"""The Qdrant-backed vector store against a running Qdrant server."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from services import QDRANT_URL, requires_qdrant

from geolytics.chunking import SentenceChunker
from geolytics.embedding.hashing import HashingEmbedder
from geolytics.index.memory import InMemoryVectorStore
from geolytics.index.qdrant import QdrantVectorStore
from geolytics.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

pytestmark = requires_qdrant


@pytest.fixture(scope="module")
def store():
    s = QdrantVectorStore(url=QDRANT_URL)
    yield s
    s.close()


@pytest.fixture
def collection(store, request):
    name = f"test_{request.node.name[:40]}"
    yield name
    store.drop_collection(name)


@pytest.fixture
def corpus_chunks(corpus_documents):
    chunker = SentenceChunker(max_tokens=40, overlap_sentences=0, min_tokens=1)
    return [c for d in corpus_documents for c in chunker.chunk(d)]


@pytest.fixture
def corpus_documents(fixture_site):
    from geolytics.config import Settings
    from geolytics.crawl.crawler import Crawler

    settings = Settings(env="test", crawl_delay_seconds=0.0, crawl_max_pages=10)
    with Crawler(settings=settings, cache_dir=None) as crawler:
        return [r.document for r in crawler.crawl(f"{fixture_site}/index.html")]


class TestRoundTrip:
    def test_upsert_and_search(self, store, collection, corpus_chunks):
        embedder = HashingEmbedder(dim=128)
        store.create_collection(collection, dim=128)
        store.upsert(collection, corpus_chunks, embedder.embed([c.text for c in corpus_chunks]))

        assert store.count(collection) == len(corpus_chunks)
        hits = store.search(collection, embedder.embed_query(["emergency callout"])[0], top_k=5)
        assert len(hits) == 5
        assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
        assert all(a.score >= b.score for a, b in zip(hits, hits[1:], strict=False))

    def test_payload_survives_the_round_trip(self, store, collection, corpus_chunks):
        embedder = HashingEmbedder(dim=128)
        store.create_collection(collection, dim=128)
        store.upsert(collection, corpus_chunks, embedder.embed([c.text for c in corpus_chunks]))

        original = {c.chunk_id: c for c in corpus_chunks}
        for hit in store.search(collection, np.ones(128, dtype=np.float32), top_k=5):
            source = original[hit.chunk_id]
            assert hit.chunk.text == source.text
            assert hit.chunk.doc_id == source.doc_id
            assert hit.chunk.heading_path == source.heading_path
            assert hit.chunk.ordinal == source.ordinal

    def test_self_match_ranks_first(self, store, collection, corpus_chunks):
        embedder = HashingEmbedder(dim=128)
        vectors = embedder.embed([c.text for c in corpus_chunks])
        store.create_collection(collection, dim=128)
        store.upsert(collection, corpus_chunks, vectors)

        hits = store.search(collection, vectors[2], top_k=1)
        assert hits[0].chunk_id == corpus_chunks[2].chunk_id

    def test_create_collection_clears_previous_points(self, store, collection, corpus_chunks):
        embedder = HashingEmbedder(dim=128)
        store.create_collection(collection, dim=128)
        store.upsert(collection, corpus_chunks, embedder.embed([c.text for c in corpus_chunks]))
        assert store.count(collection) > 0

        store.create_collection(collection, dim=128)
        assert store.count(collection) == 0

    def test_drop_is_idempotent(self, store, collection):
        store.create_collection(collection, dim=8)
        store.drop_collection(collection)
        store.drop_collection(collection)

    def test_per_query_metric_override_is_refused(self, store, collection, corpus_chunks):
        embedder = HashingEmbedder(dim=128)
        store.create_collection(collection, dim=128, metric="cosine")
        store.upsert(collection, corpus_chunks, embedder.embed([c.text for c in corpus_chunks]))
        with pytest.raises(ValueError, match="per collection"):
            store.search(collection, np.ones(128, dtype=np.float32), top_k=3, metric="dot")


def assert_same_ranking(remote, local, tol: float = 1e-5) -> None:
    """Assert two rankings agree, treating equal scores as unordered.

    Scores are compared elementwise -- that is the substantive claim, since it
    says both backends computed the same similarities. Ids are compared only
    within groups of equal score: chunks that tie carry no defined order, and
    each backend breaks ties by its own internal id, so requiring identical
    positions there would fail on a difference that does not exist.
    """
    assert len(remote) == len(local)
    for a, b in zip(remote, local, strict=True):
        assert a.score == pytest.approx(b.score, abs=tol), (
            f"score diverged at rank {a.rank}: {a.score} vs {b.score}"
        )

    def groups(hits):
        """Ids grouped into runs of equal score, in rank order.

        Only membership is compared. The scores themselves were already
        checked elementwise above, and comparing them again here would fail on
        last-bit differences between the two backends.
        """
        out, current, last = [], [], None
        for h in hits:
            if last is not None and abs(h.score - last) > tol:
                out.append(frozenset(current))
                current = []
            current.append(h.chunk_id)
            last = h.score
        if current:
            out.append(frozenset(current))
        return out

    assert groups(remote) == groups(local), "membership of an equal-score group diverged"


class TestAgreementWithExactSearch:
    def test_qdrant_matches_exact_search_on_a_small_corpus(
        self, store, collection, corpus_chunks
    ):
        """Qdrant falls back to exact search below its indexing threshold.

        The experiments run against the NumPy store; the application runs
        against Qdrant. If the two disagreed at this size, every experimental
        result would describe a different system from the deployed one.
        """
        embedder = HashingEmbedder(dim=128)
        vectors = embedder.embed([c.text for c in corpus_chunks])

        store.create_collection(collection, dim=128, metric="cosine")
        store.upsert(collection, corpus_chunks, vectors)

        memory = InMemoryVectorStore()
        memory.create_collection(collection, dim=128, metric="cosine")
        memory.upsert(collection, corpus_chunks, vectors)

        # Retrieve the whole corpus rather than a top-k slice. When more chunks
        # tie than fit in the cut, which tied chunks come back is undefined, so
        # a truncated comparison would fail on a difference that is not real.
        depth = len(corpus_chunks)
        for query in ("emergency callout", "bathroom installation", "how much does it cost"):
            q = embedder.embed_query([query])[0]
            assert_same_ranking(
                store.search(collection, q, top_k=depth),
                memory.search(collection, q, top_k=depth),
            )

    def test_distinct_scores_appear_in_identical_order(
        self, store, collection, corpus_chunks
    ):
        """The part of the ranking that is actually ordered must match exactly."""
        embedder = HashingEmbedder(dim=128)
        vectors = embedder.embed([c.text for c in corpus_chunks])

        store.create_collection(collection, dim=128, metric="cosine")
        store.upsert(collection, corpus_chunks, vectors)
        memory = InMemoryVectorStore()
        memory.create_collection(collection, dim=128, metric="cosine")
        memory.upsert(collection, corpus_chunks, vectors)

        q = embedder.embed_query(["emergency callout"])[0]
        remote = store.search(collection, q, top_k=8)
        local = memory.search(collection, q, top_k=8)

        def untied(hits):
            counts = Counter(round(h.score, 6) for h in hits)
            return [h.chunk_id for h in hits if counts[round(h.score, 6)] == 1]

        assert untied(remote) == untied(local)
        assert untied(remote), "fixture produced no distinct scores to compare"


class TestRetrieversOverQdrant:
    def test_dense_retriever_works_against_qdrant(self, store, collection, corpus_chunks):
        embedder = HashingEmbedder(dim=128)
        store.create_collection(collection, dim=128)
        store.upsert(collection, corpus_chunks, embedder.embed([c.text for c in corpus_chunks]))

        hits = DenseRetriever(store, embedder, collection).retrieve("callout price", top_k=3)
        assert len(hits) == 3

    def test_hybrid_retriever_works_against_qdrant(self, store, collection, corpus_chunks):
        embedder = HashingEmbedder(dim=128)
        store.create_collection(collection, dim=128)
        store.upsert(collection, corpus_chunks, embedder.embed([c.text for c in corpus_chunks]))

        hybrid = HybridRetriever(
            dense=DenseRetriever(store, embedder, collection),
            lexical=BM25Retriever(corpus_chunks),
        )
        hits = hybrid.retrieve("emergency callout within 90 minutes", top_k=5)
        assert hits
        assert any("90 minutes" in h.chunk.text for h in hits)
