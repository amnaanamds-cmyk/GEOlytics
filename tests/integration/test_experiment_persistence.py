"""Experiment runs written to PostgreSQL and read back through the API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from services import POSTGRES_DSN, requires_postgres

from geolytics.chunking import SentenceChunker, build_chunker
from geolytics.db.models import Base, ExperimentRun, QueryRecord
from geolytics.embedding.hashing import HashingEmbedder
from geolytics.evaluation.harness import Condition, ExperimentHarness
from geolytics.evaluation.heuristic_qagen import generate_heuristic_query_set
from geolytics.evaluation.qagen import QAGenerationConfig
from geolytics.pipeline import persist_runs
from geolytics.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

pytestmark = requires_postgres


def _dense(context):
    return DenseRetriever(context.store, context.embedder, context.collection)


def _hybrid(context):
    return HybridRetriever(dense=_dense(context), lexical=BM25Retriever(context.chunks))


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setenv("GEOLYTICS_POSTGRES_DSN", POSTGRES_DSN)
    monkeypatch.setenv("GEOLYTICS_ENV", "test")

    from geolytics.config import get_settings
    from geolytics.db.session import get_engine, get_session_factory

    caches = (get_settings, get_engine, get_session_factory)
    for cached in caches:
        cached.cache_clear()

    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    for cached in caches:
        cached.cache_clear()


@pytest.fixture
def runs(fixture_site):
    from geolytics.config import Settings
    from geolytics.crawl.crawler import Crawler

    settings = Settings(env="test", crawl_delay_seconds=0.0, crawl_max_pages=6)
    with Crawler(settings=settings, cache_dir=None) as crawler:
        documents = [r.document for r in crawler.crawl(f"{fixture_site}/index.html")]

    query_set, _ = generate_heuristic_query_set(
        documents, QAGenerationConfig(questions_per_unit=2, min_unit_tokens=10)
    )
    embedder = HashingEmbedder(dim=128)
    harness = ExperimentHarness(documents, query_set, embedder, top_k=10, cutoffs=(1, 5, 10))
    conditions = [
        Condition("fixed+dense", build_chunker("fixed", chunk_tokens=48), _dense),
        Condition("sentence+hybrid", SentenceChunker(max_tokens=48, min_tokens=1), _hybrid),
    ]
    return harness.run_all(conditions), query_set


class TestPersistRuns:
    def test_writes_a_row_per_condition(self, db, runs):
        run_list, query_set = runs
        ids = persist_runs("grid", run_list, query_set)
        assert len(ids) == len(run_list)

        from geolytics.db.session import session_scope

        with session_scope() as session:
            rows = session.query(ExperimentRun).filter_by(experiment="grid").all()
            assert {r.condition for r in rows} == {r.condition for r in run_list}
            assert all(r.aggregate for r in rows)
            assert all(r.index_stats["n_chunks"] > 0 for r in rows)

    def test_writes_one_query_record_per_scored_query(self, db, runs):
        run_list, query_set = runs
        persist_runs("grid", run_list, query_set)

        from geolytics.db.session import session_scope

        with session_scope() as session:
            for run in run_list:
                row = (
                    session.query(ExperimentRun)
                    .filter_by(experiment="grid", condition=run.condition)
                    .one()
                )
                records = session.query(QueryRecord).filter_by(run_id=row.id).all()
                assert len(records) == run.n_queries_scored
                assert all(r.metrics for r in records)
                assert all(r.query_text for r in records)
                assert all(r.provenance == "heuristic" for r in records)

    def test_per_query_scores_round_trip_exactly(self, db, runs):
        """The statistics are recomputed from these rows, so they must be exact."""
        run_list, query_set = runs
        persist_runs("grid", run_list, query_set)

        from geolytics.db.session import session_scope

        run = run_list[0]
        with session_scope() as session:
            row = (
                session.query(ExperimentRun)
                .filter_by(experiment="grid", condition=run.condition)
                .one()
            )
            stored = {
                r.query_id: r.metrics
                for r in session.query(QueryRecord).filter_by(run_id=row.id).all()
            }

        for query_id, metrics in run.per_query.items():
            assert stored[query_id] == pytest.approx(metrics)

    def test_ranked_chunk_ids_are_kept_for_pooling(self, db, runs):
        run_list, query_set = runs
        persist_runs("grid", run_list, query_set)

        from geolytics.db.session import session_scope

        with session_scope() as session:
            records = session.query(QueryRecord).all()
            assert any(r.ranked_chunk_ids for r in records)


class TestApiReadsPersistedRuns:
    @pytest.fixture
    def client(self, db, runs):
        run_list, query_set = runs
        persist_runs("grid", run_list, query_set)

        from geolytics.api.app import create_app

        with TestClient(create_app()) as c:
            yield c

    def test_returns_conditions_and_comparisons(self, client):
        body = client.get("/experiments/grid?metric=ndcg@10").json()
        assert len(body["runs"]) == 2
        assert len(body["comparisons"]) == 1
        assert body["comparisons"][0]["p_adjusted"] is not None

    def test_comparison_is_computed_from_stored_per_query_rows(self, client, runs):
        run_list, _ = runs
        body = client.get("/experiments/grid?metric=ndcg@10").json()
        comparison = body["comparisons"][0]
        assert comparison["n"] == run_list[0].n_queries_scored

        by_name = {r.condition: r for r in run_list}
        assert comparison["mean_a"] == pytest.approx(
            by_name[comparison["name_a"]].mean("ndcg@10"), abs=1e-9
        )

    def test_index_stats_travel_with_the_run(self, client):
        body = client.get("/experiments/grid").json()
        assert all(r["index_stats"]["n_chunks"] > 0 for r in body["runs"])
        assert all("mean_tokens" in r["index_stats"] for r in body["runs"])
