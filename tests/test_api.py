"""HTTP API: request validation and the experiments endpoint.

Runs against SQLite rather than PostgreSQL so the suite needs no service. The
models use portable column types, so the schema round-trips; only Postgres-
specific behaviour would need a real database.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOLYTICS_POSTGRES_DSN", f"sqlite:///{tmp_path/'test.db'}")
    monkeypatch.setenv("GEOLYTICS_ENV", "test")

    from geolytics.config import get_settings
    from geolytics.db.session import get_engine, get_session_factory

    for cached in (get_settings, get_engine, get_session_factory):
        cached.cache_clear()

    from geolytics.db.models import Base

    Base.metadata.create_all(get_engine())

    from geolytics.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client

    for cached in (get_settings, get_engine, get_session_factory):
        cached.cache_clear()


@pytest.fixture
def seeded(client):
    """Two conditions scored on the same queries, one clearly better.

    The two conditions vary on *different* cycles so the paired differences are
    not constant -- identical variation would collapse the bootstrap interval to
    a point and make the comparison degenerate.
    """
    from geolytics.db.models import ExperimentRun, QueryRecord
    from geolytics.db.session import session_scope

    with session_scope() as session:
        for condition, base, cycle, step in (
            ("fixed+dense", 0.40, 7, 0.01),
            ("sentence+dense", 0.55, 5, 0.015),
        ):
            run = ExperimentRun(
                experiment="chunking",
                condition=condition,
                chunker={"strategy": condition.split("+")[0]},
                retriever={"retriever": "dense"},
                index_stats={"n_chunks": 100, "mean_tokens": 200.0},
                aggregate={"ndcg@10": base},
                n_queries=30,
            )
            session.add(run)
            session.flush()
            for i in range(30):
                session.add(
                    QueryRecord(
                        run_id=run.id,
                        query_id=f"q{i}",
                        query_text=f"question {i}",
                        metrics={"ndcg@10": base + (i % cycle) * step},
                    )
                )
    return client


class TestHealth:
    def test_reports_version_and_services(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "postgres" in body["services"] and "qdrant" in body["services"]

    def test_never_raises_on_unreachable_dependencies(self, client):
        assert client.get("/health").status_code == 200


class TestAuditValidation:
    def test_rejects_an_unknown_chunker(self, client):
        response = client.post("/audits", json={"url": "https://a.example", "chunker": "nope"})
        assert response.status_code == 422

    def test_rejects_an_unknown_retriever(self, client):
        response = client.post("/audits", json={"url": "https://a.example", "retriever": "nope"})
        assert response.status_code == 422

    def test_rejects_a_non_url(self, client):
        assert client.post("/audits", json={"url": "not-a-url"}).status_code == 422

    def test_rejects_an_out_of_range_page_limit(self, client):
        response = client.post("/audits", json={"url": "https://a.example", "max_pages": 9999})
        assert response.status_code == 422

    def test_missing_audit_returns_404(self, client):
        assert client.get("/audits/12345").status_code == 404


class TestExperiments:
    def test_returns_runs_and_corrected_comparisons(self, seeded):
        body = seeded.get("/experiments/chunking?metric=ndcg@10").json()
        assert len(body["runs"]) == 2
        assert len(body["comparisons"]) == 1

        comparison = body["comparisons"][0]
        assert comparison["n"] == 30
        assert comparison["p_adjusted"] is not None
        assert comparison["ci_low"] < comparison["mean_diff"] < comparison["ci_high"]
        assert comparison["effect_label"] in {"negligible", "small", "medium", "large"}

    def test_detects_the_seeded_difference(self, seeded):
        comparison = seeded.get("/experiments/chunking?metric=ndcg@10").json()["comparisons"][0]
        assert comparison["significant"]
        assert abs(comparison["mean_diff"]) == pytest.approx(0.15, abs=0.01)

    def test_methods_note_describes_the_test(self, seeded):
        note = seeded.get("/experiments/chunking").json()["methods_note"]
        assert "Wilcoxon" in note and "Holm" in note and "bootstrap" in note

    def test_baseline_selection(self, seeded):
        body = seeded.get("/experiments/chunking?baseline=fixed%2Bdense").json()
        assert all(c["name_b"] == "fixed+dense" for c in body["comparisons"])

    def test_unknown_baseline_is_a_400(self, seeded):
        assert seeded.get("/experiments/chunking?baseline=nope").status_code == 400

    def test_unknown_experiment_is_a_404(self, seeded):
        assert seeded.get("/experiments/missing").status_code == 404

    def test_unscored_metric_is_a_409(self, seeded):
        assert seeded.get("/experiments/chunking?metric=map@99").status_code == 409
