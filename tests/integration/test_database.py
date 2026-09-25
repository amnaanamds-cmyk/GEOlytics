"""Audit pipeline against real PostgreSQL and a real HTTP server."""

from __future__ import annotations

import pytest
from services import POSTGRES_DSN, create_org, requires_postgres
from sqlalchemy import select, text

from geolytics.db.models import Audit, Base, Crawl, Page, PageScore, Site
from geolytics.pipeline import mark_audit_failed, persist_audit, run_audit, site_documents

pytestmark = requires_postgres


@pytest.fixture
def db(crawl_settings, monkeypatch):
    """A clean schema in the test database for each test."""
    monkeypatch.setenv("GEOLYTICS_POSTGRES_DSN", POSTGRES_DSN)

    from geolytics.config import get_settings
    from geolytics.db.session import get_engine, get_session_factory

    for cached in (get_settings, get_engine, get_session_factory):
        cached.cache_clear()

    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    for cached in (get_settings, get_engine, get_session_factory):
        cached.cache_clear()


@pytest.fixture
def org_id(db):
    from geolytics.db.session import session_scope

    with session_scope() as session:
        return create_org(session).id


@pytest.fixture
def audit_row(db, org_id):
    from geolytics.db.session import session_scope

    with session_scope() as session:
        site = Site(
            org_id=org_id, url="https://acme.example/", host="acme.example", name="Acme"
        )
        session.add(site)
        session.flush()
        audit = Audit(
            org_id=org_id, site_id=site.id, status="pending", config={"max_pages": 5}
        )
        session.add(audit)
        session.flush()
        return audit.id


class TestSchema:
    def test_every_table_is_created(self, db):
        with db.connect() as conn:
            names = {
                r[0]
                for r in conn.execute(
                    text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
                )
            }
        assert {
            "sites", "crawls", "pages", "audits",
            "page_scores", "experiment_runs", "query_records",
        } <= names

    def test_json_columns_round_trip_on_postgres(self, db):
        from geolytics.db.models import ExperimentRun
        from geolytics.db.session import session_scope

        with session_scope() as session:
            org = create_org(session, slug="json-test")
            run = ExperimentRun(
                org_id=org.id,
                experiment="e",
                condition="c",
                chunker={"strategy": "sentence", "params": {"max_tokens": 256}},
                aggregate={"ndcg@10": 0.5},
                index_stats={"n_chunks": 10},
            )
            session.add(run)
            session.flush()
            run_id = run.id

        with session_scope() as session:
            loaded = session.get(ExperimentRun, run_id)
            assert loaded.chunker["params"]["max_tokens"] == 256
            assert loaded.aggregate["ndcg@10"] == 0.5

    def test_cascade_delete_removes_children(self, db, audit_row):
        from geolytics.db.session import session_scope

        with session_scope() as session:
            audit = session.get(Audit, audit_row)
            site_id = audit.site_id
            session.delete(session.get(Site, site_id))

        with session_scope() as session:
            assert session.get(Audit, audit_row) is None


class TestAuditPersistence:
    def test_full_audit_persists_and_reads_back(self, db, audit_row, crawl_settings, fixture_site):
        outcome = run_audit(
            f"{fixture_site}/index.html",
            max_pages=5,
            chunker="sentence",
            settings=crawl_settings,
        )
        assert outcome.documents, "crawl produced no documents"
        assert outcome.chunks, "chunking produced no chunks"
        assert outcome.scores

        persist_audit(audit_row, outcome)

        from geolytics.db.session import session_scope

        with session_scope() as session:
            audit = session.get(Audit, audit_row)
            assert audit.status == "complete"
            assert audit.overall_score is not None
            # Unfitted weights must be recorded as such, not presented as a result.
            assert audit.weights_fitted is False

            pages = session.scalars(
                select(Page).where(Page.crawl_id == audit.crawl_id)
            ).all()
            assert len(pages) == len(outcome.documents)
            assert all(p.text for p in pages)
            assert any(p.sections for p in pages)

            scores = session.scalars(
                select(PageScore).where(PageScore.audit_id == audit_row)
            ).all()
            assert len(scores) == len(outcome.scores)
            assert all(0.0 <= s.score <= 100.0 for s in scores)
            assert all(s.signals for s in scores)

    def test_crawl_row_records_the_user_agent(self, db, audit_row, crawl_settings, fixture_site):
        outcome = run_audit(f"{fixture_site}/index.html", max_pages=3, settings=crawl_settings)
        persist_audit(audit_row, outcome)

        from geolytics.db.session import session_scope

        with session_scope() as session:
            audit = session.get(Audit, audit_row)
            crawl = session.get(Crawl, audit.crawl_id)
            assert crawl.user_agent
            assert crawl.pages_fetched == len(outcome.documents)

    def test_documents_rehydrate_without_recrawling(
        self, db, audit_row, crawl_settings, fixture_site
    ):
        outcome = run_audit(f"{fixture_site}/index.html", max_pages=5, settings=crawl_settings)
        persist_audit(audit_row, outcome)

        from geolytics.db.session import session_scope

        with session_scope() as session:
            audit = session.get(Audit, audit_row)
            restored = site_documents(session, audit.crawl_id)

        assert len(restored) == len(outcome.documents)
        original = {d.doc_id: d for d in outcome.documents}
        for document in restored:
            assert document.text == original[document.doc_id].text
            assert document.sections == original[document.doc_id].sections

    def test_no_duplicate_documents_from_url_aliases(
        self, db, crawl_settings, fixture_site
    ):
        """'/' and '/index.html' serve the same page; only one may be indexed."""
        outcome = run_audit(f"{fixture_site}/index.html", max_pages=10, settings=crawl_settings)
        texts = [" ".join(d.text.split()) for d in outcome.documents]
        assert len(texts) == len(set(texts))

    def test_failure_is_recorded(self, db, audit_row):
        mark_audit_failed(audit_row, "ValueError: boom")

        from geolytics.db.session import session_scope

        with session_scope() as session:
            audit = session.get(Audit, audit_row)
            assert audit.status == "failed"
            assert "boom" in audit.error

    def test_persist_rejects_an_unknown_audit(self, db, crawl_settings, fixture_site):
        outcome = run_audit(f"{fixture_site}/index.html", max_pages=2, settings=crawl_settings)
        with pytest.raises(ValueError, match="does not exist"):
            persist_audit(999999, outcome)
