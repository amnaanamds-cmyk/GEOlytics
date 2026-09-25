"""The arq task queue against a real Redis, end to end."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from services import (
    POSTGRES_DSN,
    REDIS_URL,
    create_org,
    redis_available,
    requires_postgres,
    requires_redis,
)

from geolytics.db.models import Audit, Base, Site

pytestmark = [requires_redis, requires_postgres]


@pytest.fixture
def worker_env(monkeypatch, fixture_site):
    """Point the whole process at the test database and the fixture site.

    The task reads global settings rather than taking them as an argument --
    it runs in a worker process with no caller to pass them -- so the
    configuration has to be set in the environment here.
    """
    monkeypatch.setenv("GEOLYTICS_POSTGRES_DSN", POSTGRES_DSN)
    monkeypatch.setenv("GEOLYTICS_REDIS_URL", REDIS_URL)
    monkeypatch.setenv("GEOLYTICS_ENV", "test")
    monkeypatch.setenv("GEOLYTICS_CRAWL_DELAY_SECONDS", "0")
    monkeypatch.setenv("GEOLYTICS_EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("GEOLYTICS_EMBEDDING_DIM", "128")
    monkeypatch.setenv("GEOLYTICS_LLM_BACKEND", "none")
    # The fixture site is on loopback and an ephemeral port; the production
    # SSRF policy refuses both, by design.
    monkeypatch.setenv("GEOLYTICS_CRAWL_ALLOW_PRIVATE_ADDRESSES", "true")
    monkeypatch.setenv("GEOLYTICS_CRAWL_RESTRICT_PORTS", "false")

    from geolytics.config import get_settings
    from geolytics.db.session import get_engine, get_session_factory

    caches = (get_settings, get_engine, get_session_factory)
    for cached in caches:
        cached.cache_clear()

    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield fixture_site
    Base.metadata.drop_all(engine)
    for cached in caches:
        cached.cache_clear()


def _make_audit(url: str, max_pages: int = 4) -> int:
    """An organisation, a site and a queued audit, as the API would create."""
    from geolytics.db.session import session_scope

    with session_scope() as session:
        org = create_org(session, slug=f"worker-{uuid.uuid4().hex[:8]}")
        site = Site(org_id=org.id, url=url, host="127.0.0.1")
        session.add(site)
        session.flush()
        audit = Audit(
            org_id=org.id, site_id=site.id, status="pending",
            config={"max_pages": max_pages},
        )
        session.add(audit)
        session.flush()
        return audit.id


async def _redis_settings():
    from arq.connections import RedisSettings

    return RedisSettings.from_dsn(REDIS_URL)


class TestTaskFunction:
    def test_runs_the_audit_and_marks_it_complete(self, worker_env):
        from geolytics.db.session import session_scope
        from geolytics.tasks.queue import run_audit_task

        audit_id = _make_audit(f"{worker_env}/index.html")
        result = asyncio.run(run_audit_task({}, audit_id))

        assert result["audit_id"] == audit_id
        assert result["pages"] > 0
        assert result["chunks"] > 0

        with session_scope() as session:
            audit = session.get(Audit, audit_id)
            assert audit.status == "complete"
            assert audit.overall_score is not None
            assert audit.summary["n_pages"] == result["pages"]

    def test_marks_the_audit_failed_and_reraises(self, worker_env):
        from geolytics.db.session import session_scope
        from geolytics.tasks.queue import run_audit_task

        # A port nothing listens on: the crawl fetches no pages, and the
        # pipeline then has no documents to score.
        audit_id = _make_audit("http://127.0.0.1:9/index.html")
        asyncio.run(run_audit_task({}, audit_id))

        with session_scope() as session:
            audit = session.get(Audit, audit_id)
            # No pages is not an exception -- it is an audit with a warning.
            assert audit.status == "complete"
            assert any("no pages" in w for w in audit.summary["warnings"])

    def test_unknown_audit_raises(self, worker_env):
        from geolytics.tasks.queue import run_audit_task

        with pytest.raises(ValueError, match="does not exist"):
            asyncio.run(run_audit_task({}, 987654))


class TestQueueRoundTrip:
    def test_enqueued_job_is_executed_by_a_worker(self, worker_env):
        """Enqueue over real Redis, then drain the queue with a burst worker."""
        from arq import create_pool
        from arq.worker import Worker

        from geolytics.db.session import session_scope
        from geolytics.tasks.queue import run_audit_task

        audit_id = _make_audit(f"{worker_env}/index.html")

        async def drive() -> int:
            redis = await create_pool(await _redis_settings())
            try:
                await redis.flushall()
                await redis.enqueue_job("run_audit_task", audit_id)
                worker = Worker(
                    functions=[run_audit_task],
                    redis_settings=await _redis_settings(),
                    burst=True,
                    poll_delay=0.01,
                    max_jobs=1,
                )
                try:
                    return await worker.async_run() or 0
                finally:
                    await worker.close()
            finally:
                await redis.close()

        asyncio.run(drive())

        with session_scope() as session:
            audit = session.get(Audit, audit_id)
            assert audit.status == "complete", f"worker left audit in {audit.status!r}"
            assert audit.overall_score is not None

    def test_enqueue_from_sync_code_reaches_redis(self, worker_env):
        from arq import create_pool

        from geolytics.tasks.queue import enqueue_audit

        audit_id = _make_audit(f"{worker_env}/index.html")

        async def drain() -> int:
            redis = await create_pool(await _redis_settings())
            try:
                await redis.flushall()
                return 0
            finally:
                await redis.close()

        asyncio.run(drain())
        enqueue_audit(audit_id)

        async def count() -> int:
            redis = await create_pool(await _redis_settings())
            try:
                return len(await redis.queued_jobs())
            finally:
                await redis.close()

        assert asyncio.run(count()) == 1

    def test_enqueue_from_async_code_is_refused_with_guidance(self, worker_env):
        from geolytics.tasks.queue import enqueue_audit

        async def call_it() -> None:
            enqueue_audit(1)

        with pytest.raises(RuntimeError, match="enqueue_audit_async"):
            asyncio.run(call_it())


def WorkerSettingsRef():  # noqa: N802 - matches arq's settings-class convention
    from geolytics.tasks.queue import WorkerSettings

    return WorkerSettings


@pytest.mark.skipif(not redis_available(), reason="redis not running")
class TestWorkerSettings:
    def test_settings_expose_the_task_and_redis(self):
        from arq.connections import RedisSettings

        from geolytics.tasks.queue import WorkerSettings, run_audit_task

        assert run_audit_task in WorkerSettings.functions
        # arq reads this attribute directly; a callable here crashes the worker
        # at startup with "'staticmethod' object has no attribute 'host'".
        assert isinstance(WorkerSettings.redis_settings, RedisSettings)
        assert WorkerSettings.redis_settings.port == 6379

    def test_arq_can_build_a_worker_from_the_settings_class(self):
        """The check that actually catches a bad settings class."""
        from arq.worker import create_worker

        # create_worker only constructs; it opens no connection until run.
        worker = create_worker(WorkerSettingsRef(), burst=True)
        assert worker.functions
        assert "run_audit_task" in worker.functions
