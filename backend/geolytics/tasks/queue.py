"""arq task queue.

arq rather than Celery: it is asyncio-native (so it shares FastAPI's model),
its configuration is a single settings class, and it uses the same Redis. For
a crawl-and-score workload -- a handful of long jobs, not a high-throughput
stream -- Celery's extra surface buys nothing. Swap in Celery if the project
specifically needs to demonstrate it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from arq.connections import RedisSettings

from geolytics.config import get_settings

logger = logging.getLogger(__name__)


async def run_audit_task(ctx: dict[str, Any], audit_id: int) -> dict[str, Any]:
    """Crawl, score and persist one audit.

    The pipeline is synchronous and I/O-bound (network, then CPU for
    embeddings), so it runs in a thread rather than blocking the event loop and
    stalling every other job on the worker.
    """
    from geolytics.db.models import Audit
    from geolytics.db.session import session_scope
    from geolytics.pipeline import mark_audit_failed, persist_audit, run_audit

    with session_scope() as session:
        audit = session.get(Audit, audit_id)
        if audit is None:
            raise ValueError(f"audit {audit_id} does not exist")
        config = dict(audit.config or {})
        site = audit.site
        url = site.url
        audit.status = "running"

    try:
        outcome = await asyncio.to_thread(
            run_audit,
            url,
            # Already capped to the plan's per-audit ceiling when the audit was
            # accepted, so the worker never has to know about plans.
            max_pages=int(config.get("max_pages", 25)),
            chunker=str(config.get("chunker", "sentence")),
        )
        await asyncio.to_thread(persist_audit, audit_id, outcome)
    except Exception as exc:
        logger.exception("audit %s failed", audit_id)
        await asyncio.to_thread(mark_audit_failed, audit_id, f"{type(exc).__name__}: {exc}")
        raise

    return {
        "audit_id": audit_id,
        "pages": len(outcome.documents),
        "chunks": len(outcome.chunks),
        "overall_score": outcome.overall_score,
    }


def enqueue_audit(audit_id: int) -> None:
    """Enqueue from synchronous request-handling code.

    FastAPI runs a plain `def` route in a worker thread, where no event loop is
    running and `asyncio.run` is correct. Calling this from inside a coroutine
    would raise a confusing "cannot be called from a running event loop", so
    that case is detected and pointed at the async version instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_enqueue(audit_id))
        return
    raise RuntimeError(
        "enqueue_audit() is for synchronous callers; "
        "await enqueue_audit_async() from inside a coroutine"
    )


async def enqueue_audit_async(audit_id: int) -> None:
    """Enqueue from async code."""
    await _enqueue(audit_id)


async def _enqueue(audit_id: int) -> None:
    from arq import create_pool

    redis = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    try:
        await redis.enqueue_job("run_audit_task", audit_id)
    finally:
        await redis.close()


class WorkerSettings:
    """Worker configuration for `arq geolytics.tasks.queue.WorkerSettings`.

    arq reads these as plain class *attributes* -- it does not call them -- so
    `redis_settings` must be a `RedisSettings` instance, not a method. It is
    therefore resolved when this module is imported, which for the worker
    process is startup.
    """

    functions = [run_audit_task]
    max_jobs = 2  # Embedding is CPU-bound; more concurrency just thrashes.
    job_timeout = 1800
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
