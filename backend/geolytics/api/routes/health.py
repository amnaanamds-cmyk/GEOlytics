"""Liveness and readiness checks."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Response, status
from sqlalchemy import text

from geolytics import __version__
from geolytics.api.schemas import HealthOut
from geolytics.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health/live", status_code=status.HTTP_204_NO_CONTENT)
def live() -> Response:
    """Process is up. Deliberately checks nothing else.

    A liveness probe that touches the database restarts healthy application
    instances whenever the database blips, turning one outage into two.
    """
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/health/ready", response_model=HealthOut)
def ready(response: Response) -> HealthOut:
    """Ready to serve traffic: dependencies reachable.

    Returns 503 when a hard dependency is down so a load balancer stops
    sending traffic here. Qdrant is soft -- audits queue and retry without it,
    while reads keep working -- so it does not fail readiness on its own.
    """
    settings = get_settings()
    services = {
        "postgres": _check_postgres(),
        "qdrant": _check_http(f"{settings.qdrant_url}/readyz"),
        "redis": _check_redis(settings.redis_url),
    }
    hard = ("postgres", "redis")
    healthy = all(services[name] == "ok" for name in hard)
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthOut(
        status="ok" if healthy else "degraded",
        version=__version__,
        embedding_backend=settings.embedding_backend,
        services=services,
    )


@router.get("/health", response_model=HealthOut)
def health(response: Response) -> HealthOut:
    return ready(response)


def _check_postgres() -> str:
    try:
        from geolytics.db.session import get_engine

        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:  # noqa: BLE001 - health must report, never raise
        return f"unavailable: {type(exc).__name__}"


def _check_redis(url: str) -> str:
    try:
        import redis

        client = redis.Redis.from_url(url, socket_connect_timeout=2)
        client.ping()
        client.close()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {type(exc).__name__}"


def _check_http(url: str) -> str:
    try:
        response = httpx.get(url, timeout=3.0)
        return "ok" if response.is_success else f"http {response.status_code}"
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {type(exc).__name__}"
