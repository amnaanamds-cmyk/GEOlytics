"""Liveness and dependency checks."""

from __future__ import annotations

import httpx
from fastapi import APIRouter
from sqlalchemy import text

from geolytics import __version__
from geolytics.api.schemas import HealthOut
from geolytics.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    settings = get_settings()
    return HealthOut(
        status="ok",
        version=__version__,
        embedding_backend=settings.embedding_backend,
        services={
            "postgres": _check_postgres(),
            "qdrant": _check_http(f"{settings.qdrant_url}/readyz"),
        },
    )


def _check_postgres() -> str:
    try:
        from geolytics.db.session import get_engine

        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:  # noqa: BLE001 - health must report, never raise
        return f"unavailable: {type(exc).__name__}"


def _check_http(url: str) -> str:
    try:
        response = httpx.get(url, timeout=3.0)
        return "ok" if response.is_success else f"http {response.status_code}"
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {type(exc).__name__}"
