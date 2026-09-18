"""Application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from geolytics import __version__
from geolytics.api.routes import audits, experiments, health
from geolytics.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    application = FastAPI(
        title="GEOlytics",
        version=__version__,
        description=(
            "Automated GEO and RAG auditing. Evaluates factors that plausibly affect "
            "AI-search visibility by simulating a retrieval-augmented generative engine. "
            "It does not observe, and does not claim to explain, the internal ranking "
            "decisions of any commercial search or answer engine."
        ),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"] if settings.env != "production" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(health.router)
    application.include_router(audits.router)
    application.include_router(experiments.router)
    return application


app = create_app()
