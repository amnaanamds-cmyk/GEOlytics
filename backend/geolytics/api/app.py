"""Application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from geolytics import __version__
from geolytics.api.middleware import RateLimitMiddleware, RequestContextMiddleware
from geolytics.api.routes import audits, auth, billing, experiments, health, orgs
from geolytics.config import Settings, get_settings
from geolytics.crawl.guard import UrlPolicyError
from geolytics.observability import configure_logging, request_id_var
from geolytics.ratelimit import build_limiter
from geolytics.tenancy.quotas import QuotaExceeded

logger = logging.getLogger(__name__)

DESCRIPTION = """
Automated Generative Engine Optimization (GEO) and RAG auditing.

Evaluates factors that plausibly affect AI-search visibility by simulating a
retrieval-augmented generative engine over a customer's own site. It does not
observe, and does not claim to explain, the internal ranking decisions of any
commercial search or answer engine.

**Authentication.** Send either a JWT access token from `/auth/login` or an
API key from `/org/keys` as `Authorization: Bearer <token>`. Every response is
scoped to the organisation that credential belongs to.
"""

# Paths that must answer even when a caller is over their rate limit: health
# checks are how a load balancer decides whether to keep the instance.
_RATE_LIMIT_EXEMPT = frozenset({"/health", "/health/ready", "/health/live", "/openapi.json"})


def _error(status_code: int, error: str, message: str, **extra: object) -> JSONResponse:
    """One error shape across the whole API, always carrying the request id."""
    body: dict[str, object] = {"error": error, "message": message}
    body.update(extra)
    if rid := request_id_var.get():
        body["request_id"] = rid
    return JSONResponse(status_code=status_code, content=body)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    logger.info(
        "starting",
        extra={
            "version": __version__,
            "env": settings.env,
            "embedding_backend": settings.embedding_backend,
            "billing_enabled": settings.billing_enabled,
        },
    )
    if settings.env != "production" and settings.secret_key_is_ephemeral:
        # Worth saying out loud: tokens will stop working on restart, and two
        # replicas would reject each other's tokens.
        logger.warning(
            "GEOLYTICS_SECRET_KEY is unset; tokens are signed with a per-process key "
            "and will not survive a restart"
        )
    yield
    logger.info("shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(
        level=settings.log_level,
        service=settings.service_name,
        env=settings.env,
        json_output=settings.env != "development",
    )

    application = FastAPI(
        title="GEOlytics",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        # Interactive docs are useful internally and are attack surface
        # externally; they are off in production.
        docs_url=None if settings.env == "production" else "/docs",
        redoc_url=None if settings.env == "production" else "/redoc",
    )
    application.state.settings = settings

    if settings.rate_limit_enabled:
        application.add_middleware(
            RateLimitMiddleware,
            limiter=build_limiter(
                settings.redis_url, settings.rate_limit_per_minute, settings.rate_limit_burst
            ),
            auth_limiter=build_limiter(
                settings.redis_url,
                settings.auth_rate_limit_per_minute,
                burst=5,
                namespace="rl:auth",
            ),
            exempt_paths=_RATE_LIMIT_EXEMPT,
        )

    application.add_middleware(
        RequestContextMiddleware, hsts=settings.env == "production"
    )
    application.add_middleware(
        CORSMiddleware,
        # Explicit origins, never "*": the API is credentialed, and a wildcard
        # would let any site spend a signed-in customer's quota.
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
        max_age=600,
    )

    for router in (
        health.router, auth.router, orgs.router,
        audits.router, experiments.router, billing.router,
    ):
        application.include_router(router)

    _install_exception_handlers(application)
    return application


def _install_exception_handlers(application: FastAPI) -> None:
    @application.exception_handler(QuotaExceeded)
    async def _quota(request: Request, exc: QuotaExceeded) -> JSONResponse:
        return _error(
            status.HTTP_402_PAYMENT_REQUIRED,
            "quota_exceeded",
            str(exc),
            limit=exc.limit_name,
            allowed=exc.limit,
            used=exc.used,
        )

    @application.exception_handler(UrlPolicyError)
    async def _policy(request: Request, exc: UrlPolicyError) -> JSONResponse:
        return _error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "url_not_allowed",
            exc.detail,
            reason=exc.reason,
        )

    @application.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "validation_error",
            "request body or parameters are invalid",
            # str(): a validation error can carry the offending input, which
            # for /auth/signup is a password.
            details=[
                {"field": ".".join(str(p) for p in e["loc"]), "problem": e["msg"]}
                for e in exc.errors()
            ],
        )

    @application.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Logged with the stack trace; the caller gets the request id and
        # nothing else. Internal messages leak table names and file paths.
        logger.exception(
            "unhandled exception",
            extra={"path": request.url.path, "method": request.method},
        )
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "an unexpected error occurred; quote the request id when reporting it",
        )


app = create_app()
