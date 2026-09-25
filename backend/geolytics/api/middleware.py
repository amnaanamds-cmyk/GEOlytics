"""HTTP middleware: correlation ids, access logs, security headers, rate limits."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from geolytics.observability import bind_request, new_request_id, request_id_var

logger = logging.getLogger("geolytics.access")

Handler = Callable[[Request], Awaitable[Response]]

# Sent on every response. Defence in depth for the JSON API and, more
# importantly, for the interactive docs pages, which do render HTML.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, log the outcome, and stamp security headers."""

    def __init__(self, app, hsts: bool = False) -> None:
        super().__init__(app)
        self.hsts = hsts

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        # Accept an inbound id so a trace spans the frontend and the API, but
        # bound its length -- it is echoed into a response header.
        incoming = (request.headers.get("x-request-id") or "")[:64]
        request_id = incoming or new_request_id()
        bind_request(request_id)
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        principal = getattr(request.state, "principal", None)
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "org_id": getattr(principal, "org_id", None),
                "actor": getattr(principal, "audit_actor", None),
            },
        )

        response.headers["X-Request-ID"] = request_id
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if self.hsts:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket limiting, keyed by API key, user or client address.

    Keying by credential rather than by IP is what makes this useful: many
    customers share one NAT address, and one abusive key should not throttle
    the rest.

    Unauthenticated auth endpoints get a much tighter bucket, because that is
    where credential stuffing lands.
    """

    def __init__(self, app, limiter, auth_limiter, exempt_paths: frozenset[str]) -> None:
        super().__init__(app)
        self.limiter = limiter
        self.auth_limiter = auth_limiter
        self.exempt_paths = exempt_paths

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        path = request.url.path
        if path in self.exempt_paths or request.method == "OPTIONS":
            return await call_next(request)

        is_auth = path.startswith("/auth/")
        limiter = self.auth_limiter if is_auth else self.limiter
        result = limiter.check(self._key(request, is_auth))

        if not result.allowed:
            logger.warning(
                "rate limited",
                extra={"path": path, "retry_after": result.retry_after},
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "message": "too many requests",
                    "retry_after": result.retry_after,
                    "request_id": request_id_var.get(),
                },
                headers=result.headers(),
            )

        response = await call_next(request)
        for header, value in result.headers().items():
            response.headers.setdefault(header, value)
        return response

    def _key(self, request: Request, is_auth: bool) -> str:
        """Bucket key: the credential when there is one, else the peer address.

        The raw credential is never used as the key -- only a short hash of it
        -- so Redis never holds anything that works as a token.
        """
        if is_auth:
            return f"auth:{self._peer(request)}"

        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer ") and len(header) > 12:
            import hashlib

            digest = hashlib.sha256(header[7:].strip().encode()).hexdigest()[:24]
            return f"cred:{digest}"
        return f"ip:{self._peer(request)}"

    @staticmethod
    def _peer(request: Request) -> str:
        from geolytics.api.deps import client_ip

        return client_ip(request)
