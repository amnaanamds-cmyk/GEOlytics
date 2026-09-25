"""Structured logging and request correlation.

Logs are JSON because they are read by a log aggregator, not by a person
tailing a file: an operator asking "what happened to audit 4821 for org 12"
needs to filter on fields, which text lines cannot support.

Every log line carries the request id, and every response returns it in
`X-Request-ID`, so a customer can quote one from a failed call and it can be
found in the logs directly.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
org_id_var: contextvars.ContextVar[int | None] = contextvars.ContextVar("org_id", default=None)

# Never log these, whatever key they arrive under. A password or token in a
# log store is a credential leak that outlives the request by months.
_REDACTED_KEYS = frozenset(
    {
        "password", "new_password", "current_password", "secret", "token",
        "access_token", "refresh_token", "api_key", "authorization",
        "secret_key", "password_hash", "stripe_secret_key", "client_secret",
    }
)
_REDACTED = "[redacted]"

_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"asctime", "message", "taskName"}


def redact(value: Any, _depth: int = 0) -> Any:
    """Recursively replace sensitive values. Depth-bounded against cycles."""
    if _depth > 6:
        return "[truncated]"
    if isinstance(value, dict):
        return {
            k: (_REDACTED if k.lower() in _REDACTED_KEYS else redact(v, _depth + 1))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v, _depth + 1) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def __init__(self, service: str = "geolytics", env: str = "development") -> None:
        super().__init__()
        self.service = service
        self.env = env

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service,
            "env": self.env,
        }

        if rid := request_id_var.get():
            payload["request_id"] = rid
        if (org := org_id_var.get()) is not None:
            payload["org_id"] = org
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        extras = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS}
        if extras:
            payload.update(redact(extras))

        # default=str so a stray datetime or Decimal cannot make logging raise
        # and take down the request it was describing.
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", service: str = "geolytics", env: str = "development",
                      json_output: bool = True) -> None:
    """Install the root handler. Safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter(service=service, env=env)
        if json_output
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root.addHandler(handler)

    # uvicorn installs its own handlers; let them propagate to ours instead so
    # access logs are structured too.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True


def new_request_id() -> str:
    return uuid.uuid4().hex


def bind_request(request_id: str, org_id: int | None = None) -> None:
    request_id_var.set(request_id)
    org_id_var.set(org_id)
