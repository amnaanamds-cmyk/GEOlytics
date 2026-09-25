# syntax=docker/dockerfile:1

# ---- builder -------------------------------------------------------------
# A separate stage so compilers and build headers never reach the runtime
# image: they are attack surface and roughly double its size.
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY backend ./backend

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# sentence-transformers is intentionally excluded: it pulls torch (~2.5 GB)
# and only the worker needs it. Build the worker image with
# `--build-arg EXTRAS=...,embed` when running real embeddings.
ARG EXTRAS="vector,crawl,queue,postgres,lexical"
RUN pip install --no-cache-dir ".[${EXTRAS}]"

# ---- runtime -------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PATH="/opt/venv/bin:$PATH" \
    GEOLYTICS_ENV=production

RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 --shell /usr/sbin/nologin geolytics

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=geolytics:geolytics backend ./backend
COPY --chown=geolytics:geolytics migrations ./migrations
COPY --chown=geolytics:geolytics alembic.ini ./

# Never root: a container escape from an application bug should land as an
# unprivileged user with nothing to write to.
USER geolytics

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health/live || exit 1

# One worker per process; scale with replicas, not threads. The audit path is
# CPU-bound on embeddings, so in-process concurrency mostly adds contention.
CMD ["uvicorn", "geolytics.api.app:app", \
     "--app-dir", "backend", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*", \
     "--no-server-header"]
