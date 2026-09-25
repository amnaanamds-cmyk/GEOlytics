# Deploying

## Shape

```
              ┌──────────────┐
  browser ───▶│  dashboard   │  Next.js. Holds the session in httpOnly
              │  (Next.js)   │  cookies and proxies every API call, so the
              └──────┬───────┘  browser never handles a credential.
                     ▼
              ┌──────────────┐
              │     API      │  Stateless. Scale with replicas.
              └──────┬───────┘  Makes NO outbound requests.
         ┌───────────┼───────────┐
         ▼           ▼           ▼
   ┌──────────┐ ┌────────┐ ┌──────────┐
   │ Postgres │ │ Redis  │ │  Qdrant  │
   └──────────┘ └────┬───┘ └──────────┘
                     ▼
              ┌──────────────┐
              │   worker     │  Crawls and embeds. The ONLY component
              │    (arq)     │  with outbound network access.
              └──────────────┘
```

**Put the worker on its own egress-restricted network with no instance role.**
It is the only component that fetches customer-supplied URLs, so it is the only
one in the SSRF blast radius. The API needs no outbound access at all.

## First deploy

```bash
cp .env.example .env.production
# Generate a real signing key — this one is not optional:
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Set at minimum:

```bash
GEOLYTICS_ENV=production
GEOLYTICS_SECRET_KEY=<the generated key>
GEOLYTICS_POSTGRES_DSN=postgresql+psycopg://...
GEOLYTICS_REDIS_URL=redis://...
GEOLYTICS_QDRANT_URL=http://...
GEOLYTICS_CORS_ORIGINS=["https://app.yourdomain.com"]
GEOLYTICS_EMBEDDING_BACKEND=sentence-transformers
```

Then:

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml up -d
```

The `migrate` service runs `alembic upgrade head` before the API starts.

Production **refuses to boot** on an unsafe configuration — a placeholder
signing key, wildcard CORS, a crawler allowed to reach private addresses, or
the hashing embedder. That is deliberate: each of those is silent at boot and
expensive later. If the API exits at startup, read the error; it names every
problem at once.

## Images

Two, from one codebase:

- `Dockerfile` — the API. No torch, so it stays small. Multi-stage, non-root,
  read-only root filesystem, `no-new-privileges`.
- `Dockerfile.worker` — the API image plus `sentence-transformers`. Mount a
  volume at `HF_HOME` so a restart does not re-download the model.

```bash
docker build -t geolytics-api:latest .
docker build -f Dockerfile.worker -t geolytics-worker:latest .
```

## Migrations

```bash
alembic upgrade head          # apply
alembic revision --autogenerate -m "what changed"
alembic downgrade -1          # roll back one
```

The database URL comes from application settings, not `alembic.ini`, so a
migration can never target a different database from the one the app uses. CI
fails the build if the models and migrations have diverged.

`geolytics init-db` exists for throwaway databases only. Use Alembic for
anything whose data you care about.

## Bootstrapping the first tenant

```bash
GEOLYTICS_ADMIN_PASSWORD='...' \
  geolytics create-org "Acme Plumbing" --email owner@acme.com --plan growth
```

Or leave signup enabled and let the first customer create their own. Set
`GEOLYTICS_SIGNUP_ENABLED=false` to close public signup afterwards.

## Health checks

| Endpoint | Use |
|---|---|
| `GET /health/live` | liveness. Checks nothing else, on purpose — a liveness probe that touches the database restarts healthy instances whenever the database blips, turning one outage into two |
| `GET /health/ready` | readiness. 503 when Postgres or Redis is unreachable, so the load balancer stops sending traffic. Qdrant is soft: audits queue and reads keep working without it |

## Scaling

- **API**: stateless, scale horizontally. All replicas must share
  `GEOLYTICS_SECRET_KEY` or they will reject each other's tokens.
- **Worker**: one audit at a time per worker (`max_jobs = 2`). Embedding is
  CPU-bound, so add replicas rather than threads. Plan concurrency limits cap
  how much any one customer can occupy.
- **Postgres**: the per-query metric rows grow fastest. Plan a retention job
  against `query_records`; billing history lives in `usage_counters` and must
  outlive it.
- **Qdrant**: one collection per audit. Drop collections for deleted audits;
  nothing does this automatically yet.

## Billing

```bash
GEOLYTICS_BILLING_ENABLED=true
GEOLYTICS_STRIPE_SECRET_KEY=sk_live_...
GEOLYTICS_STRIPE_WEBHOOK_SECRET=whsec_...
GEOLYTICS_STRIPE_PRICE_IDS={"starter":"price_...","growth":"price_..."}
```

Point a Stripe webhook at `POST /billing/webhook` for
`checkout.session.completed`, `customer.subscription.*` and
`invoice.payment_failed`.

The webhook secret is mandatory when billing is on, and the app refuses to
start without it: an unverified webhook endpoint is a free upgrade button.

**Not verified in this repository.** Billing is tested against a fake client
and locally signed events. Run Stripe's test mode end to end before charging
anyone.

## Logs

JSON, one object per line, every line carrying the request id that is also
returned in `X-Request-ID`. A customer quoting an id from a failed call can be
found directly. Passwords, tokens and keys are redacted before a line is
written.

## Before you charge anyone

- [ ] `GEOLYTICS_SECRET_KEY` set, and identical across replicas
- [ ] Worker on an egress-restricted network, no instance role
- [ ] TLS terminated; HSTS is sent automatically in production
- [ ] Postgres backups, and a **restore** actually tested
- [ ] Stripe run end to end in test mode
- [ ] `docs/security.md` § 8 read — email verification, password reset and MFA
      are **not** implemented
- [ ] `SECURITY.md` published with a contact address
