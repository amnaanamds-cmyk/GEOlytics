# End-to-end checks

These drive a real browser against a running stack. They are not unit tests:
they assert that signup, an audit run, the charts, the quota refusal, dark
mode and sign-out all work against the live API and worker.

```bash
# 1. Backend, worker and a site to crawl
docker compose up -d
alembic upgrade head
uvicorn geolytics.api.app:app --app-dir backend --port 8000 &
arq geolytics.tasks.queue.WorkerSettings &

# 2. Frontend
npm run build && GEOLYTICS_API_URL=http://127.0.0.1:8000 npm run start &

# 3. Drive it
SHOT_DIR=./shots npm run e2e
```

`SHOT_DIR` receives full-page screenshots at each milestone, which is the
fastest way to see what actually rendered.

`dashboard.mjs` expects a site to crawl at `http://127.0.0.1:8099`; the repo's
`tests/fixtures/site/` is served there by
`python -m http.server 8099 --directory tests/fixtures/site`.

The crawler refuses loopback addresses under the production URL policy, so the
backend must run with `GEOLYTICS_CRAWL_ALLOW_PRIVATE_ADDRESSES=true` and
`GEOLYTICS_CRAWL_RESTRICT_PORTS=false` for these checks — development settings
only, never production.
