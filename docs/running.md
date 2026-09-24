# Running the system

Everything below was executed against a live stack — PostgreSQL 16, Redis 7 and
Qdrant 1.19.1 — not just written down.

## 1. Services

The normal path:

```bash
docker compose up -d
```

### Without a Docker daemon

Behind a registry rate limit or on a host with no daemon, run the services
natively. This is the exact recipe used to verify the system:

```bash
# PostgreSQL + Redis from the distro
apt-get install -y postgresql postgresql-contrib redis-server
pg_ctlcluster 16 main start
su postgres -c "psql -c \"CREATE ROLE geolytics LOGIN PASSWORD 'geolytics' SUPERUSER\""
su postgres -c "createdb -O geolytics geolytics"
su postgres -c "createdb -O geolytics geolytics_test"   # for tests/integration
redis-server --daemonize yes --save '' --appendonly no

# Qdrant from its release binary
mkdir -p /opt/qdrant/storage && cd /opt/qdrant
curl -L -o qdrant.tgz \
  https://github.com/qdrant/qdrant/releases/download/v1.19.1/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar xzf qdrant.tgz
QDRANT__STORAGE__STORAGE_PATH=/opt/qdrant/storage ./qdrant &
```

Keep the Qdrant **server** and `qdrant-client` within one minor version of each
other. The client warns on a mismatch, and `search()` was removed from the
client in 1.19 — which is why `pyproject.toml` floors it at 1.12 and
`docker-compose.yml` pins the server to a matching release.

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                       # core + tooling
pip install -e ".[dev,vector,crawl,queue,postgres,lexical]"   # everything but torch
pip install -e ".[all]"                       # + sentence-transformers/torch
```

`sentence-transformers` is optional on purpose: it pulls torch (~2.5 GB from
PyPI) and every other layer runs without it.

## 3. Configure

```bash
cp .env.example .env
```

For a first run the defaults work: the `hashing` embedder needs no model
download, and `GEOLYTICS_LLM_BACKEND=none` falls back to the template query
generator. Both are development fixtures — see the caveats below.

## 4. Run

```bash
geolytics init-db
make api      # http://localhost:8000/docs
make worker   # arq, in a second shell
```

## 5. Exercise it

```bash
# Score a site
geolytics audit https://example.com --max-pages 20

# Run the chunking x retrieval grid, store it, read it back over HTTP
geolytics experiment https://example.com --persist-as chunking
curl 'localhost:8000/experiments/chunking?metric=ndcg@10&baseline=fixed%2Bdense'

# Fit GEO weights against measured visibility (needs a real LLM)
geolytics calibrate https://example.com --max-pages 30

# Prove cosine/dot/euclidean are the same ranking for your embedder
geolytics check-metrics
```

Or queue an audit through the API and let the worker run it:

```bash
curl -X POST localhost:8000/audits -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com","max_pages":10,"chunker":"sentence"}'
curl localhost:8000/audits/1
```

## 6. Tests

```bash
make test-unit          # no services, no network
make test-integration   # skips whatever is not running
make test               # both
```

Integration tests serve a fixture site (`tests/fixtures/site/`) over a local
HTTP server, so the crawler is exercised against real HTTP — robots.txt,
`Crawl-delay`, link following, caching, content deduplication — without
crawling anyone's actual website.

## Two development defaults you must change before reporting results

Both are loud on purpose, but they are easy to leave switched on.

**`GEOLYTICS_EMBEDDING_BACKEND=hashing`** is a character-n-gram hashing
fixture. It carries no semantics beyond lexical overlap. `build_embedder`
refuses it when `GEOLYTICS_ENV=production`. Switch to
`sentence-transformers` for anything real:

```bash
GEOLYTICS_EMBEDDING_BACKEND=sentence-transformers
GEOLYTICS_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GEOLYTICS_EMBEDDING_DIM=384
```

**`GEOLYTICS_LLM_BACKEND=none`** makes query generation fall back to templates.
Those queries are marked `provenance="heuristic"`, and the experiment output
prints a warning, because template questions reuse their source passage's
vocabulary and flatter lexical retrieval. Install Ollama and set:

```bash
GEOLYTICS_LLM_BACKEND=ollama
GEOLYTICS_OLLAMA_MODEL=llama3.1:8b
```

`geolytics calibrate` refuses to run at all without an LLM, because it measures
visibility inside generated answers and there would be none.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `'QdrantClient' object has no attribute 'search'` | client older than this code expects, or a very old pin — use `qdrant-client>=1.12` |
| Qdrant version-compatibility `UserWarning` | server and client differ by more than one minor version |
| `'staticmethod' object has no attribute 'host'` from `arq` | `WorkerSettings.redis_settings` must be a `RedisSettings` **instance**, not a method |
| `enqueue_audit() is for synchronous callers` | called from inside a coroutine; `await enqueue_audit_async()` instead |
| Audit stays `pending` | no worker running, or it is pointed at a different Redis database |
| `0 pages were cited` from `calibrate` | the model returned answers without `[n]` citation markers |
| Experiment reports very few queries | check `n_units_too_short` in the generation summary and lower `min_unit_tokens` |
