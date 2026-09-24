# GEOlytics

**Automated Generative Engine Optimization (GEO) and RAG Auditing System**

GEOlytics crawls a website, indexes it through a configurable RAG pipeline, and
evaluates the content and retrieval factors that plausibly affect its visibility
in AI-generated answers. It also runs the pipeline as a controlled experiment,
so the chunking and retrieval choices behind the audit are measured rather than
assumed.

## Scope, stated precisely

This system **simulates and evaluates** factors affecting AI-search visibility.
It does **not** observe the retrieval or ranking internals of ChatGPT,
Perplexity or Google AI Overviews, and it makes no claim about why any of them
selected or ignored a page — that is not observable from outside.

What it can support: *"under a retrieval-augmented generation pipeline with
these components, content with property X was retrieved and cited more often
than content without it."* Correlating that against real engines' observed
behaviour is a separate, explicitly correlational step (see
[docs/methodology.md](docs/methodology.md)).

## What's here

| Layer | Module | Notes |
|-------|--------|-------|
| Crawling | `geolytics.crawl` | robots.txt-compliant, rate-limited, disk-cached |
| Extraction | `geolytics.crawl.extract` | heading-hierarchy-preserving |
| Chunking | `geolytics.chunking` | fixed / sentence / semantic / parent-child |
| Embedding | `geolytics.embedding` | Sentence-Transformers, plus a hashing fixture for CI |
| Indexing | `geolytics.index` | Qdrant for the app, exact NumPy search for experiments |
| Retrieval | `geolytics.retrieval` | dense, BM25, hybrid (RRF / weighted), cross-encoder rerank |
| Evaluation | `geolytics.evaluation` | judgments, projection, metrics, **paired statistics**, harness |
| GEO | `geolytics.geo` | content signals, simulated engine, visibility, fitted scoring |
| API | `geolytics.api` | FastAPI |
| Persistence | `geolytics.db` | PostgreSQL via SQLAlchemy 2.0 |
| Calibration | `geolytics.geo.calibration` | fits signal weights against measured visibility |
| Tasks | `geolytics.tasks` | arq + Redis |

Not yet built: the Next.js frontend. The API it will consume is running and
documented at `/docs`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + test tooling
pip install -e ".[all]"          # everything, including torch

cp .env.example .env
docker compose up -d             # postgres, qdrant, redis

make test-unit                   # 238 tests, no services, no network
geolytics init-db
make api                         # http://localhost:8000/docs
make worker                      # in another shell
make test-integration            # 72 more, against the live stack
```

No Docker daemon, or blocked by a registry rate limit? The native recipe is in
[docs/running.md](docs/running.md) — that is the path this stack was verified
on.

### Command line

```bash
geolytics audit https://example.com --max-pages 20
geolytics experiment https://example.com --persist-as chunking
geolytics calibrate https://example.com --max-pages 30   # fits the GEO weights
geolytics check-metrics                                  # see below
```

Two defaults are development fixtures and are refused or flagged in anything
resembling a real run: the `hashing` embedder (no semantics, refused when
`GEOLYTICS_ENV=production`) and `GEOLYTICS_LLM_BACKEND=none` (template query
generation, marked `provenance="heuristic"`). Both are covered in
[docs/running.md](docs/running.md).

## The research contribution

The defensible contribution is **the comparison**, not the wrapper around an
LLM. Three things make the comparison hold up:

### 1. Ground truth that is portable across chunkers

Retrieval precision and recall need relevance judgments, and none exist for an
arbitrary website. GEOlytics generates them — but a judgment naming a chunk id
is worthless the moment the chunker changes, which is precisely the variable
under test.

So judgments are anchored to the **source document** (`SpanRelevance`: a
character span plus the gold passage text) and projected onto each chunking
independently (`evaluation.projection`). One authored judgment set scores every
condition fairly.

Synthetic queries are lexically biased toward their source passage, which
flatters BM25. That bias is measured per query (`lexical_overlap`), queries
above a threshold are dropped, and the method is validated against
hand-labelled judgments with Cohen's κ (`evaluation.agreement`).

### 2. Similarity metrics are not a real experimental axis

For L2-normalised vectors, `dot(a,b) = cos(a,b)` and `‖a−b‖² = 2 − 2·dot(a,b)`.
Euclidean distance is a strictly decreasing function of the dot product, so all
three induce **identical rankings**. Sentence-Transformers normalises by
default; Qdrant normalises on upsert for `Cosine` collections.

`geolytics check-metrics` proves it empirically (Kendall's τ = 1.0, identical
top-k on 100% of queries). Report that as a result and spend the freed axis on
a factor that actually varies — embedding model, hybrid fusion weight α, or a
reranker.

The check is tie-aware: identical passages tie, and float32 noise between two
mathematically equivalent metrics must not read as a ranking difference.

### 3. Every comparison carries a significance test

`ExperimentHarness` stores **per-query** scores, never just means. Comparisons
report the paired difference, a percentile bootstrap 95% CI, a Wilcoxon
signed-rank p-value, Cliff's δ, and a Holm–Bonferroni correction across the
comparison family:

```
| A               | B           | metric  | mean A | mean B | diff    | 95% CI             | p      | p (Holm) | delta  | effect | sig |
| sentence+hybrid | fixed+dense | ndcg@10 | 0.8214 | 0.7103 | +0.1111 | [+0.0412, +0.1802] | 0.0021 | 0.0063   | +0.341 | medium | yes |
```

Wilcoxon rather than a paired t-test because per-query retrieval metrics are
bounded, discrete and skewed. Result tables always carry `n chunks` and
`mean tok`, so a reader can tell a retrieval effect from a chunk-size effect.

### 4. GEO weights are fitted, not asserted

Inventing "schema markup is 15% of the score" is the most attackable move
available. Default weights are uniform and **labelled unfitted** — the API
returns a caveat string and the CLI prints a warning.

`geolytics calibrate` closes the loop: it runs the simulated generative engine
over the evaluation queries, measures the position-weighted share of each
answer that came from each page, and fits the weights by non-negative least
squares against that. It records R², the observation count and the
collinearity between signals, and it refuses to fit at all when there are
fewer pages than signals or when too few pages were cited for the target to
carry any variance.

The signal set comes from Aggarwal et al., *GEO: Generative Engine
Optimization* (KDD 2024) plus properties that follow from how RAG pipelines
work. **Verify that paper's exact metric definitions and reported figures
before citing numbers.**

## Testing

```bash
make test-unit         # 238 tests: no network, no GPU, no services
make test-integration  #  72 tests: skips whatever is not running
make lint
make typecheck
```

310 tests in total, all passing. The unit suite needs nothing running — the
`hashing` embedder is deterministic and the in-memory store does exact search.

The integration suite exercises what unit tests cannot:

| Area | What is actually run |
|---|---|
| Crawler | a real HTTP server for `tests/fixtures/site/` — robots.txt, `Crawl-delay`, link following, host scoping, caching, content dedup |
| Qdrant | real server: upsert, payload fidelity, recreate, and agreement with exact NumPy search |
| PostgreSQL | real schema, JSON round-trips, cascade deletes, audit persistence, document rehydration |
| Redis / arq | an enqueued job drained by a real burst worker |
| LLM + engine | a stub `/api/generate` server: HTTP, JSON salvaging, citation parsing, visibility |
| Sentence-Transformers | the wrapper against a stub model — prefixes, normalisation, dtype, dimension |

Each service-backed test skips itself when its service is absent, so the
default run stays green on a bare machine.

## Documentation

- [docs/running.md](docs/running.md) — how to run it, including without Docker, and the two fixtures to switch off before reporting anything
- [docs/methodology.md](docs/methodology.md) — the experimental protocol, threats to validity, and what to write in the report
- [docs/architecture.md](docs/architecture.md) — layering and the reasoning behind each component choice

## Licence

MIT
