# Architecture

```
                    ┌──────────────┐
  target website ──▶│   crawl/     │  robots.txt, rate limit, disk cache
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │crawl.extract │  heading-aware → Document + Section[]
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  chunking/   │  fixed │ sentence │ semantic │ parent-child
                    └──────┬───────┘        ChunkingStrategy
                           ▼
                    ┌──────────────┐
                    │  embedding/  │  Sentence-Transformers │ hashing (CI)
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │    index/    │  Qdrant (app) │ NumPy exact (experiments)
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  retrieval/  │  dense │ BM25 │ hybrid │ reranked
                    └──────┬───────┘
                ┌──────────┴──────────┐
                ▼                     ▼
        ┌──────────────┐      ┌──────────────┐
        │ evaluation/  │      │     geo/     │
        │ the research │      │ the product  │
        │    record    │      │              │
        └──────┬───────┘      └──────┬───────┘
               │                     │
        qrels, projection,    signals, simulated
        metrics, paired       engine, visibility,
        statistics, harness   fitted scoring
               │                     │
               └──────────┬──────────┘
                          ▼
                   ┌──────────────┐
                   │     api/     │  FastAPI  ──▶ (Next.js, not yet built)
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │     db/      │  PostgreSQL
                   └──────────────┘
```

The two consumers — `evaluation/` and `geo/` — sit on the *same* chunking,
embedding, indexing and retrieval components. That is the point of the
layering: a finding about chunking transfers directly into what the product
does, rather than being a separate paper exercise.

## Component choices and their reasons

**arq, not Celery.** asyncio-native (shares FastAPI's model), one settings
class, same Redis. The workload is a handful of long crawl jobs, not a
high-throughput stream, so Celery's extra surface buys nothing here.

**Qdrant, not pgvector,** despite PostgreSQL already being in the stack.
pgvector would remove a service; Qdrant earns its place because this project is
*about* retrieval comparison and uses named vectors, payload filtering and
sparse/hybrid support. Defend it on those grounds, not on "it's a vector DB".

**Exact NumPy search for experiments.** HNSW's approximate recall is a second
source of variance layered on top of the factor under test. Experiments use
exact search to isolate the effect; the deployed app uses Qdrant. If the report
claims the two agree, *measure* the ANN recall gap rather than assuming it.

**A hashing embedder for CI.** The evaluation suite must run from a clean clone
with no GPU and no model download. It is a fixture, not a baseline — it carries
no semantics beyond lexical overlap, and `build_embedder` refuses it in
production settings.

**Chunks are not a database table.** They are derived from a page plus a
chunking configuration, they multiply by the number of strategies under test,
and the vectors live in Qdrant. Persisting them relationally would duplicate
the index with no query that needs it. `Page.sections` is stored so chunking
can be re-run without re-crawling.

**`QueryRecord` stores per-query metrics.** Storing only aggregate means would
make every significance test in the report unreproducible after the fact.

## Key interfaces

```python
class ChunkingStrategy(ABC):
    name: str
    def chunk(self, document: Document) -> list[Chunk]: ...
    @property
    def params(self) -> dict[str, Any]: ...      # recorded with every result

class Embedder(ABC):
    @property
    def dim(self) -> int: ...
    @property
    def normalized(self) -> bool: ...            # decides the metric question
    def embed(self, texts) -> np.ndarray: ...
    def embed_query(self, texts) -> np.ndarray:  # asymmetric models need this
        ...

class Retriever(ABC):
    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]: ...
```

`Embedder.embed_query` is separate from `embed` because asymmetric models (E5,
BGE, GTR) expect a different prefix on queries than on passages. Getting it
wrong is a silent ~10-point nDCG loss, so the split is mandatory in the
interface even where a backend treats both the same.

`ScoredChunk.score` is always higher-is-better; distance metrics are negated on
the way out, so every retriever, fusion step and metric can assume descending
order.

## Adding a chunking strategy

1. Subclass `ChunkingStrategy`, implement `chunk()` and `params`.
2. Register it in `chunking/registry.py`.
3. Add it to the grid in `experiments/chunking_grid.py`.

Nothing else changes — projection, metrics and statistics are chunker-agnostic
by construction.
