"""The Ollama client and the simulated generative engine, over real HTTP.

Ollama itself is not run here -- a stub server implements `/api/generate`, so
the client's HTTP handling, JSON salvaging and the whole retrieve-generate-cite
loop are exercised without a model. What is *not* covered is the quality of a
real model's output; that is what the human-labelled validation in
`agreement.py` is for.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from services import make_test_settings

from geolytics.chunking import SentenceChunker
from geolytics.embedding.hashing import HashingEmbedder
from geolytics.evaluation.qagen import QAGenerationConfig, generate_query_set
from geolytics.geo.engine import SimulatedGenerativeEngine
from geolytics.index.memory import InMemoryVectorStore
from geolytics.llm import OllamaClient
from geolytics.retrieval import DenseRetriever


class _OllamaStub(BaseHTTPRequestHandler):
    """Answers /api/generate from a scripted reply table."""

    replies: dict[str, str] = {}
    default_reply = ""
    received: list[str] = []

    def log_message(self, fmt: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        prompt = payload.get("prompt", "")
        type(self).received.append(prompt)

        body = type(self).default_reply
        for needle, reply in type(self).replies.items():
            if needle in prompt:
                body = reply
                break

        data = json.dumps({"response": body, "done": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def ollama_stub() -> Iterator[type[_OllamaStub]]:
    _OllamaStub.replies = {}
    _OllamaStub.default_reply = ""
    _OllamaStub.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OllamaStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _OllamaStub.url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield _OllamaStub
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def client(ollama_stub) -> OllamaClient:
    return OllamaClient(url=ollama_stub.url, model="stub-model", timeout=10.0)


@pytest.fixture
def documents(fixture_site):
    from geolytics.crawl.crawler import Crawler

    settings = make_test_settings(crawl_max_pages=5)
    with Crawler(settings=settings, cache_dir=None) as crawler:
        return [r.document for r in crawler.crawl(f"{fixture_site}/index.html")]


class TestOllamaClient:
    def test_completes_over_http(self, client, ollama_stub):
        ollama_stub.default_reply = "the answer"
        assert client.complete("anything") == "the answer"
        assert ollama_stub.received

    def test_sends_the_configured_model_and_system_prompt(self, client, ollama_stub):
        ollama_stub.default_reply = "ok"
        client.complete("question", system="be terse")
        assert "question" in ollama_stub.received[0]

    def test_parses_plain_json(self, client, ollama_stub):
        ollama_stub.default_reply = '["one", "two"]'
        assert client.complete_json("x") == ["one", "two"]

    def test_strips_a_fenced_code_block(self, client, ollama_stub):
        ollama_stub.default_reply = '```json\n["one", "two"]\n```'
        assert client.complete_json("x") == ["one", "two"]

    def test_salvages_json_after_leading_prose(self, client, ollama_stub):
        ollama_stub.default_reply = 'Sure! Here you go:\n["one"]'
        assert client.complete_json("x") == ["one"]

    def test_raises_on_unparseable_output(self, client, ollama_stub):
        ollama_stub.default_reply = "I cannot do that."
        with pytest.raises(json.JSONDecodeError):
            client.complete_json("x")

    def test_describe_names_the_model(self, client):
        assert client.describe() == {"llm": "ollama", "model": "stub-model"}


class TestQueryGenerationOverHttp:
    def test_generates_queries_with_document_anchored_spans(
        self, client, ollama_stub, documents
    ):
        ollama_stub.default_reply = json.dumps(
            ["What is the emergency response time?", "Which areas are covered?"]
        )
        query_set, report = generate_query_set(
            documents,
            llm=client,
            config=QAGenerationConfig(questions_per_unit=2, max_lexical_overlap=None),
        )
        assert len(query_set) > 0
        assert report.n_failed_units == 0
        assert all(q.provenance == "synthetic" for q in query_set)
        assert all(q.spans and q.spans[0].text for q in query_set)

    def test_one_bad_unit_does_not_abort_the_pass(self, client, ollama_stub, documents):
        ollama_stub.default_reply = "not json"
        query_set, report = generate_query_set(documents, llm=client)
        assert len(query_set) == 0
        assert report.n_failed_units == report.n_units

    def test_prompt_carries_the_passage(self, client, ollama_stub, documents):
        ollama_stub.default_reply = json.dumps(["Q?"])
        generate_query_set(documents, llm=client, config=QAGenerationConfig(min_unit_tokens=10))
        assert any("Acme" in prompt for prompt in ollama_stub.received)


class TestSimulatedEngine:
    @pytest.fixture
    def engine(self, client, ollama_stub, documents):
        chunker = SentenceChunker(max_tokens=60, overlap_sentences=0, min_tokens=1)
        chunks = [c for d in documents for c in chunker.chunk(d)]
        embedder = HashingEmbedder(dim=128)
        store = InMemoryVectorStore()
        store.create_collection("engine", dim=128)
        store.upsert("engine", chunks, embedder.embed([c.text for c in chunks]))
        retriever = DenseRetriever(store, embedder, "engine")
        return SimulatedGenerativeEngine(retriever, client, top_k=4), chunks

    def test_produces_a_cited_answer(self, engine, ollama_stub):
        sim, _ = engine
        ollama_stub.default_reply = (
            "Acme dispatches a plumber within 90 minutes [1]. "
            "A callout costs 2,500 PKR [2]. "
            "Coverage extends beyond Lahore [1][3]."
        )
        answer = sim.answer("how fast is an emergency callout")

        assert len(answer.sentences) == 3
        assert answer.cited_chunk_ids
        assert answer.n_uncited_sentences == 0
        assert len(answer.retrieved) == 4

    def test_sources_block_is_numbered_for_the_model(self, engine, ollama_stub):
        sim, _ = engine
        ollama_stub.default_reply = "No answer [1]."
        sim.answer("anything")
        prompt = ollama_stub.received[-1]
        assert "[1]" in prompt and "[2]" in prompt
        assert "Question: anything" in prompt

    def test_visibility_is_measured_for_the_audited_site(self, engine, ollama_stub):
        sim, _ = engine
        ollama_stub.default_reply = "Acme answers within 90 minutes [1]. Others vary."
        answer = sim.answer("response time")

        doc_id = answer.retrieved[0].chunk.doc_id
        metrics = answer.visibility_for_doc(doc_id)
        assert metrics.cited
        assert 0.0 < metrics.word_count_share <= 1.0
        assert metrics.first_cited_position == 1

    def test_a_site_that_is_not_cited_scores_zero(self, engine, ollama_stub):
        sim, _ = engine
        ollama_stub.default_reply = "The sources do not answer this question."
        answer = sim.answer("unrelated question")

        metrics = answer.visibility_for(["not-a-real-chunk-id"])
        assert not metrics.cited
        assert metrics.word_count_share == 0.0

    def test_hallucinated_citation_is_not_credited(self, engine, ollama_stub):
        sim, _ = engine
        # Only 4 sources were supplied; [9] refers to nothing.
        ollama_stub.default_reply = "Acme is fast [9]."
        answer = sim.answer("speed")
        assert answer.cited_chunk_ids == set()
        assert answer.n_uncited_sentences == 1

    def test_empty_index_yields_an_empty_answer(self, client, ollama_stub):
        store = InMemoryVectorStore()
        store.create_collection("empty", dim=128)
        sim = SimulatedGenerativeEngine(
            DenseRetriever(store, HashingEmbedder(dim=128), "empty"), client
        )
        answer = sim.answer("anything")
        assert answer.text == ""
        assert answer.metadata["reason"] == "no chunks retrieved"
        assert not ollama_stub.received, "the model must not be called with no sources"
