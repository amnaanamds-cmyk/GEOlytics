"""Fitting GEO weights against visibility measured in the simulated engine.

Uses a synthetic corpus rather than the fixture site: fitting ten signal
weights needs more than ten pages, and the corpus has to vary along the
signals or the regression has nothing to separate.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from geolytics.chunking import SentenceChunker
from geolytics.crawl.extract import extract_document
from geolytics.embedding.hashing import HashingEmbedder
from geolytics.geo.calibration import (
    as_dict,
    calibrate_weights,
    measure_visibility,
    top_correlations,
)
from geolytics.geo.engine import SimulatedGenerativeEngine
from geolytics.geo.scoring import GEOScorer
from geolytics.index.memory import InMemoryVectorStore
from geolytics.llm import OllamaClient
from geolytics.retrieval.dense import DenseRetriever

TOPICS = [
    "drain clearing", "boiler servicing", "leak detection", "tap replacement",
    "tank cleaning", "pipe thawing", "radiator bleeding", "shower fitting",
    "toilet repair", "pump installation", "gutter clearing", "valve replacement",
    "water testing", "sink unblocking", "meter fitting",
]


def _page(index: int, topic: str) -> tuple[str, str]:
    """A page whose GEO signals vary systematically with its index.

    Even-indexed pages are "well optimised": schema markup, statistics, a
    citation and a date. Odd-indexed pages are thin. Without that spread every
    signal would be constant and the fit would be meaningless.
    """
    slug = topic.replace(" ", "-")
    rich = index % 2 == 0
    schema = (
        '<script type="application/ld+json">'
        f'{{"@context":"https://schema.org","@type":"Service","name":"{topic}",'
        '"provider":{"@type":"LocalBusiness","name":"Acme"}}</script>'
        if rich
        else ""
    )
    stats = (
        f"<p>According to a 2024 survey, {60 + index} percent of {topic} jobs "
        f"were completed within 2 hours. A {topic} callout costs "
        f"{1000 + index * 100} PKR. Source: Acme service records.</p>"
        if rich
        else f"<p>We do {topic} well. Call us.</p>"
    )
    html = f"""<!DOCTYPE html><html><head><title>{topic.title()} — Acme</title>{schema}</head>
<body><h1>{topic.title()}</h1>
<p>Acme provides {topic} across Lahore. {topic.title()} is booked by phone or online.</p>
<h2>What {topic} includes</h2>
{stats}
<h2>Booking {topic}</h2>
<p>Acme quotes every {topic} job before work begins. The {topic} quote is held for 30 days.</p>
</body></html>"""
    return f"https://acme.example/{slug}", html


class _CitingStub(BaseHTTPRequestHandler):
    """Answers by citing the first two sources it was given."""

    def log_message(self, fmt: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps(
            {
                "response": (
                    "Acme handles this service across Lahore [1]. "
                    "Pricing is published up front [2]. "
                    "Bookings are confirmed by phone."
                ),
                "done": True,
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def citing_llm() -> Iterator[OllamaClient]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CitingStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield OllamaClient(url=f"http://127.0.0.1:{server.server_address[1]}", model="stub")
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def corpus():
    documents, html_by_doc = [], {}
    for i, topic in enumerate(TOPICS):
        url, html = _page(i, topic)
        document = extract_document(html, url)
        documents.append(document)
        html_by_doc[document.doc_id] = html
    return documents, html_by_doc


@pytest.fixture
def harness(corpus, citing_llm):
    documents, html_by_doc = corpus
    chunker = SentenceChunker(max_tokens=60, overlap_sentences=0, min_tokens=1)
    chunks_by_doc = {d.doc_id: chunker.chunk(d) for d in documents}
    chunks = [c for group in chunks_by_doc.values() for c in group]

    embedder = HashingEmbedder(dim=256)
    store = InMemoryVectorStore()
    store.create_collection("cal", dim=256)
    store.upsert("cal", chunks, embedder.embed([c.text for c in chunks]))

    engine = SimulatedGenerativeEngine(
        DenseRetriever(store, embedder, "cal"), citing_llm, top_k=4
    )
    queries = [f"how does {t} work and what does it cost" for t in TOPICS]
    return documents, chunks_by_doc, html_by_doc, engine, queries


class TestMeasureVisibility:
    def test_every_document_gets_a_score(self, harness):
        documents, _, _, engine, queries = harness
        visibility = measure_visibility(engine, queries, [d.doc_id for d in documents])
        assert set(visibility) == {d.doc_id for d in documents}
        assert all(0.0 <= v <= 1.0 for v in visibility.values())

    def test_some_documents_are_cited(self, harness):
        documents, _, _, engine, queries = harness
        visibility = measure_visibility(engine, queries, [d.doc_id for d in documents])
        assert sum(1 for v in visibility.values() if v > 0) >= 2

    def test_no_queries_gives_all_zeros(self, harness):
        documents, _, _, engine, _ = harness
        visibility = measure_visibility(engine, [], [d.doc_id for d in documents])
        assert set(visibility.values()) == {0.0}


class TestCalibrate:
    def test_produces_fitted_weights(self, harness):
        documents, chunks_by_doc, html_by_doc, engine, queries = harness
        report = calibrate_weights(documents, chunks_by_doc, engine, queries, html_by_doc)

        assert report.usable, report.summary()
        assert report.weights.fitted
        assert report.weights.n_observations == len(documents)
        assert all(w >= 0 for w in report.weights.weights.values())
        assert sum(report.weights.weights.values()) == pytest.approx(1.0)
        assert "non-negative least squares" in report.weights.provenance()

    def test_reports_r_squared_and_correlations(self, harness):
        documents, chunks_by_doc, html_by_doc, engine, queries = harness
        report = calibrate_weights(documents, chunks_by_doc, engine, queries, html_by_doc)

        assert report.weights.r_squared is not None
        assert report.correlations
        # Collinearity is reported rather than hidden; the list may be empty.
        assert isinstance(top_correlations(report), list)

    def test_refuses_to_fit_on_too_few_pages(self, harness):
        documents, chunks_by_doc, html_by_doc, engine, queries = harness
        report = calibrate_weights(
            documents[:4], chunks_by_doc, engine, queries, html_by_doc
        )
        assert not report.weights.fitted
        assert not report.usable
        assert any("needs more than" in w for w in report.warnings)

    def test_refuses_to_fit_when_nothing_was_cited(self, harness, corpus):
        documents, chunks_by_doc, html_by_doc, engine, _ = harness
        report = calibrate_weights(documents, chunks_by_doc, engine, [], html_by_doc)
        assert not report.weights.fitted
        assert any("carries no signal" in w for w in report.warnings)

    def test_fitted_weights_change_the_score(self, harness):
        documents, chunks_by_doc, html_by_doc, engine, queries = harness
        report = calibrate_weights(documents, chunks_by_doc, engine, queries, html_by_doc)

        from geolytics.geo.signals import compute_signals

        signals = compute_signals(
            documents[0],
            chunks=chunks_by_doc[documents[0].doc_id],
            html=html_by_doc[documents[0].doc_id],
        )
        unfitted = GEOScorer().score(signals)
        fitted = GEOScorer(weights=report.weights).score(signals)

        assert unfitted.score != pytest.approx(fitted.score)
        assert not unfitted.weights.fitted
        assert fitted.weights.fitted

    def test_serialises_for_the_report(self, harness):
        documents, chunks_by_doc, html_by_doc, engine, queries = harness
        report = calibrate_weights(documents, chunks_by_doc, engine, queries, html_by_doc)
        payload = as_dict(report)

        assert json.loads(json.dumps(payload))["fitted"] is True
        assert set(payload) >= {"weights", "r_squared", "provenance", "visibility_by_doc"}
