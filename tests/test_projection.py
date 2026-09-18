"""Projecting document-level gold spans onto a specific chunking."""

from __future__ import annotations

import pytest

from geolytics.chunking import Document, FixedSizeChunker, SentenceChunker
from geolytics.evaluation.projection import project_query, project_query_set
from geolytics.evaluation.qrels import Query, QuerySet, SpanRelevance

TEXT = (
    "Acme clears blocked drains with a motorised auger. "
    "Acme services boilers including the heat exchanger. "
    "Acme detects leaks acoustically before opening a wall. "
    "Acme replaces taps in under one hour."
)
GOLD = "Acme detects leaks acoustically before opening a wall."


@pytest.fixture
def document() -> Document:
    return Document(doc_id="d1", url="https://acme.example/", title="Acme", text=TEXT)


@pytest.fixture
def query(document) -> Query:
    start = TEXT.index(GOLD)
    return Query(
        query_id="q1",
        text="how are leaks found",
        spans=(
            SpanRelevance(
                doc_id="d1", start_char=start, end_char=start + len(GOLD), text=GOLD
            ),
        ),
    )


class TestProjection:
    def test_same_gold_projects_onto_different_chunkers(self, document, query):
        for chunker in (
            SentenceChunker(max_tokens=12, overlap_sentences=0, min_tokens=1),
            FixedSizeChunker(chunk_tokens=12, overlap_tokens=0),
        ):
            projected = project_query(query, chunker.chunk(document))
            assert projected.n_relevant > 0, f"{chunker} found no relevant chunk"

    def test_matched_chunk_actually_contains_the_answer(self, document, query):
        chunks = SentenceChunker(max_tokens=12, overlap_sentences=0, min_tokens=1).chunk(document)
        projected = project_query(query, chunks)
        by_id = {c.chunk_id: c for c in chunks}
        assert any("leaks" in by_id[cid].text for cid in projected.relevant_ids)

    def test_unrelated_chunks_are_not_relevant(self, document, query):
        chunks = SentenceChunker(max_tokens=12, overlap_sentences=0, min_tokens=1).chunk(document)
        projected = project_query(query, chunks)
        by_id = {c.chunk_id: c for c in chunks}
        for cid in projected.relevant_ids:
            assert "auger" not in by_id[cid].text or "leaks" in by_id[cid].text

    def test_text_fallback_when_offsets_are_wrong(self, document):
        """A span with bogus offsets still matches via normalised text."""
        query = Query(
            query_id="q2",
            text="how are leaks found",
            spans=(
                SpanRelevance(
                    doc_id="d1", start_char=99999, end_char=99999, text=GOLD
                ),
            ),
        )
        chunks = SentenceChunker(max_tokens=12, overlap_sentences=0, min_tokens=1).chunk(document)
        assert project_query(query, chunks).n_relevant > 0

    def test_span_on_another_document_does_not_match(self, document, query):
        other = Document(doc_id="d2", url="u", title="t", text=TEXT)
        chunks = SentenceChunker(max_tokens=12, min_tokens=1).chunk(other)
        assert project_query(query, chunks).n_relevant == 0

    def test_higher_coverage_threshold_is_stricter(self, document):
        start = TEXT.index(GOLD)
        query = Query(
            query_id="q3",
            text="leaks",
            spans=(
                SpanRelevance(doc_id="d1", start_char=start, end_char=start + len(GOLD), text=GOLD),
            ),
        )
        chunks = FixedSizeChunker(chunk_tokens=5, overlap_tokens=0).chunk(document)
        loose = project_query(query, chunks, min_coverage=0.2).n_relevant
        strict = project_query(query, chunks, min_coverage=0.95).n_relevant
        assert loose >= strict

    def test_rejects_invalid_coverage(self, document, query):
        with pytest.raises(ValueError, match="min_coverage"):
            project_query(query, [], min_coverage=0.0)

    def test_grades_survive_projection(self, document):
        start = TEXT.index(GOLD)
        query = Query(
            query_id="q4",
            text="leaks",
            spans=(
                SpanRelevance(
                    doc_id="d1", start_char=start, end_char=start + len(GOLD), grade=2, text=GOLD
                ),
            ),
        )
        chunks = SentenceChunker(max_tokens=12, min_tokens=1).chunk(document)
        projected = project_query(query, chunks)
        assert set(projected.grades.values()) == {2}


class TestProjectQuerySet:
    def test_reports_unmatched_queries(self, document, query):
        unmatched = Query(
            query_id="q_missing",
            text="something else",
            spans=(SpanRelevance(doc_id="d1", start_char=0, end_char=0, text="not present here"),),
        )
        query_set = QuerySet(name="s", queries=[query, unmatched])
        chunks = SentenceChunker(max_tokens=12, min_tokens=1).chunk(document)
        projected, stats = project_query_set(query_set, chunks)
        assert stats.n_queries == 2
        assert stats.n_unmatched == 1
        assert stats.match_rate == pytest.approx(0.5)

    def test_judged_drops_unmatched(self, document, query):
        unmatched = Query(
            query_id="q_missing",
            text="x",
            spans=(SpanRelevance(doc_id="d1", start_char=0, end_char=0, text="absent text"),),
        )
        chunks = SentenceChunker(max_tokens=12, min_tokens=1).chunk(document)
        projected, _ = project_query_set(QuerySet("s", [query, unmatched]), chunks)
        judged = projected.judged()
        assert len(judged) == 1
        assert judged.metadata["dropped_unjudged"] == 1
