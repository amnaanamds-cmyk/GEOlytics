"""Chunking strategies."""

from __future__ import annotations

import pytest

from geolytics.chunking import (
    Document,
    FixedSizeChunker,
    ParentChildChunker,
    SemanticChunker,
    SentenceChunker,
    build_chunker,
)
from geolytics.chunking.base import WhitespaceTokenCounter, split_sentences


class TestSentenceSplitting:
    def test_splits_on_terminators(self):
        assert len(split_sentences("One. Two! Three?")) == 3

    def test_does_not_split_abbreviations(self):
        sentences = split_sentences("Call Dr. Khan today. He is available.")
        assert len(sentences) == 2
        assert sentences[0] == "Call Dr. Khan today."

    def test_does_not_split_initials(self):
        assert len(split_sentences("Contact J. Smith now. He replies fast.")) == 2

    def test_empty_input(self):
        assert split_sentences("   ") == []


class TestFixedSizeChunker:
    def test_respects_token_budget(self, simple_document):
        chunks = FixedSizeChunker(chunk_tokens=10, overlap_tokens=2).chunk(simple_document)
        counter = WhitespaceTokenCounter()
        assert chunks
        assert all(counter.count(c.text) <= 10 for c in chunks)

    def test_overlap_repeats_tokens(self, simple_document):
        chunks = FixedSizeChunker(chunk_tokens=10, overlap_tokens=3).chunk(simple_document)
        assert len(chunks) >= 2
        first_tail = chunks[0].text.split()[-3:]
        second_head = chunks[1].text.split()[:3]
        assert first_tail == second_head

    def test_covers_every_token(self, simple_document):
        chunks = FixedSizeChunker(chunk_tokens=8, overlap_tokens=0).chunk(simple_document)
        assert " ".join(c.text for c in chunks).split() == simple_document.text.split()

    def test_rejects_overlap_at_or_above_chunk_size(self):
        with pytest.raises(ValueError, match="overlap"):
            FixedSizeChunker(chunk_tokens=10, overlap_tokens=10)

    def test_empty_document(self):
        empty = Document(doc_id="e", url="u", title="t", text="")
        assert FixedSizeChunker().chunk(empty) == []

    def test_chunk_ids_are_unique(self, corpus):
        chunker = FixedSizeChunker(chunk_tokens=12, overlap_tokens=0)
        ids = [c.chunk_id for d in corpus for c in chunker.chunk(d)]
        assert len(ids) == len(set(ids))


class TestSentenceChunker:
    def test_never_splits_a_sentence(self, simple_document):
        chunks = SentenceChunker(max_tokens=12, overlap_sentences=0).chunk(simple_document)
        for chunk in chunks:
            for sentence in split_sentences(chunk.text):
                assert (
                    sentence in simple_document.text
                    or sentence.rstrip(".") in simple_document.text
                )

    def test_oversized_sentence_survives_intact(self):
        long_sentence = "word " * 100 + "end."
        document = Document(doc_id="d", url="u", title="t", text=long_sentence)
        chunks = SentenceChunker(max_tokens=10).chunk(document)
        assert len(chunks) == 1
        assert chunks[0].text.endswith("end.")

    def test_carries_heading_path(self, simple_document):
        chunks = SentenceChunker(max_tokens=20).chunk(simple_document)
        assert all(c.heading_path == ("Acme Plumbing",) for c in chunks)

    def test_does_not_span_sections(self, corpus):
        from geolytics.chunking.base import Section

        document = Document(
            doc_id="multi",
            url="u",
            title="t",
            text="Alpha one two. Beta three four.",
            sections=(
                Section(text="Alpha one two.", heading_path=("A",), start_char=0),
                Section(text="Beta three four.", heading_path=("B",), start_char=15),
            ),
        )
        chunks = SentenceChunker(max_tokens=100, min_tokens=1).chunk(document)
        assert {c.heading_path for c in chunks} == {("A",), ("B",)}


class TestSemanticChunker:
    def test_cuts_at_a_topic_shift(self, embedder):
        text = (
            "Boiler pressure should sit near one bar. "
            "Boiler pressure drops when a radiator is bled. "
            "Boiler pressure is restored with the filling loop. "
            "Bathroom tiles are laid on a primed substrate. "
            "Bathroom tiles need a day to set before grouting. "
            "Bathroom tiles are sealed along every wet edge."
        )
        document = Document(doc_id="d", url="u", title="t", text=text)
        chunks = SemanticChunker(embedder, breakpoint_percentile=80.0).chunk(document)
        assert len(chunks) >= 2
        joined = [c.text for c in chunks]
        assert not any("Boiler" in t and "Bathroom tiles are laid" in t for t in joined)

    def test_short_input_stays_whole(self, embedder):
        document = Document(doc_id="d", url="u", title="t", text="One. Two.")
        assert len(SemanticChunker(embedder, min_sentences=2).chunk(document)) == 1

    def test_enforces_max_tokens(self, embedder):
        text = " ".join(f"Uniform sentence number {i} about one single topic." for i in range(40))
        document = Document(doc_id="d", url="u", title="t", text=text)
        chunks = SemanticChunker(embedder, max_tokens=30).chunk(document)
        assert all(len(c.text.split()) <= 30 for c in chunks)

    def test_rejects_invalid_percentile(self, embedder):
        with pytest.raises(ValueError, match="percentile"):
            SemanticChunker(embedder, breakpoint_percentile=100.0)


class TestParentChildChunker:
    def test_children_carry_parent_text(self, simple_document):
        chunker = ParentChildChunker(
            parent=SentenceChunker(max_tokens=100, min_tokens=1),
            child=SentenceChunker(max_tokens=10, overlap_sentences=0, min_tokens=1),
        )
        chunks = chunker.chunk(simple_document)
        assert chunks
        assert all(c.parent_id is not None for c in chunks)
        assert all(c.parent_text is not None for c in chunks)

    def test_context_text_returns_parent(self, simple_document):
        chunker = ParentChildChunker(
            parent=SentenceChunker(max_tokens=100, min_tokens=1),
            child=SentenceChunker(max_tokens=10, overlap_sentences=0, min_tokens=1),
        )
        chunk = chunker.chunk(simple_document)[0]
        assert chunk.context_text == chunk.parent_text
        assert len(chunk.context_text) >= len(chunk.text)

    def test_child_is_shorter_than_parent(self, simple_document):
        chunker = ParentChildChunker(
            parent=SentenceChunker(max_tokens=100, min_tokens=1),
            child=SentenceChunker(max_tokens=8, overlap_sentences=0, min_tokens=1),
        )
        chunks = chunker.chunk(simple_document)
        assert any(len(c.text) < len(c.parent_text) for c in chunks)


class TestRegistry:
    def test_builds_each_strategy(self, embedder, simple_document):
        for name in ("fixed", "sentence", "parent_child"):
            assert build_chunker(name).chunk(simple_document) is not None
        assert build_chunker("semantic", embedder=embedder).chunk(simple_document) is not None

    def test_semantic_requires_an_embedder(self):
        with pytest.raises(ValueError, match="embedder"):
            build_chunker("semantic")

    def test_unknown_name(self):
        with pytest.raises(ValueError, match="unknown chunker"):
            build_chunker("nope")

    def test_nested_parent_child_spec(self, simple_document):
        chunker = build_chunker(
            "parent_child",
            parent={"name": "sentence", "max_tokens": 80},
            child={"name": "fixed", "chunk_tokens": 8, "overlap_tokens": 0},
        )
        assert chunker.chunk(simple_document)
        assert chunker.params["child"]["strategy"] == "fixed"
