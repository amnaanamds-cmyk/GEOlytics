"""GEO signals, visibility metrics, and scoring."""

from __future__ import annotations

import pytest

from conftest import SAMPLE_HTML
from geolytics.chunking import Document, SentenceChunker
from geolytics.crawl.extract import extract_document
from geolytics.geo.engine import parse_citations
from geolytics.geo.scoring import (
    GEOScorer,
    SignalWeights,
    fit_weights,
    signal_correlations,
)
from geolytics.geo.signals import SIGNAL_NAMES, compute_signals
from geolytics.geo.visibility import AnswerSentence, citation_visibility
from geolytics.index.base import ScoredChunk


@pytest.fixture
def scored_page():
    document = extract_document(SAMPLE_HTML, "https://acme.example/")
    chunks = SentenceChunker(max_tokens=40, min_tokens=1).chunk(document)
    return compute_signals(document, chunks=chunks, html=SAMPLE_HTML)


class TestSignals:
    def test_all_signals_present_and_bounded(self, scored_page):
        assert set(scored_page.values) == set(SIGNAL_NAMES)
        assert all(0.0 <= v <= 1.0 for v in scored_page.values.values())

    def test_detects_schema_markup(self, scored_page):
        assert scored_page.values["schema_coverage"] > 0

    def test_absent_schema_scores_zero(self):
        document = Document(doc_id="d", url="u", title="t", text="No markup at all here.")
        report = compute_signals(document, chunks=[], html="<html><p>No markup</p></html>")
        assert report.values["schema_coverage"] == 0.0

    def test_missing_html_warns_rather_than_guessing(self):
        document = Document(doc_id="d", url="u", title="t", text="Some text.")
        report = compute_signals(document)
        assert report.values["schema_coverage"] == 0.0
        assert any("no HTML" in w for w in report.warnings)
        assert any("no chunks" in w for w in report.warnings)

    def test_statistics_are_counted(self, scored_page):
        assert scored_page.raw["n_qualified_statistics"] > 0
        assert scored_page.values["statistic_density"] > 0

    def test_heading_hierarchy_rewards_a_single_h1(self):
        good = compute_signals(
            Document(doc_id="d", url="u", title="t", text="x"),
            html="<h1>A</h1><h2>B</h2><h2>C</h2>",
        )
        bad = compute_signals(
            Document(doc_id="d", url="u", title="t", text="x"),
            html="<h1>A</h1><h1>B</h1><h1>C</h1>",
        )
        assert good.values["heading_structure"] > bad.values["heading_structure"]

    def test_heading_hierarchy_penalises_skipped_levels(self):
        smooth = compute_signals(
            Document(doc_id="d", url="u", title="t", text="x"), html="<h1>A</h1><h2>B</h2>"
        )
        skipped = compute_signals(
            Document(doc_id="d", url="u", title="t", text="x"),
            html="<h1>A</h1><h4>B</h4><h2>C</h2><h6>D</h6>",
        )
        assert smooth.values["heading_structure"] > skipped.values["heading_structure"]

    def test_self_containedness_penalises_dangling_pronouns(self):
        from geolytics.chunking.base import Chunk

        def chunk(text: str, i: int) -> Chunk:
            return Chunk(chunk_id=f"c{i}", doc_id="d", text=text, ordinal=i,
                         start_char=0, end_char=0)

        document = Document(doc_id="d", url="u", title="t", text="x")
        good = compute_signals(
            document, chunks=[chunk("Acme replaces taps quickly.", 0)], html="<p>x</p>"
        )
        bad = compute_signals(
            document, chunks=[chunk("It also includes labour.", 0)], html="<p>x</p>"
        )
        assert good.values["self_containedness"] == 1.0
        assert bad.values["self_containedness"] == 0.0

    def test_extractability_falls_with_boilerplate(self):
        document = Document(doc_id="d", url="u", title="t", text="Real content only.")
        clean = compute_signals(document, html="<p>Real content only.</p>")
        noisy = compute_signals(
            document, html="<nav>" + "link " * 500 + "</nav><p>Real content only.</p>"
        )
        assert clean.values["extractability"] > noisy.values["extractability"]

    def test_freshness_is_graded(self):
        document = Document(doc_id="d", url="u", title="t", text="Updated in 2024.")
        dated = Document(
            doc_id="d", url="u", title="t", text="Text.", metadata={"modified_at": "2024-01-01"}
        )
        undated = Document(doc_id="d", url="u", title="t", text="No date here.")
        assert compute_signals(dated).values["freshness"] == 1.0
        assert compute_signals(document).values["freshness"] == 0.5
        assert compute_signals(undated).values["freshness"] == 0.0

    def test_vector_order_matches_signal_names(self, scored_page):
        vector = scored_page.vector()
        assert vector == [scored_page.values[n] for n in SIGNAL_NAMES]


class TestScoring:
    def test_uniform_weights_are_flagged_unfitted(self, scored_page):
        score = GEOScorer().score(scored_page)
        assert not score.weights.fitted
        assert "UNFITTED" in score.weights.provenance()

    def test_score_is_bounded(self, scored_page):
        assert 0.0 <= GEOScorer().score(scored_page).score <= 100.0

    def test_recommendations_target_weak_signals(self):
        document = Document(doc_id="d", url="u", title="t", text="We are the best. Trust us.")
        report = compute_signals(document, chunks=[], html="<p>We are the best.</p>")
        score = GEOScorer().score(report)
        assert score.recommendations
        assert any("structured data" in r or "figures" in r for r in score.recommendations)

    def test_strong_page_earns_fewer_recommendations(self, scored_page):
        weak = compute_signals(
            Document(doc_id="d", url="u", title="t", text="Best service ever."),
            chunks=[],
            html="<p>Best service ever.</p>",
        )
        assert len(GEOScorer().score(scored_page).recommendations) < len(
            GEOScorer().score(weak).recommendations
        )

    def test_contributions_sum_to_the_score(self, scored_page):
        score = GEOScorer().score(scored_page)
        assert sum(score.contributions.values()) == pytest.approx(score.score, abs=0.05)

    def test_weights_are_normalised(self):
        weights = SignalWeights(weights={"a": 2.0, "b": 2.0}).normalized()
        assert sum(weights.weights.values()) == pytest.approx(1.0)


class TestFitWeights:
    def _reports(self, n: int):
        from geolytics.geo.signals import SignalReport

        reports = []
        for i in range(n):
            values = {name: ((i * (j + 3)) % 10) / 10.0 for j, name in enumerate(SIGNAL_NAMES)}
            reports.append(SignalReport(doc_id=f"d{i}", url=f"u{i}", values=values))
        return reports

    def test_fits_non_negative_weights(self):
        reports = self._reports(40)
        observed = [r.values["statistic_density"] * 0.8 for r in reports]
        weights = fit_weights(reports, observed)
        assert weights.fitted
        assert all(w >= 0 for w in weights.weights.values())

    def test_records_provenance(self):
        reports = self._reports(40)
        weights = fit_weights(reports, [r.values["schema_coverage"] for r in reports])
        assert "fitted by non-negative least squares" in weights.provenance()
        assert weights.n_observations == 40

    def test_refuses_underdetermined_fit(self):
        reports = self._reports(5)
        with pytest.raises(ValueError, match="needs more than"):
            fit_weights(reports, [0.1] * 5)

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="one observed"):
            fit_weights(self._reports(40), [0.1] * 3)

    def test_correlations_are_square_and_symmetric(self):
        matrix = signal_correlations(self._reports(20))
        assert set(matrix) == set(SIGNAL_NAMES)
        a, b = SIGNAL_NAMES[0], SIGNAL_NAMES[1]
        assert matrix[a][b] == pytest.approx(matrix[b][a])


class TestVisibility:
    def _answer(self):
        return [
            AnswerSentence("Acme dispatches within 90 minutes.", ("target_1",)),
            AnswerSentence("Other firms vary widely.", ("other_1",)),
            AnswerSentence("Callouts cost 2,500 PKR.", ("target_2",)),
        ]

    def test_word_share_counts_only_cited_sentences(self):
        metrics = citation_visibility(self._answer(), {"target_1", "target_2"})
        assert 0.0 < metrics.word_count_share < 1.0
        assert metrics.cited
        assert metrics.first_cited_position == 1

    def test_position_weighting_favours_early_citations(self):
        early = citation_visibility(self._answer(), {"target_1"})
        late = citation_visibility(self._answer(), {"target_2"})
        assert early.position_adjusted_share > late.position_adjusted_share

    def test_no_decay_reduces_to_word_share(self):
        metrics = citation_visibility(self._answer(), {"target_1"}, decay="none")
        assert metrics.position_adjusted_share == pytest.approx(metrics.word_count_share)

    def test_uncited_target(self):
        metrics = citation_visibility(self._answer(), {"absent"})
        assert not metrics.cited
        assert metrics.word_count_share == 0.0
        assert "not cited" in metrics.summary()

    def test_empty_answer(self):
        metrics = citation_visibility([], {"x"})
        assert metrics.n_sentences == 0
        assert "empty answer" in metrics.notes

    def test_citation_count_share(self):
        metrics = citation_visibility(self._answer(), {"target_1", "target_2"})
        assert metrics.citation_count_share == pytest.approx(2 / 3)

    def test_rejects_invalid_half_life(self):
        with pytest.raises(ValueError, match="half_life"):
            citation_visibility(self._answer(), {"target_1"}, half_life=0.0)


class TestCitationParsing:
    def _retrieved(self, n: int):
        from geolytics.chunking.base import Chunk

        return [
            ScoredChunk(
                chunk=Chunk(chunk_id=f"c{i}", doc_id="d", text=f"t{i}", ordinal=i,
                            start_char=0, end_char=0),
                score=1.0,
                rank=i + 1,
            )
            for i in range(n)
        ]

    def test_resolves_markers_to_chunk_ids(self):
        sentences = parse_citations("Acme is fast [1]. Costs vary [2][3].", self._retrieved(3))
        assert sentences[0].cited_chunk_ids == ("c0",)
        assert sentences[1].cited_chunk_ids == ("c1", "c2")

    def test_drops_hallucinated_markers(self):
        sentences = parse_citations("Acme is fast [7].", self._retrieved(3))
        assert sentences[0].cited_chunk_ids == ()

    def test_deduplicates_repeated_markers(self):
        sentences = parse_citations("Acme is fast [1][1].", self._retrieved(3))
        assert sentences[0].cited_chunk_ids == ("c0",)

    def test_uncited_sentence(self):
        sentences = parse_citations("This has no citation.", self._retrieved(3))
        assert sentences[0].cited_chunk_ids == ()
