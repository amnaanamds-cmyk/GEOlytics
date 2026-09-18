"""Synthetic query generation and judgment agreement."""

from __future__ import annotations

import json

import pytest

from geolytics.evaluation.agreement import build_pool, cohens_kappa, judgment_agreement
from geolytics.evaluation.qagen import (
    QAGenerationConfig,
    generate_query_set,
    lexical_overlap,
)
from geolytics.evaluation.qrels import Query, QuerySet, Relevance
from geolytics.llm import EchoLLMClient


class TestLexicalOverlap:
    def test_verbatim_lift_scores_one(self):
        overlap = lexical_overlap(
            "blocked drain auger", "we clear a blocked drain with an auger"
        )
        assert overlap == 1.0

    def test_full_paraphrase_scores_low(self):
        assert lexical_overlap("pipe obstruction removal", "we clear a blocked drain") < 0.5

    def test_empty_query(self):
        assert lexical_overlap("", "anything") == 0.0


class TestGeneration:
    def test_produces_document_anchored_spans(self, corpus):
        llm = EchoLLMClient([json.dumps(["What is covered?", "How long does it take?"])])
        query_set, report = generate_query_set(
            corpus, llm, QAGenerationConfig(questions_per_unit=2, min_unit_tokens=5)
        )
        assert len(query_set) > 0
        assert report.n_generated == len(query_set)
        for query in query_set:
            assert query.spans
            assert query.spans[0].text
            assert query.source_doc_id

    def test_records_lexical_overlap_per_query(self, corpus):
        llm = EchoLLMClient([json.dumps(["What is covered by the service?"])])
        query_set, _ = generate_query_set(
            corpus, llm, QAGenerationConfig(questions_per_unit=1, min_unit_tokens=5)
        )
        assert all("lexical_overlap" in q.metadata for q in query_set)

    def test_drops_queries_above_the_overlap_threshold(self, corpus):
        verbatim = corpus[0].text.split(". ")[0]
        llm = EchoLLMClient([json.dumps([verbatim])])
        _, report = generate_query_set(
            corpus,
            llm,
            QAGenerationConfig(
                questions_per_unit=1, min_unit_tokens=5, max_lexical_overlap=0.5
            ),
        )
        assert report.n_dropped_overlap > 0

    def test_survives_a_malformed_llm_response(self, corpus):
        llm = EchoLLMClient(["not json at all"])
        query_set, report = generate_query_set(
            corpus, llm, QAGenerationConfig(min_unit_tokens=5)
        )
        assert len(query_set) == 0
        assert report.n_failed_units == report.n_units

    def test_prompt_includes_the_paraphrase_rule(self, corpus):
        llm = EchoLLMClient([json.dumps(["Q?"])])
        generate_query_set(
            corpus, llm, QAGenerationConfig(avoid_verbatim=True, min_unit_tokens=5)
        )
        assert any("Paraphrase" in call for call in llm.calls)

    def test_generation_chunker_is_recorded(self, corpus):
        llm = EchoLLMClient([json.dumps(["Q?"])])
        query_set, _ = generate_query_set(corpus, llm, QAGenerationConfig(min_unit_tokens=5))
        assert "generation_chunker" in query_set.metadata

    def test_query_ids_are_unique(self, corpus):
        llm = EchoLLMClient([json.dumps(["What is covered?", "How long?"])])
        query_set, _ = generate_query_set(
            corpus, llm, QAGenerationConfig(questions_per_unit=2, min_unit_tokens=5)
        )
        ids = [q.query_id for q in query_set]
        assert len(ids) == len(set(ids))


class TestCohensKappa:
    def test_perfect_agreement(self):
        assert cohens_kappa([True, False, True], [True, False, True]) == pytest.approx(1.0)

    def test_chance_level_is_near_zero(self):
        a = [True, False] * 50
        b = [True, True, False, False] * 25
        assert abs(cohens_kappa(a, b)) < 0.2

    def test_total_disagreement_is_negative(self):
        assert cohens_kappa([True, False], [False, True]) < 0

    def test_both_constant(self):
        assert cohens_kappa([False] * 10, [False] * 10) == 1.0

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same items"):
            cohens_kappa([True], [True, False])


class TestJudgmentAgreement:
    def _sets(self):
        synthetic = QuerySet(
            "syn",
            [Query(query_id="q1", text="t", judgments=(Relevance("c1"), Relevance("c2")))],
        )
        human = QuerySet(
            "hum",
            [Query(query_id="q1", text="t", judgments=(Relevance("c1"), Relevance("c3")))],
        )
        return synthetic, human

    def test_computes_the_confusion_cells(self):
        synthetic, human = self._sets()
        result = judgment_agreement(synthetic, human, {"q1": ["c1", "c2", "c3", "c4"]})
        assert result.n_pairs == 4
        assert result.both_relevant == 1
        assert result.only_a == 1
        assert result.only_b == 1
        assert result.both_irrelevant == 1

    def test_interpretation_is_labelled(self):
        synthetic, human = self._sets()
        result = judgment_agreement(synthetic, human, {"q1": ["c1", "c2", "c3", "c4"]})
        assert result.interpretation in {
            "worse than chance", "slight", "fair", "moderate", "substantial", "almost perfect"
        }
        assert "kappa" in result.summary()

    def test_ignores_queries_outside_the_pool(self):
        synthetic, human = self._sets()
        assert judgment_agreement(synthetic, human, {"q_other": ["c1"]}).n_pairs == 0


class TestBuildPool:
    def test_unions_rankings_without_duplicates(self):
        pool = build_pool([{"q1": ["a", "b"]}, {"q1": ["b", "c"]}], depth=10)
        assert pool["q1"] == ["a", "b", "c"]

    def test_respects_depth(self):
        pool = build_pool([{"q1": ["a", "b", "c", "d"]}], depth=2)
        assert pool["q1"] == ["a", "b"]
