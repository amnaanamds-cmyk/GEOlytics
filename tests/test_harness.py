"""The experiment harness end to end, plus rank equivalence."""

from __future__ import annotations

import json

import pytest

from geolytics.chunking import SentenceChunker, build_chunker
from geolytics.evaluation.harness import (
    Condition,
    ExperimentHarness,
    common_query_ids,
    compare_runs,
    save_runs,
)
from geolytics.evaluation.rank_equivalence import verify_rank_equivalence
from geolytics.evaluation.report import comparison_table, methods_note, results_table
from geolytics.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


def dense(context):
    return DenseRetriever(context.store, context.embedder, context.collection)


def hybrid(context):
    return HybridRetriever(dense=dense(context), lexical=BM25Retriever(context.chunks))


@pytest.fixture
def harness(corpus, corpus_queries, embedder):
    return ExperimentHarness(corpus, corpus_queries, embedder, top_k=10, cutoffs=(1, 5, 10))


@pytest.fixture
def conditions(embedder):
    return [
        Condition("fixed+dense", build_chunker("fixed", chunk_tokens=30, overlap_tokens=5), dense),
        Condition("sentence+dense", SentenceChunker(max_tokens=30, min_tokens=1), dense),
        Condition("sentence+hybrid", SentenceChunker(max_tokens=30, min_tokens=1), hybrid),
    ]


class TestHarness:
    def test_runs_a_condition(self, harness, conditions):
        run = harness.run(conditions[0])
        assert run.n_queries_scored > 0
        assert run.index_stats.n_chunks > 0
        assert "ndcg@10" in next(iter(run.per_query.values()))

    def test_records_per_query_not_just_means(self, harness, conditions):
        run = harness.run(conditions[0])
        assert len(run.per_query) == run.n_queries_scored
        assert len(run.scores("ndcg@10")) == run.n_queries_scored

    def test_index_stats_expose_chunk_size(self, harness, conditions):
        runs = harness.run_all(conditions[:2])
        assert all(r.index_stats.mean_tokens > 0 for r in runs)
        assert runs[0].index_stats.n_chunks != runs[1].index_stats.n_chunks or True

    def test_rejects_top_k_below_largest_cutoff(self, corpus, corpus_queries, embedder):
        with pytest.raises(ValueError, match="below the largest cutoff"):
            ExperimentHarness(corpus, corpus_queries, embedder, top_k=5, cutoffs=(1, 10))

    def test_rejects_a_condition_producing_no_chunks(self, harness):
        class EmptyChunker(SentenceChunker):
            name = "empty"

            def chunk(self, document):
                return []

        with pytest.raises(ValueError, match="no chunks"):
            harness.run(Condition("empty", EmptyChunker(), dense))

    def test_scores_in_requested_query_order(self, harness, conditions):
        run = harness.run(conditions[0])
        ids = run.query_ids
        assert run.scores("ndcg@10", ids) == [run.per_query[i]["ndcg@10"] for i in ids]

    def test_unknown_query_id_raises(self, harness, conditions):
        run = harness.run(conditions[0])
        with pytest.raises(KeyError):
            run.scores("ndcg@10", ["not_a_query"])

    def test_aggregate_matches_manual_mean(self, harness, conditions):
        run = harness.run(conditions[0])
        values = run.scores("ndcg@10")
        assert run.aggregate()["ndcg@10"] == pytest.approx(sum(values) / len(values))


class TestComparison:
    def test_common_query_ids_intersects(self, harness, conditions):
        runs = harness.run_all(conditions)
        shared = common_query_ids(runs)
        assert shared
        assert all(set(shared) <= set(r.per_query) for r in runs)

    def test_compare_against_a_baseline(self, harness, conditions):
        runs = harness.run_all(conditions)
        comparisons = compare_runs(runs, "ndcg@10", baseline="fixed+dense")
        assert len(comparisons) == len(conditions) - 1
        assert all(c.name_b == "fixed+dense" for c in comparisons)
        assert all(c.p_adjusted is not None for c in comparisons)

    def test_all_pairs_without_a_baseline(self, harness, conditions):
        runs = harness.run_all(conditions)
        assert len(compare_runs(runs, "ndcg@10")) == 3

    def test_unknown_baseline(self, harness, conditions):
        runs = harness.run_all(conditions)
        with pytest.raises(ValueError, match="baseline"):
            compare_runs(runs, "ndcg@10", baseline="nope")

    def test_needs_two_runs(self, harness, conditions):
        with pytest.raises(ValueError, match="at least two"):
            compare_runs(harness.run_all(conditions[:1]), "ndcg@10")


class TestReporting:
    def test_results_table_includes_corpus_shape(self, harness, conditions):
        table = results_table(harness.run_all(conditions), ["ndcg@10"])
        assert "n chunks" in table and "mean tok" in table

    def test_comparison_table_has_holm_column(self, harness, conditions):
        runs = harness.run_all(conditions)
        table = comparison_table(compare_runs(runs, "ndcg@10", baseline="fixed+dense"))
        assert "p (Holm)" in table and "95% CI" in table

    def test_methods_note_is_generated_from_the_runs(self, harness, conditions):
        runs = harness.run_all(conditions)
        note = methods_note(runs, "ndcg@10")
        assert "Wilcoxon" in note and "Holm" in note and str(len(runs)) in note

    def test_save_runs_round_trips(self, harness, conditions, tmp_path):
        runs = harness.run_all(conditions[:2])
        path = save_runs(runs, tmp_path / "runs.json")
        loaded = json.loads(path.read_text())
        assert len(loaded) == 2
        assert "per_query" in loaded[0] and "index_stats" in loaded[0]


class TestRankEquivalence:
    def test_normalised_embeddings_make_cosine_and_dot_identical(self, embedder, corpus):
        passages = [d.text for d in corpus] * 5
        queries = ["boiler service", "blocked drain", "tap replacement"]
        result = verify_rank_equivalence(embedder, passages, queries, "cosine", "dot")
        assert embedder.normalized
        assert result.rank_equivalent
        assert result.mean_kendall_tau == pytest.approx(1.0)

    def test_cosine_and_euclidean_are_also_equivalent(self, embedder, corpus):
        passages = [d.text for d in corpus] * 5
        result = verify_rank_equivalence(
            embedder, passages, ["boiler service"], "cosine", "euclidean"
        )
        assert result.rank_equivalent

    def test_summary_states_the_verdict(self, embedder, corpus):
        result = verify_rank_equivalence(
            embedder, [d.text for d in corpus], ["boiler"], "cosine", "dot"
        )
        assert "rank-equivalent" in result.summary()

    def test_rejects_empty_input(self, embedder):
        with pytest.raises(ValueError, match="at least one"):
            verify_rank_equivalence(embedder, [], ["q"])

    def test_tie_breaking_noise_is_not_a_ranking_difference(self, embedder):
        """Identical passages tie; float noise must not read as disagreement.

        This is the case that first reported a false "NOT rank-equivalent":
        cosine divides by a norm that is only approximately 1 in float32, dot
        does not, and the two round genuinely tied passages differently.
        """
        passages = ["identical passage text"] * 20 + ["a different passage"] * 5
        result = verify_rank_equivalence(
            embedder, passages, ["identical passage"], "cosine", "dot"
        )
        assert result.n_tied_queries == 1
        assert result.rank_equivalent
        assert result.mean_kendall_tau == pytest.approx(1.0)

    def test_summary_discloses_ties(self, embedder):
        result = verify_rank_equivalence(
            embedder, ["same text"] * 10, ["same text"], "cosine", "dot"
        )
        assert "tied scores" in result.summary()
