"""Retrieval evaluation: judgments, projection, metrics, statistics, harness."""

from geolytics.evaluation.agreement import AgreementResult, build_pool, judgment_agreement
from geolytics.evaluation.harness import (
    Condition,
    ExperimentHarness,
    RetrievalContext,
    RunResult,
    common_query_ids,
    compare_runs,
    save_runs,
)
from geolytics.evaluation.metrics import (
    METRIC_FUNCTIONS,
    average_precision,
    evaluate_ranking,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from geolytics.evaluation.projection import project_query, project_query_set
from geolytics.evaluation.qagen import QAGenerationConfig, generate_query_set, lexical_overlap
from geolytics.evaluation.qrels import Query, QuerySet, Relevance, SpanRelevance
from geolytics.evaluation.rank_equivalence import verify_rank_equivalence
from geolytics.evaluation.report import comparison_table, methods_note, results_table
from geolytics.evaluation.stats import (
    PairedComparison,
    apply_correction,
    bootstrap_ci,
    cliffs_delta,
    compare_paired,
    holm_bonferroni,
)

__all__ = [
    "METRIC_FUNCTIONS",
    "AgreementResult",
    "Condition",
    "ExperimentHarness",
    "PairedComparison",
    "QAGenerationConfig",
    "Query",
    "QuerySet",
    "Relevance",
    "RetrievalContext",
    "RunResult",
    "SpanRelevance",
    "apply_correction",
    "average_precision",
    "bootstrap_ci",
    "build_pool",
    "cliffs_delta",
    "common_query_ids",
    "compare_paired",
    "compare_runs",
    "comparison_table",
    "evaluate_ranking",
    "generate_query_set",
    "hit_rate_at_k",
    "holm_bonferroni",
    "judgment_agreement",
    "lexical_overlap",
    "methods_note",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "project_query",
    "project_query_set",
    "recall_at_k",
    "results_table",
    "save_runs",
    "verify_rank_equivalence",
]
