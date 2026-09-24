"""Fit GEO signal weights against visibility measured in the simulated engine.

This closes the loop the scoring module opens. `geo.scoring.fit_weights` can
turn observations into weights, but something has to *produce* the
observations: run the simulated generative engine over an evaluation query
set, measure how much of each answer came from each page, and use that as the
regression target.

The resulting weights are associations within one corpus, not causal effects.
Two pages that both score well on several signals at once cannot be separated
by this fit -- which is why `report` carries R^2 and the signal correlation
matrix rather than only the coefficients.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from geolytics.chunking.base import Chunk, Document
from geolytics.geo.engine import SimulatedGenerativeEngine
from geolytics.geo.scoring import SignalWeights, fit_weights, signal_correlations
from geolytics.geo.signals import SIGNAL_NAMES, SignalReport, compute_signals
from geolytics.geo.visibility import Decay


@dataclass(slots=True)
class CalibrationReport:
    """Everything needed to defend a fitted weight vector."""

    weights: SignalWeights
    n_pages: int
    n_queries: int
    n_pages_cited: int
    mean_visibility: float
    visibility_by_doc: dict[str, float]
    correlations: dict[str, dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Whether the fit is worth reporting at all."""
        return self.weights.fitted and self.n_pages_cited >= 2

    def summary(self) -> str:
        lines = [
            f"Fitted on {self.n_pages} pages over {self.n_queries} queries; "
            f"{self.n_pages_cited} pages were cited at least once "
            f"(mean visibility {self.mean_visibility:.3f}).",
            self.weights.provenance(),
        ]
        ranked = sorted(self.weights.weights.items(), key=lambda kv: -kv[1])
        lines.append("Weights: " + ", ".join(f"{n}={w:.3f}" for n, w in ranked if w > 0))
        lines.extend(f"WARNING: {w}" for w in self.warnings)
        return "\n".join(lines)


def measure_visibility(
    engine: SimulatedGenerativeEngine,
    queries: Sequence[str],
    doc_ids: Sequence[str],
    decay: Decay = "exponential",
) -> dict[str, float]:
    """Mean position-adjusted share of each document across all queries.

    Averaged over *every* query, not only the ones where the page was cited:
    a page that wins one answer and is absent from forty others is not a
    visible page, and averaging over citations alone would hide that.
    """
    totals = dict.fromkeys(doc_ids, 0.0)
    if not queries:
        return totals

    for query in queries:
        answer = engine.answer(query)
        if not answer.sentences:
            continue
        cited_by_doc: dict[str, set[str]] = {}
        for hit in answer.retrieved:
            cited_by_doc.setdefault(hit.chunk.doc_id, set()).add(hit.chunk_id)
        for doc_id, chunk_ids in cited_by_doc.items():
            if doc_id in totals:
                totals[doc_id] += answer.visibility_for(chunk_ids).position_adjusted_share

    return {doc_id: total / len(queries) for doc_id, total in totals.items()}


def calibrate_weights(
    documents: Sequence[Document],
    chunks_by_doc: Mapping[str, Sequence[Chunk]],
    engine: SimulatedGenerativeEngine,
    queries: Sequence[str],
    html_by_doc: Mapping[str, str] | None = None,
    names: Sequence[str] = SIGNAL_NAMES,
) -> CalibrationReport:
    """Measure visibility, then fit signal weights against it."""
    html_by_doc = html_by_doc or {}
    reports: list[SignalReport] = [
        compute_signals(
            document,
            chunks=chunks_by_doc.get(document.doc_id, []),
            html=html_by_doc.get(document.doc_id),
        )
        for document in documents
    ]

    visibility = measure_visibility(engine, queries, [d.doc_id for d in documents])
    observed = [visibility[d.doc_id] for d in documents]
    cited = sum(1 for v in observed if v > 0)
    warnings: list[str] = []

    if len(documents) <= len(names):
        warnings.append(
            f"fitting {len(names)} weights needs more than {len(names)} pages; "
            f"only {len(documents)} were supplied, so weights were left unfitted"
        )
        weights = SignalWeights.uniform(names)
    elif cited < 2:
        # With one or zero cited pages the target is almost constant and NNLS
        # will happily return a vector that explains nothing.
        warnings.append(
            f"only {cited} page(s) were cited by the simulated engine; "
            "the regression target carries no signal, so weights were left unfitted"
        )
        weights = SignalWeights.uniform(names)
    else:
        weights = fit_weights(reports, observed, names=names)

    correlations: dict[str, dict[str, float]] = {}
    if len(reports) >= 2:
        correlations = signal_correlations(reports, names=names)

    return CalibrationReport(
        weights=weights,
        n_pages=len(documents),
        n_queries=len(queries),
        n_pages_cited=cited,
        mean_visibility=(sum(observed) / len(observed)) if observed else 0.0,
        visibility_by_doc=visibility,
        correlations=correlations,
        warnings=warnings,
    )


def top_correlations(
    report: CalibrationReport, threshold: float = 0.8
) -> list[tuple[str, str, float]]:
    """Signal pairs collinear enough to make their coefficients unreliable."""
    out: list[tuple[str, str, float]] = []
    names = list(report.correlations)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            value = report.correlations[a].get(b)
            if value is not None and abs(value) >= threshold:
                out.append((a, b, value))
    return sorted(out, key=lambda t: -abs(t[2]))


def as_dict(report: CalibrationReport) -> dict[str, Any]:
    return {
        "weights": report.weights.weights,
        "fitted": report.weights.fitted,
        "r_squared": report.weights.r_squared,
        "provenance": report.weights.provenance(),
        "n_pages": report.n_pages,
        "n_queries": report.n_queries,
        "n_pages_cited": report.n_pages_cited,
        "mean_visibility": report.mean_visibility,
        "visibility_by_doc": report.visibility_by_doc,
        "collinear_pairs": [
            {"a": a, "b": b, "r": r} for a, b, r in top_correlations(report)
        ],
        "warnings": report.warnings,
    }
