"""GEO scoring: combine signals into a 0-100 score with *fitted* weights.

The weakest move available here is to invent weights -- "schema markup is 15%
of the score" -- and defend them with intuition. This module does the opposite:
the default weights are uniform and explicitly labelled as un-fitted, and
`fit_weights` derives them by regressing the signals against visibility
actually measured in the simulation.

That turns "why is schema weighted 15%?" from an opinion into a regression
coefficient with a confidence interval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import optimize

from geolytics.geo.signals import SIGNAL_NAMES, SignalReport


@dataclass(frozen=True, slots=True)
class SignalWeights:
    """Per-signal weights, plus how they were obtained."""

    weights: dict[str, float]
    fitted: bool = False
    n_observations: int = 0
    r_squared: float | None = None
    target: str | None = None

    @classmethod
    def uniform(cls, names: Sequence[str] = SIGNAL_NAMES) -> SignalWeights:
        w = 1.0 / len(names)
        return cls(weights=dict.fromkeys(names, w), fitted=False)

    def normalized(self) -> SignalWeights:
        total = sum(self.weights.values())
        if total <= 0:
            return SignalWeights.uniform(tuple(self.weights))
        from dataclasses import replace

        return replace(self, weights={k: v / total for k, v in self.weights.items()})

    def provenance(self) -> str:
        if not self.fitted:
            return (
                "UNFITTED uniform weights -- placeholder only. Report no weighted "
                "GEO score until fit_weights() has been run against measured visibility."
            )
        r2 = "n/a" if self.r_squared is None else f"{self.r_squared:.3f}"
        return (
            f"fitted by non-negative least squares against {self.target!r} "
            f"on n={self.n_observations} pages (R^2 = {r2})"
        )


@dataclass(frozen=True, slots=True)
class GEOScore:
    """A page's GEO score and the breakdown behind it."""

    doc_id: str
    url: str
    score: float
    contributions: dict[str, float]
    signals: dict[str, float]
    weights: SignalWeights
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "score": self.score,
            "contributions": self.contributions,
            "signals": self.signals,
            "weights_fitted": self.weights.fitted,
            "weights_provenance": self.weights.provenance(),
            "recommendations": self.recommendations,
        }


# What to tell a site owner when a signal scores low. Phrased as actions on the
# page, never as claims about what a specific engine will then do.
_ADVICE: dict[str, str] = {
    "statistic_density": (
        "Add concrete figures (prices, timelines, measured outcomes) instead of "
        "vague qualifiers -- quantified claims give a generative answer something "
        "specific to quote."
    ),
    "citation_density": (
        "Cite and link the sources behind your claims; attributed statements are "
        "more quotable than unsourced assertions."
    ),
    "quotation_density": (
        "Include direct quotations from named people (customers, staff, experts) "
        "rather than paraphrased testimonials."
    ),
    "authority_markers": (
        "Attribute claims explicitly ('according to', 'as of <date>', named "
        "credentials) rather than stating them anonymously."
    ),
    "schema_coverage": (
        "Add JSON-LD structured data for the page's real type (Organization, "
        "LocalBusiness, Product, FAQPage) so extractors get typed fields."
    ),
    "heading_structure": (
        "Use one h1 and a proper h2/h3 hierarchy with no skipped levels -- "
        "extractors segment on heading levels."
    ),
    "self_containedness": (
        "Start sections and paragraphs with the named subject, not a pronoun. A "
        "passage retrieved on its own must still say what it is about."
    ),
    "extractability": (
        "Reduce boilerplate relative to content: navigation, banners and footers "
        "dilute the page's extracted text."
    ),
    "answer_directness": (
        "Answer the question in the first sentence of a section, then elaborate, "
        "rather than building up to it."
    ),
    "freshness": (
        "Publish and expose a visible last-updated date in the page and in "
        "structured data."
    ),
}


class GEOScorer:
    """Turns a `SignalReport` into a 0-100 score plus targeted recommendations."""

    def __init__(
        self,
        weights: SignalWeights | None = None,
        names: Sequence[str] = SIGNAL_NAMES,
        advice_threshold: float = 0.5,
    ) -> None:
        self.names = list(names)
        self.weights = (weights or SignalWeights.uniform(names)).normalized()
        self.advice_threshold = advice_threshold

    def score(self, report: SignalReport) -> GEOScore:
        contributions = {
            name: self.weights.weights.get(name, 0.0) * report.values.get(name, 0.0)
            for name in self.names
        }
        total = 100.0 * sum(contributions.values())

        # Recommend in order of recoverable score, not raw weakness: a weak
        # signal with a small weight is not worth the site owner's time.
        gaps = sorted(
            (
                (self.weights.weights.get(n, 0.0) * (1.0 - report.values.get(n, 0.0)), n)
                for n in self.names
                if report.values.get(n, 0.0) < self.advice_threshold
            ),
            reverse=True,
        )
        recommendations = [_ADVICE[n] for _, n in gaps if n in _ADVICE]

        return GEOScore(
            doc_id=report.doc_id,
            url=report.url,
            score=round(total, 2),
            contributions={k: round(v * 100.0, 3) for k, v in contributions.items()},
            signals=dict(report.values),
            weights=self.weights,
            recommendations=recommendations,
        )


def fit_weights(
    reports: Sequence[SignalReport],
    observed: Sequence[float],
    names: Sequence[str] = SIGNAL_NAMES,
    target: str = "position_adjusted_share",
) -> SignalWeights:
    """Fit signal weights against measured visibility.

    `observed` is one visibility value per report -- typically
    `ImpressionMetrics.position_adjusted_share` averaged over that page's
    queries in the simulated engine.

    Non-negative least squares rather than ordinary least squares: a negative
    coefficient would mean "adding citations makes you less visible", which is
    not a claim this data can support and which makes the resulting score
    impossible to explain to a site owner. Constraining to non-negative keeps
    every weight interpretable.

    Caveats to state wherever the fitted weights appear:

    * These are *associations within this corpus*, not causal effects. The
      causal version is an intervention study: rewrite pages to raise one
      signal, hold the rest fixed, re-measure.
    * The signals are correlated with each other (a well-maintained page tends
      to score well on several at once), so individual coefficients are not
      cleanly separable. Report the R^2 and the correlation matrix.
    * With fewer observations than signals the fit is underdetermined. The
      function refuses that case rather than returning a meaningless vector.
    """
    if len(reports) != len(observed):
        raise ValueError("need exactly one observed visibility value per report")
    if len(reports) <= len(names):
        raise ValueError(
            f"fitting {len(names)} weights needs more than {len(names)} pages; "
            f"got {len(reports)}. Crawl more pages or reduce the signal set."
        )

    x = np.array([r.vector(names) for r in reports], dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)

    coefficients, _ = optimize.nnls(x, y)
    residual = y - x @ coefficients
    ss_res = float(residual @ residual)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    return SignalWeights(
        weights={name: float(c) for name, c in zip(names, coefficients, strict=True)},
        fitted=True,
        n_observations=len(reports),
        r_squared=r_squared,
        target=target,
    ).normalized()


def signal_correlations(
    reports: Sequence[SignalReport], names: Sequence[str] = SIGNAL_NAMES
) -> dict[str, dict[str, float]]:
    """Pearson correlation between signals -- the collinearity check for the fit."""
    x = np.array([r.vector(names) for r in reports], dtype=np.float64)
    if x.shape[0] < 2:
        raise ValueError("need at least two reports to correlate signals")
    # A constant column has zero variance; np.corrcoef yields NaN there, which
    # is the honest answer -- surface it rather than substituting 0.
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.corrcoef(x, rowvar=False)
    return {
        a: {b: float(matrix[i, j]) for j, b in enumerate(names)}
        for i, a in enumerate(names)
    }


def as_mapping(reports: Sequence[SignalReport]) -> Mapping[str, SignalReport]:
    return {r.doc_id: r for r in reports}
