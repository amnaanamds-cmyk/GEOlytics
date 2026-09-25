"""Experiment endpoints -- the research side of the system."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from geolytics.api.deps import DbSession, require_scope
from geolytics.api.schemas import ComparisonOut, ExperimentOut, ExperimentRunOut
from geolytics.db.models import ExperimentRun, QueryRecord
from geolytics.evaluation.stats import apply_correction, compare_paired
from geolytics.tenancy.principal import Principal

router = APIRouter(prefix="/experiments", tags=["experiments"])

RequireRead = Annotated[Principal, Depends(require_scope("experiments:read"))]


@router.get("", response_model=list[str])
def list_experiments(session: DbSession, principal: RequireRead) -> list[str]:
    """Experiment names this organisation has stored."""
    return list(
        session.scalars(
            select(ExperimentRun.experiment)
            .where(ExperimentRun.org_id == principal.org_id)
            .distinct()
            .order_by(ExperimentRun.experiment)
        ).all()
    )


@router.get("/{experiment}", response_model=ExperimentOut)
def get_experiment(
    experiment: str,
    session: DbSession,
    principal: RequireRead,
    metric: str = Query(default="ndcg@10"),
    baseline: str | None = Query(default=None),
    alpha: float = Query(default=0.05, gt=0.0, lt=1.0),
) -> ExperimentOut:
    """Return each condition's means plus Holm-corrected paired comparisons."""
    runs = list(
        session.scalars(
            # Scoped in the WHERE clause: another tenant's experiment name must
            # read as absent, not merely be empty.
            select(ExperimentRun)
            .where(
                ExperimentRun.experiment == experiment,
                ExperimentRun.org_id == principal.org_id,
            )
            .order_by(ExperimentRun.condition)
        )
    )
    if not runs:
        raise HTTPException(status_code=404, detail=f"experiment {experiment!r} not found")

    scores: dict[str, dict[str, float]] = {}
    for run in runs:
        records = session.scalars(
            select(QueryRecord).where(QueryRecord.run_id == run.id)
        ).all()
        scores[run.condition] = {
            r.query_id: r.metrics[metric] for r in records if metric in r.metrics
        }

    shared = set.intersection(*(set(v) for v in scores.values())) if scores else set()
    if not shared:
        raise HTTPException(
            status_code=409,
            detail=(
                f"conditions share no queries scored on {metric!r}; "
                "cannot compute paired comparisons"
            ),
        )
    ordered_ids = sorted(shared)

    names = [run.condition for run in runs]
    if baseline is not None and baseline not in names:
        raise HTTPException(status_code=400, detail=f"unknown baseline {baseline!r}")

    pairs = (
        [(n, baseline) for n in names if n != baseline]
        if baseline
        else [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    )
    comparisons = apply_correction(
        [
            compare_paired(
                [scores[a][q] for q in ordered_ids],
                [scores[b][q] for q in ordered_ids],
                name_a=a,
                name_b=b,
                metric=metric,
                alpha=alpha,
            )
            for a, b in pairs
        ],
        alpha=alpha,
    )

    return ExperimentOut(
        experiment=experiment,
        metric=metric,
        runs=[
            ExperimentRunOut(
                condition=r.condition,
                chunker=r.chunker,
                retriever=r.retriever,
                index_stats=r.index_stats,
                aggregate=r.aggregate,
                n_queries=r.n_queries,
            )
            for r in runs
        ],
        comparisons=[
            ComparisonOut(
                name_a=c.name_a,
                name_b=c.name_b,
                metric=c.metric,
                n=c.n,
                mean_a=c.mean_a,
                mean_b=c.mean_b,
                mean_diff=c.mean_diff,
                ci_low=c.ci_low,
                ci_high=c.ci_high,
                p_value=c.p_value,
                p_adjusted=c.p_adjusted,
                effect_size=c.effect_size,
                effect_label=c.effect_label,
                significant=c.significant,
            )
            for c in comparisons
        ],
        methods_note=(
            f"Paired comparisons on {metric} over {len(ordered_ids)} queries scored under every "
            f"condition; Wilcoxon signed-rank at alpha={alpha}, Holm-Bonferroni corrected; "
            f"95% CIs are percentile bootstrap intervals on the paired differences."
        ),
    )
