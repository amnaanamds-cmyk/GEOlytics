"""Render run results and comparisons as tables for the report."""

from __future__ import annotations

from collections.abc import Sequence

from geolytics.evaluation.harness import RunResult, common_query_ids
from geolytics.evaluation.stats import PairedComparison


def results_table(runs: Sequence[RunResult], metrics: Sequence[str] | None = None) -> str:
    """Markdown table of mean metrics, with the corpus-shape columns attached.

    `n chunks` and `mean tok` are not decoration: without them a reader cannot
    tell a genuine retrieval effect from a chunk-size effect.
    """
    if not runs:
        return "(no runs)"

    metric_names = list(metrics) if metrics else sorted(next(iter(runs)).aggregate())
    header = ["condition", "n chunks", "mean tok", "n queries", *metric_names]
    rows = [
        [
            run.condition,
            str(run.index_stats.n_chunks),
            f"{run.index_stats.mean_tokens:.1f}",
            str(run.n_queries_scored),
            *[f"{run.mean(m):.4f}" for m in metric_names],
        ]
        for run in runs
    ]
    return _markdown_table(header, rows)


def comparison_table(comparisons: Sequence[PairedComparison]) -> str:
    """Markdown table of paired comparisons -- the table the viva will ask about."""
    if not comparisons:
        return "(no comparisons)"

    header = [
        "A", "B", "metric", "mean A", "mean B", "diff", "95% CI",
        "p", "p (Holm)", "delta", "effect", "sig",
    ]
    rows = []
    for c in comparisons:
        rows.append(
            [
                c.name_a,
                c.name_b,
                c.metric,
                f"{c.mean_a:.4f}",
                f"{c.mean_b:.4f}",
                f"{c.mean_diff:+.4f}",
                f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}]",
                f"{c.p_value:.4g}",
                "-" if c.p_adjusted is None else f"{c.p_adjusted:.4g}",
                f"{c.effect_size:+.3f}",
                c.effect_label,
                "yes" if c.significant else "no",
            ]
        )
    return _markdown_table(header, rows)


def methods_note(runs: Sequence[RunResult], metric: str, alpha: float = 0.05) -> str:
    """A paragraph describing how the numbers were produced.

    Paste this into the methodology section rather than writing it from memory
    -- it is generated from the runs themselves, so it cannot drift from what
    was actually executed.
    """
    shared = common_query_ids(runs)
    dropped = {r.condition: r.projection.n_unmatched for r in runs}
    dropped_note = ", ".join(f"{name}: {n}" for name, n in sorted(dropped.items()))
    return (
        f"All {len(runs)} conditions were evaluated on the same query set. "
        f"Relevance was authored as document-level spans and projected onto each "
        f"chunking independently; queries whose gold span could not be located in a "
        f"condition's chunks were excluded from that condition "
        f"(unmatched per condition -- {dropped_note}). "
        f"Paired tests were computed over the {len(shared)} queries scored under every "
        f"condition. Significance on {metric} was assessed with the Wilcoxon signed-rank "
        f"test at alpha = {alpha}, Holm-Bonferroni corrected across the comparison family; "
        f"95% confidence intervals for the mean difference are percentile bootstrap "
        f"intervals over 10,000 resamples of the paired differences, and effect sizes "
        f"are Cliff's delta."
    )


def _markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(str(header[i])), *(len(str(r[i])) for r in rows)) if rows else len(header[i])
        for i in range(len(header))
    ]
    lines = [
        "| " + " | ".join(str(h).ljust(w) for h, w in zip(header, widths, strict=True)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    lines.extend(
        "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)) + " |"
        for row in rows
    )
    return "\n".join(lines)
