"""Command line entry point: `geolytics <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from geolytics import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geolytics", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="crawl a site and score every page")
    p_audit.add_argument("url")
    p_audit.add_argument("--max-pages", type=int, default=10)
    p_audit.add_argument("--chunker", default="sentence")
    p_audit.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    p_exp = sub.add_parser(
        "experiment", help="run the chunking x retrieval grid and report paired statistics"
    )
    p_exp.add_argument("url")
    p_exp.add_argument("--max-pages", type=int, default=10)
    p_exp.add_argument("--metric", default="ndcg@10")
    p_exp.add_argument("--baseline", default="fixed+dense")
    p_exp.add_argument("--out", type=Path, default=Path("runs/experiment.json"))

    p_eq = sub.add_parser(
        "check-metrics",
        help="test whether cosine/dot/euclidean are separate conditions for this embedder",
    )
    p_eq.add_argument("--n-passages", type=int, default=200)

    sub.add_parser("init-db", help="create database tables")

    args = parser.parse_args(argv)

    if args.command == "audit":
        return _audit(args)
    if args.command == "experiment":
        return _experiment(args)
    if args.command == "check-metrics":
        return _check_metrics(args)
    if args.command == "init-db":
        from geolytics.db.session import create_all

        create_all()
        print("tables created")
        return 0
    return 1


def _audit(args: argparse.Namespace) -> int:
    from geolytics.pipeline import run_audit

    outcome = run_audit(args.url, max_pages=args.max_pages, chunker=args.chunker)

    if args.json:
        print(
            json.dumps(
                {
                    "site": outcome.site_url,
                    "overall_score": outcome.overall_score,
                    "weights_fitted": outcome.weights_fitted,
                    "warnings": outcome.warnings,
                    "pages": [s.to_dict() for s in outcome.scores],
                },
                indent=2,
            )
        )
        return 0

    print(f"\n{outcome.site_url} -- {outcome.crawl_summary}")
    print(f"overall GEO score: {outcome.overall_score:.1f}/100\n")
    for score in sorted(outcome.scores, key=lambda s: s.score, reverse=True):
        print(f"  {score.score:6.1f}  {score.url}")
        for recommendation in score.recommendations[:2]:
            print(f"          - {recommendation}")
    for warning in outcome.warnings:
        print(f"\nWARNING: {warning}", file=sys.stderr)
    return 0


def _experiment(args: argparse.Namespace) -> int:
    from geolytics.experiments.chunking_grid import run_chunking_experiment

    report = run_chunking_experiment(
        args.url,
        max_pages=args.max_pages,
        metric=args.metric,
        baseline=args.baseline,
        out_path=args.out,
    )
    print(report)
    return 0


def _check_metrics(args: argparse.Namespace) -> int:
    from geolytics.embedding.registry import build_embedder
    from geolytics.evaluation.rank_equivalence import verify_rank_equivalence

    embedder = build_embedder()
    passages = [
        f"Passage {i} covers subject {i % 17} with details about item {i % 7}."
        for i in range(args.n_passages)
    ]
    queries = [f"information about subject {i} and item {i % 7}" for i in range(20)]

    for metric_b in ("dot", "euclidean"):
        result = verify_rank_equivalence(
            embedder, passages, queries, metric_a="cosine", metric_b=metric_b
        )
        print(result.summary())

    if embedder.normalized:
        print(
            "\nThis embedder returns normalised vectors, so these metrics induce the same\n"
            "ranking by construction. Treat the similarity metric as a fixed setting, not\n"
            "an experimental factor, and vary the embedding model or fusion weight instead."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
