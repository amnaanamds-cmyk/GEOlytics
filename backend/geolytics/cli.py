"""Command line entry point: `geolytics <command>`."""

from __future__ import annotations

import argparse
import json
import re
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
    p_exp.add_argument(
        "--persist-as",
        default=None,
        metavar="NAME",
        help="also store runs in the database under this experiment name",
    )
    p_exp.add_argument(
        "--generator",
        choices=("auto", "llm", "heuristic"),
        default="auto",
        help="query generator; 'auto' falls back to templates when no LLM is configured",
    )

    p_eq = sub.add_parser(
        "check-metrics",
        help="test whether cosine/dot/euclidean are separate conditions for this embedder",
    )
    p_eq.add_argument("--n-passages", type=int, default=200)

    p_cal = sub.add_parser(
        "calibrate",
        help="fit GEO signal weights against visibility in the simulated engine",
    )
    p_cal.add_argument("url")
    p_cal.add_argument("--max-pages", type=int, default=30)
    p_cal.add_argument("--chunker", default="sentence")
    p_cal.add_argument("--out", type=Path, default=Path("runs/weights.json"))

    p_org = sub.add_parser(
        "create-org", help="create an organisation and its first owner"
    )
    p_org.add_argument("name")
    p_org.add_argument("--email", required=True)
    p_org.add_argument("--password", help="read from GEOLYTICS_ADMIN_PASSWORD if omitted")
    p_org.add_argument("--plan", default="free")

    sub.add_parser("init-db", help="create database tables (prefer: alembic upgrade head)")

    args = parser.parse_args(argv)

    if args.command == "audit":
        return _audit(args)
    if args.command == "experiment":
        return _experiment(args)
    if args.command == "create-org":
        return _create_org(args)
    if args.command == "calibrate":
        return _calibrate(args)
    if args.command == "check-metrics":
        return _check_metrics(args)
    if args.command == "init-db":
        from geolytics.db.session import create_all

        create_all()
        print(
            "tables created.\nFor anything but a throwaway database, use "
            "`alembic upgrade head` instead so the schema is versioned."
        )
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
        generator=args.generator,
        persist_as=args.persist_as,
    )
    print(report)
    return 0


def _create_org(args: argparse.Namespace) -> int:
    """Bootstrap the first tenant, for a fresh self-hosted install."""
    import os

    from sqlalchemy import func, select

    from geolytics.db.models import Membership, Organization, User
    from geolytics.db.session import session_scope
    from geolytics.security.passwords import PasswordPolicyError, hash_password
    from geolytics.tenancy.plans import PLANS

    if args.plan not in PLANS:
        print(f"unknown plan {args.plan!r}; expected one of {sorted(PLANS)}", file=sys.stderr)
        return 2

    password = args.password or os.environ.get("GEOLYTICS_ADMIN_PASSWORD")
    if not password:
        # Never prompt-free-default a password, and never accept one on the
        # command line silently: argv is visible in `ps` on a shared host.
        import getpass

        password = getpass.getpass("Owner password: ")

    try:
        password_hash = hash_password(password)
    except PasswordPolicyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    email = args.email.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", args.name.strip().lower()).strip("-")[:48] or "org"

    with session_scope() as session:
        if session.scalar(select(Organization.id).where(Organization.slug == slug)):
            print(f"an organisation with slug {slug!r} already exists", file=sys.stderr)
            return 1

        user = session.scalar(select(User).where(func.lower(User.email) == email))
        if user is None:
            user = User(email=email, password_hash=password_hash, is_verified=True)
            session.add(user)
            session.flush()

        org = Organization(slug=slug, name=args.name.strip(), plan=args.plan)
        session.add(org)
        session.flush()
        session.add(Membership(user_id=user.id, org_id=org.id, role="owner"))

        print(f"created organisation {org.name!r} (slug: {org.slug}, plan: {org.plan})")
        print(f"owner: {user.email}")
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    from geolytics.chunking.registry import build_chunker
    from geolytics.config import get_settings
    from geolytics.crawl.crawler import Crawler
    from geolytics.embedding.registry import build_embedder
    from geolytics.experiments.chunking_grid import build_query_set
    from geolytics.geo.calibration import as_dict, calibrate_weights
    from geolytics.geo.engine import SimulatedGenerativeEngine
    from geolytics.index.memory import InMemoryVectorStore
    from geolytics.llm import build_llm
    from geolytics.retrieval.dense import DenseRetriever

    settings = get_settings()

    # Calibration regresses signals against how much of a *generated answer*
    # each page won. With no model there are no answers, so the target would be
    # uniformly zero -- say so before spending a crawl on it.
    if settings.llm_backend == "none":
        print(
            "calibration needs a generative model: it measures visibility in "
            "generated answers, and no LLM backend is configured.\n"
            "Set GEOLYTICS_LLM_BACKEND=ollama (and GEOLYTICS_OLLAMA_MODEL) first.",
            file=sys.stderr,
        )
        return 2

    embedder = build_embedder(settings)

    with Crawler(settings=settings) as crawler:
        results = crawler.crawl(args.url, max_pages=args.max_pages)
        print(crawler.stats.summary(), file=sys.stderr)
    if not results:
        print(f"no pages crawled from {args.url}", file=sys.stderr)
        return 1

    documents = [r.document for r in results]
    html_by_doc = {r.document.doc_id: r.html for r in results}

    chunker = build_chunker(args.chunker, embedder=embedder)
    chunks_by_doc = {d.doc_id: chunker.chunk(d) for d in documents}
    chunks = [c for group in chunks_by_doc.values() for c in group]
    if not chunks:
        print("extraction produced no chunks", file=sys.stderr)
        return 1

    store = InMemoryVectorStore()
    store.create_collection("calibrate", dim=embedder.dim)
    store.upsert("calibrate", chunks, embedder.embed([c.text for c in chunks]))

    query_set, generation = build_query_set(documents, settings)
    print(generation.summary(), file=sys.stderr)

    engine = SimulatedGenerativeEngine(
        DenseRetriever(store, embedder, "calibrate"), build_llm(settings), top_k=5
    )
    report = calibrate_weights(
        documents, chunks_by_doc, engine, [q.text for q in query_set], html_by_doc
    )

    print(report.summary())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(as_dict(report), indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0 if report.usable else 1


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
