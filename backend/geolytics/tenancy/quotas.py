"""Plan quota accounting and enforcement.

Usage is metered in `usage_counters`, one row per (organisation, period,
metric), rather than derived from the audit tables. Retention deletes old
audits while billing history has to outlive them, so the two cannot be the
same data.

Increments are written with an UPDATE-then-INSERT that tolerates a concurrent
insert, because two workers finishing audits for the same organisation in the
same millisecond is normal, and a lost increment is unbilled usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from geolytics.db.models import Audit, Organization, Site, UsageCounter
from geolytics.tenancy.plans import Plan, get_plan

# Metric names. Constants rather than literals so a typo is an ImportError
# instead of a silently separate counter that never trips a limit.
PAGES_CRAWLED = "pages_crawled"
AUDITS_RUN = "audits_run"
EXPERIMENT_RUNS = "experiment_runs"

METRICS = (PAGES_CRAWLED, AUDITS_RUN, EXPERIMENT_RUNS)

_LIMIT_FOR_METRIC = {
    PAGES_CRAWLED: "max_pages_per_month",
    AUDITS_RUN: "max_audits_per_month",
    EXPERIMENT_RUNS: "max_experiment_runs_per_month",
}


class QuotaExceeded(Exception):
    """A request would take an organisation past a plan limit."""

    def __init__(self, limit_name: str, limit: int, used: int, requested: int = 0) -> None:
        detail = f"{limit_name}: {used}/{limit} used"
        if requested:
            detail += f", {requested} more requested"
        super().__init__(detail)
        self.limit_name = limit_name
        self.limit = limit
        self.used = used
        self.requested = requested

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    """What an organisation has consumed this period, against its plan."""

    period: str
    plan: Plan
    used: dict[str, int]
    sites: int
    concurrent_audits: int

    def remaining(self, metric: str) -> int:
        limit_name = _LIMIT_FOR_METRIC.get(metric)
        if limit_name is None:
            return 0
        return max(0, self.plan.limit(limit_name) - self.used.get(metric, 0))

    def to_dict(self) -> dict[str, object]:
        return {
            "period": self.period,
            "plan": self.plan.name,
            "used": dict(self.used),
            "limits": {
                metric: self.plan.limit(limit) for metric, limit in _LIMIT_FOR_METRIC.items()
            },
            "remaining": {metric: self.remaining(metric) for metric in METRICS},
            "sites": {"used": self.sites, "limit": self.plan.max_sites},
            "concurrent_audits": {
                "used": self.concurrent_audits,
                "limit": self.plan.max_concurrent_audits,
            },
        }


def current_period(now: datetime | None = None) -> str:
    """The billing period key, ``YYYY-MM`` in UTC.

    UTC rather than local time so a period boundary is the same instant for
    every customer and every worker, wherever they run.
    """
    return (now or datetime.now(UTC)).strftime("%Y-%m")


def get_usage(session: Session, org_id: int, period: str | None = None) -> dict[str, int]:
    period = period or current_period()
    rows = session.scalars(
        select(UsageCounter).where(
            UsageCounter.org_id == org_id, UsageCounter.period == period
        )
    ).all()
    usage = dict.fromkeys(METRICS, 0)
    usage.update({row.metric: row.value for row in rows})
    return usage


def record_usage(
    session: Session, org_id: int, metric: str, amount: int = 1, period: str | None = None
) -> int:
    """Add `amount` to a counter and return the new total.

    UPDATE first, INSERT only if no row existed, and tolerate another worker
    winning the race to insert. A plain read-modify-write here would lose
    increments under the concurrency this system is built for.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    if amount < 0:
        raise ValueError("usage amount must be non-negative")

    period = period or current_period()

    def _bump() -> int | None:
        result = session.execute(
            update(UsageCounter)
            .where(
                UsageCounter.org_id == org_id,
                UsageCounter.period == period,
                UsageCounter.metric == metric,
            )
            .values(value=UsageCounter.value + amount)
            .returning(UsageCounter.value)
        )
        row = result.first()
        return int(row[0]) if row else None

    total = _bump()
    if total is not None:
        return total

    try:
        with session.begin_nested():
            session.add(
                UsageCounter(org_id=org_id, period=period, metric=metric, value=amount)
            )
        return amount
    except IntegrityError:
        # Another worker inserted the row between our UPDATE and INSERT.
        total = _bump()
        if total is None:  # pragma: no cover - only if the row vanished again
            raise
        return total


def count_sites(session: Session, org_id: int) -> int:
    return int(
        session.scalar(select(func.count()).select_from(Site).where(Site.org_id == org_id)) or 0
    )


def count_running_audits(session: Session, org_id: int) -> int:
    """Audits that are queued or in flight, for the concurrency limit."""
    return int(
        session.scalar(
            select(func.count())
            .select_from(Audit)
            .where(Audit.org_id == org_id, Audit.status.in_(("pending", "running")))
        )
        or 0
    )


def snapshot(session: Session, org: Organization, period: str | None = None) -> UsageSnapshot:
    period = period or current_period()
    return UsageSnapshot(
        period=period,
        plan=get_plan(org.plan, org.limit_overrides),
        used=get_usage(session, org.id, period),
        sites=count_sites(session, org.id),
        concurrent_audits=count_running_audits(session, org.id),
    )


def check_audit_allowed(
    session: Session,
    org: Organization,
    requested_pages: int,
    period: str | None = None,
) -> int:
    """Check every limit an audit touches; return the page budget to use.

    Returns the number of pages the audit may crawl, which is the requested
    count capped by the per-audit limit. Capping rather than refusing is the
    kinder behaviour for the common case of a customer asking for more pages
    than their tier allows -- they get a smaller audit, not an error -- while
    the monthly limits still refuse outright, because those mean "you have
    spent what you bought".
    """
    plan = get_plan(org.plan, org.limit_overrides)
    usage = get_usage(session, org.id, period)

    if usage[AUDITS_RUN] >= plan.max_audits_per_month:
        raise QuotaExceeded("max_audits_per_month", plan.max_audits_per_month, usage[AUDITS_RUN])

    running = count_running_audits(session, org.id)
    if running >= plan.max_concurrent_audits:
        raise QuotaExceeded("max_concurrent_audits", plan.max_concurrent_audits, running)

    pages = min(max(1, requested_pages), plan.max_pages_per_audit)
    remaining_pages = plan.max_pages_per_month - usage[PAGES_CRAWLED]
    if remaining_pages <= 0:
        raise QuotaExceeded(
            "max_pages_per_month", plan.max_pages_per_month, usage[PAGES_CRAWLED], pages
        )

    return min(pages, remaining_pages)


def check_site_allowed(session: Session, org: Organization) -> None:
    plan = get_plan(org.plan, org.limit_overrides)
    used = count_sites(session, org.id)
    if used >= plan.max_sites:
        raise QuotaExceeded("max_sites", plan.max_sites, used)


def check_experiments_allowed(
    session: Session, org: Organization, period: str | None = None
) -> None:
    plan = get_plan(org.plan, org.limit_overrides)
    if not plan.experiments_enabled:
        raise QuotaExceeded("experiments_enabled", 0, 0)
    usage = get_usage(session, org.id, period)
    if usage[EXPERIMENT_RUNS] >= plan.max_experiment_runs_per_month:
        raise QuotaExceeded(
            "max_experiment_runs_per_month",
            plan.max_experiment_runs_per_month,
            usage[EXPERIMENT_RUNS],
        )
