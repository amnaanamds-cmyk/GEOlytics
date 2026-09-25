"""Subscription plans and the limits they carry.

Limits live in code rather than the database on purpose: they are part of the
product definition, they change by deployment rather than by customer, and
keeping them here means a plan change is a reviewable diff rather than an
UPDATE nobody can trace. An organisation that needs something bespoke gets
per-org overrides (`Organization.limit_overrides`) instead of a new row here.

Every limit is a *ceiling that costs real money to serve*. Pages crawled drive
bandwidth and worker time; embedded chunks drive vector storage; concurrent
audits drive worker count.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

PlanName = Literal["free", "starter", "growth", "enterprise"]


@dataclass(frozen=True, slots=True)
class Plan:
    """What one subscription tier permits."""

    name: PlanName
    display_name: str
    monthly_price_cents: int
    max_sites: int
    max_pages_per_audit: int
    max_pages_per_month: int
    max_audits_per_month: int
    max_concurrent_audits: int
    max_api_keys: int
    max_members: int
    retention_days: int
    # Experiments are the expensive path: every condition re-embeds the whole
    # corpus, so the grid multiplies cost by the number of conditions.
    experiments_enabled: bool = False
    max_experiment_runs_per_month: int = 0
    support: str = "community"

    def limit(self, name: str) -> int:
        try:
            return int(getattr(self, name))
        except AttributeError:
            raise KeyError(f"unknown limit {name!r}") from None

    def with_overrides(self, overrides: dict[str, int] | None) -> Plan:
        """Apply per-organisation overrides, ignoring names that are not limits."""
        if not overrides:
            return self
        known = {k: v for k, v in overrides.items() if hasattr(self, k)}
        return replace(self, **known) if known else self


PLANS: dict[PlanName, Plan] = {
    "free": Plan(
        name="free",
        display_name="Free",
        monthly_price_cents=0,
        max_sites=1,
        max_pages_per_audit=25,
        max_pages_per_month=100,
        max_audits_per_month=5,
        max_concurrent_audits=1,
        max_api_keys=1,
        max_members=1,
        retention_days=30,
        experiments_enabled=False,
        support="community",
    ),
    "starter": Plan(
        name="starter",
        display_name="Starter",
        monthly_price_cents=4900,
        max_sites=5,
        max_pages_per_audit=100,
        max_pages_per_month=2_500,
        max_audits_per_month=50,
        max_concurrent_audits=2,
        max_api_keys=5,
        max_members=3,
        retention_days=180,
        experiments_enabled=False,
        support="email",
    ),
    "growth": Plan(
        name="growth",
        display_name="Growth",
        monthly_price_cents=19900,
        max_sites=25,
        max_pages_per_audit=500,
        max_pages_per_month=25_000,
        max_audits_per_month=500,
        max_concurrent_audits=5,
        max_api_keys=20,
        max_members=10,
        retention_days=365,
        experiments_enabled=True,
        max_experiment_runs_per_month=50,
        support="email",
    ),
    "enterprise": Plan(
        name="enterprise",
        display_name="Enterprise",
        monthly_price_cents=0,  # negotiated; billing handled outside the app
        max_sites=1_000,
        max_pages_per_audit=5_000,
        max_pages_per_month=500_000,
        max_audits_per_month=10_000,
        max_concurrent_audits=25,
        max_api_keys=100,
        max_members=250,
        retention_days=1_095,
        experiments_enabled=True,
        max_experiment_runs_per_month=1_000,
        support="dedicated",
    ),
}

DEFAULT_PLAN: PlanName = "free"


def get_plan(name: str | None, overrides: dict[str, int] | None = None) -> Plan:
    """Resolve a plan by name, falling back to Free for anything unrecognised.

    Falling back rather than raising is deliberate: an unknown plan string --
    from a bad migration or a hand-edited row -- must degrade to the most
    restrictive tier, never to unlimited service.
    """
    plan = PLANS.get(name or DEFAULT_PLAN, PLANS[DEFAULT_PLAN])
    return plan.with_overrides(overrides)
