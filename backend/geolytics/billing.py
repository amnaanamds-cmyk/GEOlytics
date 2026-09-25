"""Stripe subscription billing.

Only three things actually matter for correctness here, and all three are
about trusting the right source:

1. **The webhook signature is verified before anything is read.** An unsigned
   webhook endpoint is a free upgrade button: anyone who knows the URL can
   POST a "subscription active, plan growth" event. `Settings` refuses to
   start with billing enabled and no webhook secret.
2. **Plan state comes from Stripe, never from the client.** The checkout
   endpoint takes a plan name only to pick a price id; what the organisation
   is actually on is whatever the webhook last told us.
3. **Webhooks are idempotent and order-tolerant.** Stripe retries on any
   non-2xx and does not guarantee ordering, so the same event may arrive
   twice and an older one may arrive after a newer one.

`StripeClient` is an interface with a real implementation and a fake. The fake
is not a shortcut: it lets the whole subscription lifecycle be tested without
network access, which is what allows this code to be exercised at all in an
environment with no Stripe credentials.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from geolytics.db.models import AuditLogEntry, Organization
from geolytics.tenancy.plans import PLANS, PlanName

logger = logging.getLogger(__name__)

# Events that change what a customer may do. Anything else is ignored rather
# than rejected, so Stripe does not retry events we simply do not care about.
HANDLED_EVENTS = frozenset(
    {
        "checkout.session.completed",
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
        "invoice.payment_failed",
    }
)

# Stripe subscription statuses that should keep service running. "past_due"
# is included deliberately: a failed card should trigger dunning, not an
# instant lockout of a paying customer.
ACTIVE_STATUSES = frozenset({"active", "trialing", "past_due"})


class BillingError(Exception):
    """A billing operation could not be completed."""


class SignatureError(BillingError):
    """A webhook signature was missing, malformed or wrong."""


@dataclass(frozen=True, slots=True)
class CheckoutResult:
    url: str
    session_id: str


@dataclass(frozen=True, slots=True)
class SubscriptionState:
    """What a webhook told us about one organisation's subscription."""

    customer_id: str
    subscription_id: str | None
    plan: PlanName
    status: str
    current_period_end: datetime | None

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES


class StripeClient(ABC):
    """The slice of Stripe this application uses."""

    @abstractmethod
    def create_checkout_session(
        self, *, customer_id: str | None, price_id: str, org_id: int,
        success_url: str, cancel_url: str, customer_email: str | None = None,
    ) -> CheckoutResult: ...

    @abstractmethod
    def create_billing_portal_session(self, *, customer_id: str, return_url: str) -> str: ...


class RealStripeClient(StripeClient):  # pragma: no cover - needs a Stripe account
    """Talks to Stripe. Not exercised by the test suite, which has no credentials."""

    def __init__(self, secret_key: str) -> None:
        try:
            import stripe
        except ImportError as exc:
            raise BillingError(
                "billing requires the 'billing' extra: pip install -e '.[billing]'"
            ) from exc
        stripe.api_key = secret_key
        self._stripe = stripe

    def create_checkout_session(
        self, *, customer_id: str | None, price_id: str, org_id: int,
        success_url: str, cancel_url: str, customer_email: str | None = None,
    ) -> CheckoutResult:
        session = self._stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            customer=customer_id,
            customer_email=None if customer_id else customer_email,
            # The org id travels with the session so the webhook can attribute
            # the subscription without trusting anything the browser returns.
            client_reference_id=str(org_id),
            metadata={"org_id": str(org_id)},
            subscription_data={"metadata": {"org_id": str(org_id)}},
        )
        return CheckoutResult(url=session.url, session_id=session.id)

    def create_billing_portal_session(self, *, customer_id: str, return_url: str) -> str:
        session = self._stripe.billing_portal.Session.create(
            customer=customer_id, return_url=return_url
        )
        return session.url


@dataclass
class FakeStripeClient(StripeClient):
    """In-memory Stripe stand-in for development and tests."""

    sessions: list[dict[str, Any]] = field(default_factory=list)
    base_url: str = "https://checkout.stripe.test"

    def create_checkout_session(
        self, *, customer_id: str | None, price_id: str, org_id: int,
        success_url: str, cancel_url: str, customer_email: str | None = None,
    ) -> CheckoutResult:
        session_id = f"cs_test_{len(self.sessions) + 1:06d}"
        self.sessions.append(
            {
                "id": session_id,
                "org_id": org_id,
                "price_id": price_id,
                "customer": customer_id,
                "success_url": success_url,
                "cancel_url": cancel_url,
            }
        )
        return CheckoutResult(url=f"{self.base_url}/{session_id}", session_id=session_id)

    def create_billing_portal_session(self, *, customer_id: str, return_url: str) -> str:
        return f"{self.base_url}/portal/{customer_id}?return_to={return_url}"


def verify_webhook_signature(
    payload: bytes, signature_header: str, secret: str, tolerance_seconds: int = 300
) -> None:
    """Verify Stripe's `Stripe-Signature` header.

    Implemented here rather than via `stripe.Webhook.construct_event` so the
    verification has no optional-dependency hole: if the `stripe` package were
    missing, a `construct_event` wrapped in a try/except would be exactly the
    place a "skip verification" fallback creeps in.

    The timestamp check is what stops a captured webhook being replayed
    forever, and the comparison is constant-time.
    """
    if not secret:
        raise SignatureError("no webhook secret configured")
    if not signature_header:
        raise SignatureError("missing Stripe-Signature header")

    timestamp: str | None = None
    signatures: list[str] = []
    for part in signature_header.split(","):
        key, _, value = part.strip().partition("=")
        if key == "t":
            timestamp = value
        elif key == "v1":
            signatures.append(value)

    if timestamp is None or not signatures:
        raise SignatureError("malformed Stripe-Signature header")

    try:
        age = abs(time.time() - int(timestamp))
    except ValueError as exc:
        raise SignatureError("invalid timestamp in Stripe-Signature") from exc
    if age > tolerance_seconds:
        raise SignatureError(f"signature timestamp is {int(age)}s old")

    expected = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.".encode() + payload,
        hashlib.sha256,
    ).hexdigest()

    if not any(hmac.compare_digest(expected, candidate) for candidate in signatures):
        raise SignatureError("signature mismatch")


def sign_webhook(payload: bytes, secret: str, timestamp: int | None = None) -> str:
    """Produce a valid `Stripe-Signature` header. For tests and local replay."""
    ts = timestamp if timestamp is not None else int(time.time())
    digest = hmac.new(
        secret.encode("utf-8"), f"{ts}.".encode() + payload, hashlib.sha256
    ).hexdigest()
    return f"t={ts},v1={digest}"


def plan_for_price(price_ids: dict[str, str], price_id: str | None) -> PlanName | None:
    """Map a Stripe price id back to a plan name."""
    if not price_id:
        return None
    for plan, configured in price_ids.items():
        if configured == price_id and plan in PLANS:
            return plan  # type: ignore[return-value]
    return None


def parse_event(event: dict[str, Any], price_ids: dict[str, str]) -> SubscriptionState | None:
    """Extract subscription state from a verified Stripe event.

    Returns None for events that do not change entitlement, so the caller can
    acknowledge them with a 200 and stop Stripe retrying.
    """
    event_type = event.get("type", "")
    if event_type not in HANDLED_EVENTS:
        return None

    obj = (event.get("data") or {}).get("object") or {}
    customer_id = obj.get("customer")
    if not isinstance(customer_id, str) or not customer_id:
        return None

    if event_type == "customer.subscription.deleted":
        return SubscriptionState(
            customer_id=customer_id,
            subscription_id=obj.get("id"),
            plan="free",
            status="canceled",
            current_period_end=_timestamp(obj.get("current_period_end")),
        )

    if event_type == "invoice.payment_failed":
        # Not a downgrade on its own. Stripe will move the subscription to
        # past_due or canceled and send that separately; acting here would cut
        # off a customer whose retry is about to succeed.
        return None

    price_id = _price_id(obj)
    plan = plan_for_price(price_ids, price_id)
    if plan is None:
        logger.warning(
            "stripe event for an unmapped price",
            extra={"price_id": price_id, "event_type": event_type},
        )
        return None

    return SubscriptionState(
        customer_id=customer_id,
        subscription_id=obj.get("subscription") or obj.get("id"),
        plan=plan,
        status=obj.get("status") or "active",
        current_period_end=_timestamp(obj.get("current_period_end")),
    )


def _price_id(obj: dict[str, Any]) -> str | None:
    items = (obj.get("items") or {}).get("data") or []
    if items:
        return (items[0].get("price") or {}).get("id")
    if obj.get("plan"):
        return obj["plan"].get("id")
    return obj.get("price")


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(value, tz=UTC)
    return None


def apply_subscription(
    session: Session, state: SubscriptionState, org_id: int | None = None
) -> Organization | None:
    """Apply verified subscription state to an organisation.

    Matching is by Stripe customer id, falling back to the org id carried in
    the checkout session's metadata for the very first event, before the
    customer id has been stored.
    """
    org = None
    if state.customer_id:
        org = session.scalar(
            select(Organization).where(
                Organization.billing_customer_id == state.customer_id
            )
        )
    if org is None and org_id is not None:
        org = session.get(Organization, org_id)
    if org is None:
        logger.warning(
            "stripe event for an unknown organisation",
            extra={"customer_id": state.customer_id, "org_id": org_id},
        )
        return None

    # Out-of-order delivery is normal. An event describing a period that has
    # already been superseded must not roll a customer back onto an old plan.
    # Timestamps are normalised because not every backend returns the offset.
    known_period = _as_utc(org.current_period_end)
    if (
        known_period is not None
        and state.current_period_end is not None
        and state.current_period_end < known_period
    ):
        logger.info(
            "ignoring stale stripe event",
            extra={"org_id": org.id, "event_period_end": str(state.current_period_end)},
        )
        return org

    org.billing_customer_id = state.customer_id
    org.billing_subscription_id = state.subscription_id
    org.plan = state.plan if state.is_active else "free"
    org.status = "active" if state.is_active else "suspended"
    org.current_period_end = state.current_period_end

    session.add(
        AuditLogEntry(
            org_id=org.id,
            actor="stripe",
            action="billing.subscription_updated",
            target=state.subscription_id,
            detail={"plan": org.plan, "status": state.status},
        )
    )
    return org


def build_client(settings: Any) -> StripeClient:
    """Real client when configured, fake otherwise.

    Falling back to the fake outside production keeps local development
    working without Stripe credentials; production refuses to start with
    billing enabled and no webhook secret, so it cannot silently run on a fake.
    """
    if settings.stripe_secret_key:
        return RealStripeClient(settings.stripe_secret_key)
    if settings.env == "production" and settings.billing_enabled:
        raise BillingError("billing is enabled but GEOLYTICS_STRIPE_SECRET_KEY is not set")
    return FakeStripeClient()
