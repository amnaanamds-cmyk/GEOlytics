"""Stripe billing: signature verification, event mapping and plan application.

The webhook signature is the entire authentication for the billing endpoint --
Stripe cannot hold a credential -- so an unverified webhook is a free upgrade
button for anyone who learns the URL.
"""

from __future__ import annotations

import json
import time

import pytest

from geolytics.billing import (
    FakeStripeClient,
    SignatureError,
    apply_subscription,
    parse_event,
    plan_for_price,
    sign_webhook,
    verify_webhook_signature,
)

SECRET = "whsec_test_secret_value_0123456789"
PRICE_IDS = {"starter": "price_starter_1", "growth": "price_growth_1"}


def event(event_type: str, obj: dict, **top) -> dict:
    return {"id": "evt_1", "type": event_type, "data": {"object": obj}, **top}


class TestSignatureVerification:
    def test_accepts_a_valid_signature(self):
        payload = b'{"id":"evt_1"}'
        verify_webhook_signature(payload, sign_webhook(payload, SECRET), SECRET)

    def test_rejects_a_wrong_secret(self):
        payload = b'{"id":"evt_1"}'
        with pytest.raises(SignatureError, match="mismatch"):
            verify_webhook_signature(payload, sign_webhook(payload, "other"), SECRET)

    def test_rejects_a_modified_body(self):
        header = sign_webhook(b'{"plan":"starter"}', SECRET)
        with pytest.raises(SignatureError, match="mismatch"):
            verify_webhook_signature(b'{"plan":"growth"}', header, SECRET)

    def test_rejects_a_replayed_old_signature(self):
        """Without the timestamp check a captured webhook replays forever."""
        payload = b'{"id":"evt_1"}'
        old = sign_webhook(payload, SECRET, timestamp=int(time.time()) - 3600)
        with pytest.raises(SignatureError, match="old"):
            verify_webhook_signature(payload, old, SECRET)

    def test_accepts_a_signature_inside_the_tolerance(self):
        payload = b'{"id":"evt_1"}'
        recent = sign_webhook(payload, SECRET, timestamp=int(time.time()) - 60)
        verify_webhook_signature(payload, recent, SECRET)

    @pytest.mark.parametrize("header", ["", "garbage", "t=123", "v1=abc", "t=abc,v1=def"])
    def test_rejects_malformed_headers(self, header):
        with pytest.raises(SignatureError):
            verify_webhook_signature(b"{}", header, SECRET)

    def test_rejects_when_no_secret_is_configured(self):
        payload = b"{}"
        with pytest.raises(SignatureError, match="no webhook secret"):
            verify_webhook_signature(payload, sign_webhook(payload, SECRET), "")

    def test_accepts_when_one_of_several_signatures_matches(self):
        """Stripe sends several v1 values during a secret rotation."""
        payload = b'{"id":"evt_1"}'
        valid = sign_webhook(payload, SECRET)
        timestamp = valid.split(",")[0].split("=")[1]
        combined = f"{valid},v1=00{'0' * 62}"
        verify_webhook_signature(payload, combined, SECRET)
        assert timestamp


class TestPriceMapping:
    def test_maps_a_known_price_to_its_plan(self):
        assert plan_for_price(PRICE_IDS, "price_growth_1") == "growth"

    def test_unknown_price_maps_to_nothing(self):
        assert plan_for_price(PRICE_IDS, "price_unknown") is None
        assert plan_for_price(PRICE_IDS, None) is None


class TestEventParsing:
    def _subscription(self, price_id: str, status: str = "active") -> dict:
        return {
            "id": "sub_1",
            "customer": "cus_1",
            "status": status,
            "current_period_end": int(time.time()) + 86400,
            "items": {"data": [{"price": {"id": price_id}}]},
        }

    def test_subscription_created_yields_the_plan(self):
        state = parse_event(
            event("customer.subscription.created", self._subscription("price_growth_1")),
            PRICE_IDS,
        )
        assert state.plan == "growth"
        assert state.is_active

    def test_deletion_downgrades_to_free(self):
        state = parse_event(
            event("customer.subscription.deleted", self._subscription("price_growth_1")),
            PRICE_IDS,
        )
        assert state.plan == "free"
        assert not state.is_active

    def test_past_due_keeps_service_running(self):
        """A failed card starts dunning; it must not lock a paying customer out."""
        state = parse_event(
            event(
                "customer.subscription.updated",
                self._subscription("price_growth_1", status="past_due"),
            ),
            PRICE_IDS,
        )
        assert state.is_active

    def test_payment_failed_alone_changes_nothing(self):
        assert parse_event(
            event("invoice.payment_failed", {"customer": "cus_1"}), PRICE_IDS
        ) is None

    def test_unhandled_event_types_are_ignored(self):
        assert parse_event(event("customer.created", {"customer": "cus_1"}), PRICE_IDS) is None

    def test_unmapped_price_is_ignored_rather_than_guessed(self):
        assert parse_event(
            event("customer.subscription.updated", self._subscription("price_mystery")),
            PRICE_IDS,
        ) is None

    def test_event_without_a_customer_is_ignored(self):
        assert parse_event(
            event("customer.subscription.updated", {"id": "sub_1"}), PRICE_IDS
        ) is None


class TestApplySubscription:
    @pytest.fixture
    def session(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEOLYTICS_POSTGRES_DSN", f"sqlite:///{tmp_path/'b.db'}")
        from geolytics.config import get_settings
        from geolytics.db.models import Base
        from geolytics.db.session import get_engine, get_session_factory

        for cached in (get_settings, get_engine, get_session_factory):
            cached.cache_clear()
        Base.metadata.create_all(get_engine())
        session = get_session_factory()()
        yield session
        session.close()
        for cached in (get_settings, get_engine, get_session_factory):
            cached.cache_clear()

    @pytest.fixture
    def org(self, session):
        from geolytics.db.models import Organization

        org = Organization(slug="acme", name="Acme", plan="free")
        session.add(org)
        session.flush()
        return org

    def _state(self, plan="growth", status="active", period_offset=86400):
        from datetime import UTC, datetime, timedelta

        from geolytics.billing import SubscriptionState

        return SubscriptionState(
            customer_id="cus_1",
            subscription_id="sub_1",
            plan=plan,
            status=status,
            current_period_end=datetime.now(UTC) + timedelta(seconds=period_offset),
        )

    def test_first_event_attaches_by_org_metadata(self, session, org):
        applied = apply_subscription(session, self._state(), org_id=org.id)
        assert applied is not None
        assert applied.plan == "growth"
        assert applied.billing_customer_id == "cus_1"

    def test_later_events_match_on_customer_id(self, session, org):
        apply_subscription(session, self._state(), org_id=org.id)
        session.flush()
        applied = apply_subscription(session, self._state(plan="starter"))
        assert applied.id == org.id
        assert applied.plan == "starter"

    def test_cancellation_drops_to_free_and_suspends(self, session, org):
        apply_subscription(session, self._state(), org_id=org.id)
        session.flush()
        applied = apply_subscription(session, self._state(plan="free", status="canceled"))
        assert applied.plan == "free"
        assert applied.status == "suspended"

    def test_a_stale_event_does_not_roll_the_plan_back(self, session, org):
        """Stripe does not guarantee ordering; an older event may arrive last."""
        apply_subscription(session, self._state(plan="growth"), org_id=org.id)
        session.flush()
        applied = apply_subscription(
            session, self._state(plan="starter", period_offset=-86400)
        )
        assert applied.plan == "growth"

    def test_applying_twice_is_idempotent(self, session, org):
        state = self._state()
        apply_subscription(session, state, org_id=org.id)
        session.flush()
        applied = apply_subscription(session, state)
        assert applied.plan == "growth"

    def test_unknown_organisation_is_ignored(self, session):
        assert apply_subscription(session, self._state()) is None

    def test_an_audit_log_entry_is_written(self, session, org):
        from geolytics.db.models import AuditLogEntry

        apply_subscription(session, self._state(), org_id=org.id)
        session.flush()
        entries = session.query(AuditLogEntry).all()
        assert entries
        assert entries[0].actor == "stripe"


class TestFakeClient:
    def test_creates_a_checkout_session(self):
        client = FakeStripeClient()
        result = client.create_checkout_session(
            customer_id=None,
            price_id="price_growth_1",
            org_id=7,
            success_url="https://app/ok",
            cancel_url="https://app/no",
        )
        assert result.session_id.startswith("cs_test_")
        assert client.sessions[0]["org_id"] == 7

    def test_round_trips_through_the_webhook_path(self):
        """A signed event built here verifies and parses, end to end."""
        body = json.dumps(
            event(
                "customer.subscription.created",
                {
                    "id": "sub_1",
                    "customer": "cus_1",
                    "status": "active",
                    "current_period_end": int(time.time()) + 86400,
                    "items": {"data": [{"price": {"id": "price_growth_1"}}]},
                },
            )
        ).encode()
        verify_webhook_signature(body, sign_webhook(body, SECRET), SECRET)
        assert parse_event(json.loads(body), PRICE_IDS).plan == "growth"
