"""Rate limiting against a real Redis, and the webhook endpoint end to end."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient
from services import REDIS_URL, requires_postgres, requires_redis

from geolytics.ratelimit import RateLimiter, build_limiter

pytestmark = requires_redis


@pytest.fixture
def limiter():
    lim = build_limiter(REDIS_URL, per_minute=60, burst=5, namespace="rl:test")
    lim.reset("subject")
    yield lim
    lim.reset("subject")


class TestTokenBucket:
    def test_allows_up_to_capacity_then_refuses(self, limiter):
        # Capacity is rate + burst; the bucket starts full.
        allowed = sum(1 for _ in range(70) if limiter.check("subject").allowed)
        assert allowed == 65
        assert not limiter.check("subject").allowed

    def test_refuses_with_a_retry_after(self, limiter):
        for _ in range(70):
            limiter.check("subject")
        result = limiter.check("subject")
        assert not result.allowed
        assert result.retry_after >= 1
        assert result.headers()["Retry-After"] == str(result.retry_after)

    def test_refills_over_time(self, limiter):
        for _ in range(70):
            limiter.check("subject")
        assert not limiter.check("subject").allowed
        # 60/minute is one token per second.
        time.sleep(1.2)
        assert limiter.check("subject").allowed

    def test_separate_keys_have_separate_buckets(self, limiter):
        for _ in range(70):
            limiter.check("subject")
        assert not limiter.check("subject").allowed
        limiter.reset("other")
        assert limiter.check("other").allowed

    def test_cost_greater_than_one_is_charged(self, limiter):
        assert limiter.check("subject", cost=60).allowed
        assert not limiter.check("subject", cost=10).allowed

    def test_headers_report_the_limit(self, limiter):
        headers = limiter.check("subject").headers()
        assert headers["X-RateLimit-Limit"] == "60"
        assert int(headers["X-RateLimit-Remaining"]) >= 0

    def test_rejects_a_nonsense_rate(self):
        with pytest.raises(ValueError, match="positive"):
            RateLimiter(None, per_minute=0)


class TestFailOpen:
    def test_an_unreachable_redis_allows_traffic_but_flags_it(self):
        """Redis being down must not take the API down with it."""

        class Broken:
            def register_script(self, script):
                raise ConnectionError("redis is gone")

        result = RateLimiter(Broken()).check("anyone")
        assert result.allowed
        assert result.degraded


@requires_postgres
class TestApiRateLimiting:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEOLYTICS_POSTGRES_DSN", f"sqlite:///{tmp_path/'rl.db'}")
        monkeypatch.setenv("GEOLYTICS_ENV", "test")
        monkeypatch.setenv("GEOLYTICS_SECRET_KEY", "r" * 48)
        monkeypatch.setenv("GEOLYTICS_REDIS_URL", REDIS_URL)
        monkeypatch.setenv("GEOLYTICS_RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("GEOLYTICS_AUTH_RATE_LIMIT_PER_MINUTE", "5")
        monkeypatch.setenv("GEOLYTICS_STRIPE_WEBHOOK_SECRET", "whsec_integration_secret_value")

        from geolytics.config import get_settings
        from geolytics.db.models import Base
        from geolytics.db.session import get_engine, get_session_factory

        caches = (get_settings, get_engine, get_session_factory)
        for cached in caches:
            cached.cache_clear()
        Base.metadata.create_all(get_engine())

        import redis as redis_module

        redis_module.Redis.from_url(REDIS_URL).flushdb()

        from geolytics.api.app import create_app

        with TestClient(create_app()) as c:
            yield c
        for cached in caches:
            cached.cache_clear()

    def test_auth_endpoints_are_limited_tightly(self, client):
        """Credential stuffing lands on /auth/login, so its bucket is small."""
        codes = [
            client.post(
                "/auth/login",
                json={"email": "nobody@example-domain.co", "password": "wrong-password-x"},
            ).status_code
            for _ in range(20)
        ]
        assert 429 in codes, "auth endpoint was never limited"
        limited = client.post(
            "/auth/login",
            json={"email": "nobody@example-domain.co", "password": "wrong-password-x"},
        )
        assert limited.status_code == 429
        assert limited.json()["error"] == "rate_limited"
        assert "Retry-After" in limited.headers

    def test_health_is_never_limited(self, client):
        """A limited health check would make a load balancer pull the instance."""
        for _ in range(60):
            client.post(
                "/auth/login",
                json={"email": "nobody@example-domain.co", "password": "wrong-password-x"},
            )
        assert client.get("/health/live").status_code == 204

    def test_successful_responses_carry_limit_headers(self, client):
        response = client.get("/org/plans")
        assert "X-RateLimit-Limit" in response.headers


@requires_postgres
class TestWebhookEndpoint:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEOLYTICS_POSTGRES_DSN", f"sqlite:///{tmp_path/'wh.db'}")
        monkeypatch.setenv("GEOLYTICS_ENV", "test")
        monkeypatch.setenv("GEOLYTICS_SECRET_KEY", "w" * 48)
        monkeypatch.setenv("GEOLYTICS_RATE_LIMIT_ENABLED", "false")
        monkeypatch.setenv("GEOLYTICS_STRIPE_WEBHOOK_SECRET", "whsec_integration_secret_value")
        monkeypatch.setenv(
            "GEOLYTICS_STRIPE_PRICE_IDS", json.dumps({"growth": "price_growth_1"})
        )

        from geolytics.config import get_settings
        from geolytics.db.models import Base
        from geolytics.db.session import get_engine, get_session_factory

        caches = (get_settings, get_engine, get_session_factory)
        for cached in caches:
            cached.cache_clear()
        Base.metadata.create_all(get_engine())

        from geolytics.api.app import create_app

        with TestClient(create_app()) as c:
            yield c
        for cached in caches:
            cached.cache_clear()

    @pytest.fixture
    def org_id(self, client):
        from geolytics.db.models import Organization
        from geolytics.db.session import session_scope

        with session_scope() as session:
            org = Organization(slug="acme", name="Acme", plan="free")
            session.add(org)
            session.flush()
            return org.id

    def _event(self, org_id: int) -> bytes:
        return json.dumps(
            {
                "id": "evt_1",
                "type": "customer.subscription.created",
                "data": {
                    "object": {
                        "id": "sub_1",
                        "customer": "cus_1",
                        "status": "active",
                        "current_period_end": int(time.time()) + 86400,
                        "metadata": {"org_id": str(org_id)},
                        "items": {"data": [{"price": {"id": "price_growth_1"}}]},
                    }
                },
            }
        ).encode()

    def test_a_signed_event_upgrades_the_plan(self, client, org_id):
        from geolytics.billing import sign_webhook
        from geolytics.db.models import Organization
        from geolytics.db.session import session_scope

        body = self._event(org_id)
        response = client.post(
            "/billing/webhook",
            content=body,
            headers={"Stripe-Signature": sign_webhook(body, "whsec_integration_secret_value")},
        )
        assert response.status_code == 200

        with session_scope() as session:
            assert session.get(Organization, org_id).plan == "growth"

    def test_an_unsigned_event_changes_nothing(self, client, org_id):
        """Without signature verification this endpoint is a free upgrade button."""
        from geolytics.db.models import Organization
        from geolytics.db.session import session_scope

        response = client.post("/billing/webhook", content=self._event(org_id))
        assert response.status_code == 400

        with session_scope() as session:
            assert session.get(Organization, org_id).plan == "free"

    def test_a_forged_signature_changes_nothing(self, client, org_id):
        from geolytics.billing import sign_webhook
        from geolytics.db.models import Organization
        from geolytics.db.session import session_scope

        body = self._event(org_id)
        response = client.post(
            "/billing/webhook",
            content=body,
            headers={"Stripe-Signature": sign_webhook(body, "whsec_attacker_guess")},
        )
        assert response.status_code == 400

        with session_scope() as session:
            assert session.get(Organization, org_id).plan == "free"

    def test_delivering_the_same_event_twice_is_harmless(self, client, org_id):
        from geolytics.billing import sign_webhook
        from geolytics.db.models import Organization
        from geolytics.db.session import session_scope

        body = self._event(org_id)
        headers = {"Stripe-Signature": sign_webhook(body, "whsec_integration_secret_value")}
        assert client.post("/billing/webhook", content=body, headers=headers).status_code == 200
        assert client.post("/billing/webhook", content=body, headers=headers).status_code == 200

        with session_scope() as session:
            assert session.get(Organization, org_id).plan == "growth"
