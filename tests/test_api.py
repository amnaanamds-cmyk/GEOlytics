"""HTTP API: authentication, authorisation, tenant isolation and quotas.

Runs against SQLite so the suite needs no service. The models use portable
column types, so the schema round-trips; only PostgreSQL-specific behaviour
would need a real database.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest_api import signup

PASSWORD = "correct-horse-staple-9"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOLYTICS_POSTGRES_DSN", f"sqlite:///{tmp_path/'test.db'}")
    monkeypatch.setenv("GEOLYTICS_ENV", "test")
    monkeypatch.setenv("GEOLYTICS_SECRET_KEY", "t" * 48)
    # Rate limiting has its own tests against a real Redis; leaving it on here
    # would make every other test share one bucket.
    monkeypatch.setenv("GEOLYTICS_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("GEOLYTICS_CRAWL_ALLOW_PRIVATE_ADDRESSES", "true")
    monkeypatch.setenv("GEOLYTICS_CRAWL_RESTRICT_PORTS", "false")

    from geolytics.config import get_settings
    from geolytics.db.session import get_engine, get_session_factory

    caches = (get_settings, get_engine, get_session_factory)
    for cached in caches:
        cached.cache_clear()

    from geolytics.db.models import Base

    Base.metadata.create_all(get_engine())

    from geolytics.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client

    for cached in caches:
        cached.cache_clear()


@pytest.fixture
def acme(client):
    return signup(client, "owner@acme-plumbing.co", "Acme Plumbing")


def set_plan(org_id: int, plan: str) -> None:
    """Move an organisation onto a plan, as a successful payment would."""
    from geolytics.db.models import Organization
    from geolytics.db.session import session_scope

    with session_scope() as session:
        session.get(Organization, org_id).plan = plan


def complete(audit_id: int) -> None:
    """Mark an audit finished, so it stops counting against concurrency."""
    from geolytics.db.models import Audit
    from geolytics.db.session import session_scope

    with session_scope() as session:
        session.get(Audit, audit_id).status = "complete"


@pytest.fixture
def globex(client):
    return signup(client, "owner@globex-industrial.co", "Globex Industrial")


class TestSignup:
    def test_creates_user_org_and_tokens(self, client):
        response = client.post(
            "/auth/signup",
            json={
                "email": "New@Acme-Plumbing.co",
                "password": PASSWORD,
                "organization_name": "Acme Plumbing",
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["organization"]["role"] == "owner"
        assert body["organization"]["plan"] == "free"
        assert body["organization"]["slug"] == "acme-plumbing"

    def test_email_is_stored_lowercased(self, client):
        client.post(
            "/auth/signup",
            json={"email": "MiXeD@Acme-Plumbing.co", "password": PASSWORD,
                  "organization_name": "Acme"},
        )
        login = client.post(
            "/auth/login", json={"email": "mixed@acme-plumbing.co", "password": PASSWORD}
        )
        assert login.status_code == 200

    def test_duplicate_email_is_refused_without_confirming_it_exists(self, client, acme):
        response = client.post(
            "/auth/signup",
            json={"email": acme.email, "password": PASSWORD, "organization_name": "Other"},
        )
        assert response.status_code == 409
        # Must not say "that email is registered".
        assert "email" not in response.json()["detail"].lower()

    def test_short_password_is_refused(self, client):
        response = client.post(
            "/auth/signup",
            json={"email": "a@b-corp.co", "password": "short", "organization_name": "X"},
        )
        assert response.status_code == 422

    def test_validation_errors_do_not_echo_the_password(self, client):
        response = client.post(
            "/auth/signup",
            json={
                "email": "not-an-email",
                "password": "sup3rs3cr3tvalue",
                "organization_name": "X",
            },
        )
        assert response.status_code == 422
        assert "sup3rs3cr3tvalue" not in response.text

    def test_second_org_gets_a_distinct_slug(self, client, acme):
        second = signup(client, "other@acme-plumbing.co", "Acme Plumbing")
        assert second.org_slug != acme.org_slug


class TestLogin:
    def test_returns_a_token_pair(self, client, acme):
        response = client.post(
            "/auth/login", json={"email": acme.email, "password": PASSWORD}
        )
        assert response.status_code == 200
        assert response.json()["access_token"]

    def test_wrong_password_and_unknown_user_give_the_same_answer(self, client, acme):
        wrong = client.post(
            "/auth/login", json={"email": acme.email, "password": "wrong-password-x"}
        )
        missing = client.post(
            "/auth/login", json={"email": "nobody@nowhere-at-all.co", "password": PASSWORD}
        )
        assert wrong.status_code == missing.status_code == 401
        assert wrong.json()["detail"] == missing.json()["detail"]

    def test_me_returns_the_account(self, acme):
        body = acme.get("/auth/me").json()
        assert body["email"] == acme.email
        assert body["organizations"][0]["role"] == "owner"


class TestTokens:
    def test_refresh_returns_a_new_pair(self, client, acme):
        response = client.post("/auth/refresh", json={"refresh_token": acme.refresh_token})
        assert response.status_code == 200
        assert response.json()["access_token"]

    def test_an_access_token_is_not_accepted_as_a_refresh_token(self, client, acme):
        response = client.post("/auth/refresh", json={"refresh_token": acme.access_token})
        assert response.status_code == 401

    def test_a_refresh_token_is_not_accepted_as_a_bearer_credential(self, client, acme):
        response = client.get(
            "/auth/me", headers={"Authorization": f"Bearer {acme.refresh_token}"}
        )
        assert response.status_code == 401

    def test_tampered_token_is_refused(self, client, acme):
        response = client.get(
            "/auth/me", headers={"Authorization": f"Bearer {acme.access_token[:-2]}xy"}
        )
        assert response.status_code == 401

    def test_missing_credential_is_401_with_a_challenge(self, client):
        response = client.get("/auth/me")
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers

    def test_changing_the_password_invalidates_existing_sessions(self, client, acme):
        changed = acme.post(
            "/auth/password",
            {"current_password": PASSWORD, "new_password": "a-brand-new-passphrase-1"},
        )
        assert changed.status_code == 204
        assert acme.get("/auth/me").status_code == 401
        assert client.post(
            "/auth/refresh", json={"refresh_token": acme.refresh_token}
        ).status_code == 401


class TestTenantIsolation:
    """The property the whole product depends on."""

    def _make_audit(self, tenant, url="http://127.0.0.1:9/index.html"):
        response = tenant.post("/audits", {"url": url, "max_pages": 3})
        assert response.status_code == 202, response.text
        return response.json()["audit_id"]

    def test_one_org_cannot_read_anothers_audit(self, acme, globex, monkeypatch):
        audit_id = self._make_audit(acme)
        # A foreign id must read as absent, not as forbidden: a 403 would
        # confirm the row exists.
        assert globex.get(f"/audits/{audit_id}").status_code == 404

    def test_one_org_cannot_delete_anothers_audit(self, acme, globex):
        audit_id = self._make_audit(acme)
        assert globex.delete(f"/audits/{audit_id}").status_code == 404
        assert acme.get(f"/audits/{audit_id}").status_code == 200

    def test_listing_shows_only_the_callers_audits(self, acme, globex):
        self._make_audit(acme)
        assert globex.get("/audits").json()["items"] == []
        assert len(acme.get("/audits").json()["items"]) == 1

    def test_one_org_cannot_read_anothers_experiment(self, acme, globex):
        from geolytics.db.models import ExperimentRun, QueryRecord
        from geolytics.db.session import session_scope

        with session_scope() as session:
            run = ExperimentRun(
                org_id=acme.org_id, experiment="secret", condition="c",
                aggregate={"ndcg@10": 0.5}, n_queries=1,
            )
            session.add(run)
            session.flush()
            session.add(
                QueryRecord(run_id=run.id, query_id="q1", query_text="t",
                            metrics={"ndcg@10": 0.5})
            )

        assert globex.get("/experiments/secret").status_code == 404
        assert globex.get("/experiments").json() == []
        assert acme.get("/experiments").json() == ["secret"]

    def test_one_org_cannot_revoke_anothers_api_key(self, acme, globex):
        created = acme.post("/org/keys", {"name": "ci"})
        assert created.status_code == 201
        key_id = created.json()["id"]
        assert globex.delete(f"/org/keys/{key_id}").status_code == 404

    def test_members_list_is_scoped(self, acme, globex):
        assert [m["email"] for m in acme.get("/org/members").json()] == [acme.email]
        assert [m["email"] for m in globex.get("/org/members").json()] == [globex.email]


class TestApiKeys:
    def test_key_authenticates_and_is_shown_once(self, acme):
        created = acme.post("/org/keys", {"name": "ci"})
        assert created.status_code == 201
        token = created.json()["token"]
        assert token.startswith("gk_live_")

        listed = acme.get("/org/keys").json()
        assert "token" not in listed[0]
        assert listed[0]["display_hint"].startswith("gk_live_")

        response = acme.client.get("/audits", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200

    def test_revoked_key_stops_working(self, acme):
        created = acme.post("/org/keys", {"name": "ci"}).json()
        token = created["token"]
        assert acme.client.get(
            "/audits", headers={"Authorization": f"Bearer {token}"}
        ).status_code == 200

        assert acme.delete(f"/org/keys/{created['id']}").status_code == 204
        assert acme.client.get(
            "/audits", headers={"Authorization": f"Bearer {token}"}
        ).status_code == 401

    def test_garbage_key_is_refused(self, acme):
        for bad in ("gk_live_nodot", "gk_live_aaa.bbb", "not-a-key-at-all"):
            response = acme.client.get("/audits", headers={"Authorization": f"Bearer {bad}"})
            assert response.status_code == 401, bad

    def test_a_viewer_cannot_mint_keys_at_all(self, client, acme):
        set_plan(acme.org_id, "growth")
        viewer = signup(client, "viewer@acme-plumbing.co", "Temp Org")
        acme.post("/org/members", {"email": viewer.email, "role": "viewer"})
        tokens = client.post(
            "/auth/login",
            json={
                "email": viewer.email,
                "password": PASSWORD,
                "organization_slug": acme.org_slug,
            },
        ).json()
        response = client.post(
            "/org/keys", json={"name": "x"},
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert response.status_code == 403

    def test_requested_scopes_are_intersected_not_trusted(self, acme):
        created = acme.post(
            "/org/keys", {"name": "narrow", "scopes": ["audits:read"]}
        ).json()
        assert created["scopes"] == ["audits:read"]
        token = created["token"]
        # Read works, write does not.
        assert acme.client.get(
            "/audits", headers={"Authorization": f"Bearer {token}"}
        ).status_code == 200
        blocked = acme.client.post(
            "/audits",
            json={"url": "http://127.0.0.1:9/x", "max_pages": 1},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert blocked.status_code == 403

    def test_api_keys_cannot_manage_members(self, acme):
        token = acme.post("/org/keys", {"name": "ci"}).json()["token"]
        response = acme.client.post(
            "/org/members",
            json={"email": "someone@acme-plumbing.co", "role": "admin"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403
        assert "API key" in response.json()["detail"]


class TestRolesAndScopes:
    def test_viewer_cannot_create_an_audit(self, client, acme):
        set_plan(acme.org_id, "growth")
        member = signup(client, "member@acme-plumbing.co", "Throwaway")
        acme.post("/org/members", {"email": member.email, "role": "viewer"})
        # Log in scoped to Acme.
        tokens = client.post(
            "/auth/login",
            json={"email": member.email, "password": PASSWORD, "organization_slug": acme.org_slug},
        ).json()
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}

        assert client.get("/audits", headers=headers).status_code == 200
        blocked = client.post(
            "/audits", json={"url": "http://127.0.0.1:9/x", "max_pages": 1}, headers=headers
        )
        assert blocked.status_code == 403

    def test_admin_cannot_promote_to_owner(self, client, acme):
        set_plan(acme.org_id, "growth")
        admin = signup(client, "admin@acme-plumbing.co", "Throwaway2")
        other = signup(client, "other@acme-plumbing.co", "Throwaway3")
        acme.post("/org/members", {"email": admin.email, "role": "admin"})

        tokens = client.post(
            "/auth/login",
            json={"email": admin.email, "password": PASSWORD, "organization_slug": acme.org_slug},
        ).json()
        response = client.post(
            "/org/members",
            json={"email": other.email, "role": "owner"},
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert response.status_code == 403

    def test_the_last_owner_cannot_be_removed(self, acme):
        response = acme.delete(f"/org/members/{acme.user_id}")
        assert response.status_code == 409
        assert "at least one owner" in response.json()["detail"]


class TestQuotas:
    def test_usage_reports_plan_limits(self, acme):
        body = acme.get("/org/usage").json()
        assert body["plan"] == "free"
        assert body["limits"]["audits_run"] == 5
        assert body["used"]["audits_run"] == 0

    def test_audit_pages_are_capped_to_the_plan(self, acme):
        response = acme.post("/audits", {"url": "http://127.0.0.1:9/x", "max_pages": 500})
        assert response.status_code == 202
        assert "limited to 25 pages" in response.json()["message"]

    def test_monthly_audit_limit_is_enforced(self, acme):
        # Free allows 5 audits and 1 concurrent, so each is completed before
        # the next is queued.

        # One URL throughout: Free allows a single site, and a new URL each
        # time would trip max_sites before max_audits_per_month.
        url = "http://127.0.0.1:9/index.html"
        for _ in range(5):
            response = acme.post("/audits", {"url": url, "max_pages": 1})
            assert response.status_code == 202, response.text
            complete(response.json()["audit_id"])

        blocked = acme.post("/audits", {"url": url, "max_pages": 1})
        assert blocked.status_code == 402
        assert blocked.json()["detail"]["limit"] == "max_audits_per_month"

    def test_concurrency_limit_is_enforced(self, acme):
        url = "http://127.0.0.1:9/index.html"
        first = acme.post("/audits", {"url": url, "max_pages": 1})
        assert first.status_code == 202
        second = acme.post("/audits", {"url": url, "max_pages": 1})
        assert second.status_code == 402
        assert second.json()["detail"]["limit"] == "max_concurrent_audits"

    def test_site_limit_is_enforced(self, acme):
        first = acme.post("/audits", {"url": "http://127.0.0.1:9/one", "max_pages": 1})
        complete(first.json()["audit_id"])

        second = acme.post("/audits", {"url": "http://127.0.0.2:9/two", "max_pages": 1})
        assert second.status_code == 402
        assert second.json()["detail"]["limit"] == "max_sites"

    def test_api_key_limit_is_enforced(self, acme):
        assert acme.post("/org/keys", {"name": "one"}).status_code == 201
        second = acme.post("/org/keys", {"name": "two"})
        assert second.status_code == 402


class TestUrlPolicyAtTheApi:
    @pytest.mark.parametrize(
        "url", ["file:///etc/passwd", "http://[::1]:80/", "https://169.254.169.254/"]
    )
    def test_dangerous_urls_are_refused_before_anything_is_queued(
        self, client, acme, url, monkeypatch
    ):
        monkeypatch.setenv("GEOLYTICS_CRAWL_ALLOW_PRIVATE_ADDRESSES", "false")
        from geolytics.config import get_settings

        get_settings.cache_clear()
        client.app.state.settings = get_settings()

        response = acme.post("/audits", {"url": url, "max_pages": 1})
        assert response.status_code == 422
        body = response.json()
        # A non-http scheme is refused by the URL type before the guard sees
        # it; a routable-looking but forbidden address reaches the guard.
        assert body.get("error") == "validation_error" or (
            body["detail"]["error"] == "url_not_allowed"
        )


class TestResponseShape:
    def test_every_response_carries_a_request_id(self, client):
        response = client.get("/health/live")
        assert response.headers["X-Request-ID"]

    def test_an_inbound_request_id_is_preserved(self, client):
        response = client.get("/health/live", headers={"X-Request-ID": "trace-me-123"})
        assert response.headers["X-Request-ID"] == "trace-me-123"

    def test_security_headers_are_present(self, client):
        headers = client.get("/health/live").headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"

    def test_liveness_does_not_touch_dependencies(self, client):
        assert client.get("/health/live").status_code == 204

    def test_plans_are_public(self, client):
        plans = client.get("/org/plans").json()
        assert {p["name"] for p in plans} == {"free", "starter", "growth", "enterprise"}
