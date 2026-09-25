"""Helpers for building an authenticated test client against SQLite.

Kept out of conftest.py so the integration package can import it by a unique
name alongside its own fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient


@dataclass
class Tenant:
    """A signed-up organisation with a token, for isolation tests."""

    client: TestClient
    email: str
    password: str
    org_slug: str
    org_id: int
    access_token: str
    refresh_token: str
    user_id: int

    @property
    def auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    def get(self, path: str, **kw: Any):
        return self.client.get(path, headers=self.auth, **kw)

    def post(self, path: str, json: Any = None, **kw: Any):
        return self.client.post(path, json=json, headers=self.auth, **kw)

    def patch(self, path: str, json: Any = None, **kw: Any):
        return self.client.patch(path, json=json, headers=self.auth, **kw)

    def delete(self, path: str, **kw: Any):
        return self.client.delete(path, headers=self.auth, **kw)


def signup(
    client: TestClient,
    email: str,
    org_name: str,
    password: str = "correct-horse-staple-9",
) -> Tenant:
    response = client.post(
        "/auth/signup",
        json={"email": email, "password": password, "organization_name": org_name},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    me = client.get(
        "/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"}
    ).json()
    return Tenant(
        client=client,
        email=email,
        password=password,
        org_slug=body["organization"]["slug"],
        org_id=body["organization"]["id"],
        access_token=body["access_token"],
        refresh_token=body["refresh_token"],
        user_id=me["id"],
    )
