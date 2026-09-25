"""API keys for machine access.

Shape: ``gk_live_<22 chars>.<43 chars>`` — an identifying prefix, a public
lookup id, and a secret.

Why three parts:

* The **environment prefix** (`gk_live_` / `gk_test_`) makes a leaked key
  recognisable in logs and lets secret scanners match it.
* The **lookup id** is stored in clear and indexed, so verification is one
  indexed row fetch rather than a hash comparison against every key in the
  table.
* The **secret** is never stored. Only its hash is, so a database disclosure
  does not hand over working credentials.

The secret is hashed with SHA-256 rather than Argon2. That is deliberate and
is the opposite of the choice made for passwords: an API key is 256 bits of
machine-generated entropy, so there is no dictionary to attack and no benefit
from a slow hash — whereas a slow hash on every API request would be a
self-inflicted denial of service. Passwords are low-entropy and human-chosen,
which is why they get Argon2.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Literal

Environment = Literal["live", "test"]

API_KEY_PREFIXES: dict[Environment, str] = {"live": "gk_live_", "test": "gk_test_"}

_LOOKUP_BYTES = 16   # -> 22 urlsafe characters
_SECRET_BYTES = 32   # -> 43 urlsafe characters


@dataclass(frozen=True, slots=True)
class GeneratedApiKey:
    """A freshly minted key. `token` is shown once and never recoverable."""

    token: str
    lookup_id: str
    secret_hash: str
    prefix: str

    @property
    def display_hint(self) -> str:
        """A safe fragment for the UI, e.g. ``gk_live_abc123…f0``."""
        return f"{self.prefix}{self.lookup_id[:6]}…{self.token[-2:]}"


@dataclass(frozen=True, slots=True)
class ParsedApiKey:
    environment: Environment
    lookup_id: str
    secret: str


def generate_api_key(environment: Environment = "live") -> GeneratedApiKey:
    """Mint a key. The caller must show `token` to the user immediately."""
    if environment not in API_KEY_PREFIXES:
        raise ValueError(f"unknown environment {environment!r}")
    prefix = API_KEY_PREFIXES[environment]
    lookup_id = secrets.token_urlsafe(_LOOKUP_BYTES)
    secret = secrets.token_urlsafe(_SECRET_BYTES)
    return GeneratedApiKey(
        token=f"{prefix}{lookup_id}.{secret}",
        lookup_id=lookup_id,
        secret_hash=hash_api_key(secret),
        prefix=prefix,
    )


def parse_api_key(token: str) -> ParsedApiKey | None:
    """Split a presented key. Returns None if it is not shaped like one."""
    if not isinstance(token, str):
        return None
    for environment, prefix in API_KEY_PREFIXES.items():
        if not token.startswith(prefix):
            continue
        body = token[len(prefix) :]
        lookup_id, separator, secret = body.partition(".")
        if not separator or not lookup_id or not secret:
            return None
        return ParsedApiKey(environment=environment, lookup_id=lookup_id, secret=secret)
    return None


def hash_api_key(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def verify_api_key(stored_hash: str, secret: str) -> bool:
    """Constant-time comparison, so a timing signal cannot reveal the secret."""
    return hmac.compare_digest(stored_hash, hash_api_key(secret))
