"""Password hashing.

Argon2id, the algorithm the OWASP Password Storage Cheat Sheet recommends
first. It is memory-hard, so a GPU or ASIC attacker gains far less against it
than against bcrypt or PBKDF2, and the library handles per-password salting.

Parameters follow OWASP's baseline (19 MiB, 2 iterations, 1 degree of
parallelism). They are named here rather than left implicit so that raising
them later is a deliberate, reviewable change -- and `needs_rehash` lets
existing users migrate transparently on their next successful login.
"""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

# OWASP-recommended Argon2id baseline.
_HASHER = PasswordHasher(
    time_cost=2,
    memory_cost=19 * 1024,  # KiB
    parallelism=1,
    hash_len=32,
    salt_len=16,
)

MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 1024


class PasswordPolicyError(ValueError):
    """A password was rejected before hashing."""


def validate_password(password: str) -> None:
    """Check length bounds.

    The upper bound is a denial-of-service guard, not a strength rule: Argon2
    is deliberately expensive, so hashing a multi-megabyte string submitted to
    an unauthenticated signup endpoint is a cheap way to burn a worker.

    Only length is enforced. Composition rules ("one digit, one symbol") push
    people toward predictable substitutions and are no longer recommended;
    length plus a breached-password check is the modern guidance.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordPolicyError(
            f"password must be at least {MIN_PASSWORD_LENGTH} characters"
        )
    if len(password) > MAX_PASSWORD_LENGTH:
        raise PasswordPolicyError(
            f"password must be at most {MAX_PASSWORD_LENGTH} characters"
        )


def hash_password(password: str) -> str:
    """Hash a password for storage. The salt is embedded in the returned string."""
    validate_password(password)
    return _HASHER.hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    """Whether `password` matches `stored_hash`.

    Returns False rather than raising on a mismatch or a corrupt stored hash,
    so callers cannot accidentally distinguish "wrong password" from "bad row"
    in a way that leaks through timing or error handling.
    """
    try:
        return _HASHER.verify(stored_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False
    except Exception:  # noqa: BLE001 - never let a hashing fault authenticate
        return False


def needs_rehash(stored_hash: str) -> bool:
    """Whether a stored hash used weaker parameters than the current policy."""
    try:
        return _HASHER.check_needs_rehash(stored_hash)
    except InvalidHashError:
        return True
