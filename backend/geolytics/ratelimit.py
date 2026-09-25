"""Redis token-bucket rate limiting.

A token bucket rather than a fixed window: a fixed window lets a caller spend
its whole allowance at 11:59:59 and the next one at 12:00:00, delivering twice
the intended rate across the boundary. A bucket refills continuously, so the
long-run rate is the configured rate and `burst` states exactly how much
short-term spikiness is tolerated.

The refill-and-take step runs as a Lua script so it is atomic on the server.
Doing it as GET, compute, SET from the application would let two concurrent
requests both read the same token count and both spend it.

Failure policy is fail-open: if Redis is unreachable, requests are allowed.
For this product that is the right trade -- Redis being down should not take
the whole API down with it -- but it does mean rate limiting is unavailable
exactly when the system is already unhealthy, so the limiter reports the
degraded state rather than hiding it.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any

# KEYS[1] = bucket key
# ARGV: 1 capacity, 2 refill tokens/sec, 3 now (float secs), 4 cost, 5 ttl secs
_TAKE_TOKENS = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

local state = redis.call('HMGET', key, 'tokens', 'updated')
local tokens = tonumber(state[1])
local updated = tonumber(state[2])

if tokens == nil then
  tokens = capacity
  updated = now
end

-- Refill for the time elapsed, never above capacity.
local elapsed = math.max(0, now - updated)
tokens = math.min(capacity, tokens + elapsed * rate)

local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end

redis.call('HMSET', key, 'tokens', tokens, 'updated', now)
redis.call('EXPIRE', key, ttl)

local deficit = 0
if allowed == 0 then
  deficit = cost - tokens
end
return {allowed, tostring(tokens), tostring(deficit)}
"""


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    """Outcome of one limiter decision."""

    allowed: bool
    remaining: int
    limit: int
    retry_after: int
    degraded: bool = False

    def headers(self) -> dict[str, str]:
        """Standard response headers so clients can back off politely."""
        out = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
        }
        if not self.allowed:
            out["Retry-After"] = str(max(1, self.retry_after))
        return out


class RateLimiter:
    """Token bucket over Redis, keyed per caller."""

    def __init__(
        self,
        redis: Any,
        per_minute: int = 120,
        burst: int = 30,
        namespace: str = "rl",
    ) -> None:
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.redis = redis
        self.per_minute = per_minute
        self.burst = max(burst, 1)
        self.namespace = namespace
        self._script = None

    @property
    def capacity(self) -> int:
        return self.per_minute + self.burst

    @property
    def refill_per_second(self) -> float:
        return self.per_minute / 60.0

    def _load(self) -> Any:
        if self._script is None:
            self._script = self.redis.register_script(_TAKE_TOKENS)
        return self._script

    def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Spend `cost` tokens for `key`."""
        bucket = f"{self.namespace}:{key}"
        # Two refill periods of idle time is long enough for any bucket to be
        # full again, so keeping it after that is wasted memory.
        ttl = int(self.capacity / self.refill_per_second) + 60

        try:
            allowed, tokens, deficit = self._load()(
                keys=[bucket],
                args=[self.capacity, self.refill_per_second, time.time(), cost, ttl],
            )
        except Exception:  # noqa: BLE001 - Redis down must not take the API down
            return RateLimitResult(
                allowed=True,
                remaining=self.capacity,
                limit=self.per_minute,
                retry_after=0,
                degraded=True,
            )

        remaining = int(float(tokens))
        retry_after = 0
        if not int(allowed):
            retry_after = max(1, int(float(deficit) / self.refill_per_second) + 1)

        return RateLimitResult(
            allowed=bool(int(allowed)),
            remaining=remaining,
            limit=self.per_minute,
            retry_after=retry_after,
        )

    def reset(self, key: str) -> None:
        with contextlib.suppress(Exception):
            self.redis.delete(f"{self.namespace}:{key}")


def build_limiter(
    redis_url: str, per_minute: int, burst: int, namespace: str = "rl"
) -> RateLimiter:
    import redis as redis_module

    client = redis_module.Redis.from_url(redis_url, decode_responses=True)
    return RateLimiter(client, per_minute=per_minute, burst=burst, namespace=namespace)
