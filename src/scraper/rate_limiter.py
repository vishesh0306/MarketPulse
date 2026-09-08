"""Token-bucket rate limiter plus randomized inter-action delay and exponential backoff."""

from __future__ import annotations

import random
import time


class RateLimitedError(Exception):
    """Raised when a soft-block/rate-limit signal is detected on the page."""


class TokenBucketRateLimiter:
    """Bounds action rate via a token bucket; callers wait for a token before each page request."""

    def __init__(self, capacity: int, refill_rate_per_second: float) -> None:
        self.capacity = capacity
        self.refill_rate_per_second = refill_rate_per_second
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate_per_second)
        self._last_refill = now

    def acquire(self, min_delay: float = 1.5, max_delay: float = 4.5) -> None:
        """Blocks (with randomized jitter) until a token is available, then consumes one."""
        self._refill()
        if self._tokens < 1:
            wait_seconds = (1 - self._tokens) / self.refill_rate_per_second
            time.sleep(wait_seconds)
            self._refill()
        self._tokens -= 1
        time.sleep(random.uniform(min_delay, max_delay))

    def backoff(self, attempt: int, base_seconds: float, max_seconds: float, multiplier: float) -> float:
        """Sleeps for an exponentially increasing, jittered duration and returns the sleep time."""
        delay = min(max_seconds, base_seconds * (multiplier**attempt))
        jittered = delay * random.uniform(0.8, 1.2)
        time.sleep(jittered)
        return jittered
