"""Token-bucket rate limiter plus randomized inter-action delay and exponential backoff."""

from __future__ import annotations


class RateLimitedError(Exception):
    """Raised when a soft-block/rate-limit signal is detected on the page."""


class TokenBucketRateLimiter:
    """Bounds action rate via a token bucket; callers await a token before each scroll/request."""

    def __init__(self, capacity: int, refill_rate_per_second: float) -> None:
        raise NotImplementedError

    def acquire(self) -> None:
        """Blocks (with randomized jitter) until a token is available."""
        raise NotImplementedError

    def backoff(self, attempt: int, base_seconds: float, max_seconds: float, multiplier: float) -> float:
        """Returns the sleep duration for the given retry attempt (exponential, capped)."""
        raise NotImplementedError
