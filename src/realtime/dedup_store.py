"""Redis-backed dedup for the real-time tail loop.

The batch pipeline dedups in-process with Python sets, which is fine for a run that
starts, finishes, and exits. A long-lived tail loop needs dedup state that (a) is bounded
regardless of uptime and (b) survives a restart so the loop doesn't re-ingest everything
it already saw. Both are handled by storing each id and content hash as its own Redis key
with a TTL: membership is a single `SET ... NX` round trip, and expiry bounds the
key count to roughly one lookback window's worth of tweets.
"""

from __future__ import annotations

from redis.asyncio import Redis

from src.processing.deduplicator import content_hash


class RedisDedupStore:
    """Async dedup over a Redis connection. `check_and_record` is the only method the tail
    loop needs: it reports whether a tweet has been seen (by exact id or by near-duplicate
    content) and records it if not, in one call."""

    def __init__(
        self,
        redis: Redis,
        *,
        id_prefix: str = "marketpulse:id:",
        content_prefix: str = "marketpulse:content:",
        ttl_seconds: int = 172_800,
    ) -> None:
        self._redis = redis
        self._id_prefix = id_prefix
        self._content_prefix = content_prefix
        self._ttl = ttl_seconds

    @staticmethod
    def content_key_fields(record: dict[str, object]) -> list[str]:
        """The fields whose hash defines a near-duplicate in the stream: same text from the
        same author, regardless of which hashtag search surfaced it."""
        return [str(record.get("text_normalized") or record.get("text", "")), str(record.get("username", ""))]

    async def check_and_record(self, tweet_id: str, content_digest: str) -> bool:
        """True if this tweet was already seen (exact id or near-duplicate content).
        Records the id and content hash when it's new."""
        id_is_new = await self._redis.set(f"{self._id_prefix}{tweet_id}", "1", ex=self._ttl, nx=True)
        if not id_is_new:
            return True
        content_is_new = await self._redis.set(
            f"{self._content_prefix}{content_digest}", tweet_id, ex=self._ttl, nx=True
        )
        return not content_is_new

    async def is_duplicate(self, record: dict[str, object]) -> bool:
        """Convenience wrapper: derive the content hash from a tweet record and dedup it."""
        digest = content_hash(self.content_key_fields(record))
        return await self.check_and_record(str(record["tweet_id"]), digest)

    async def count(self) -> int:
        """Approximate number of distinct tweet ids currently retained (for /metrics)."""
        return len([key async for key in self._redis.scan_iter(match=f"{self._id_prefix}*", count=500)])
