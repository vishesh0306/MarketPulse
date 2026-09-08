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
        """Fields whose hash defines a near-duplicate — same as the batch deduplicator
        (text + author + source_hashtag), so a tweet legitimately found under two hashtag
        searches counts once per hashtag, not once overall."""
        return [
            str(record.get("text_normalized") or record.get("text", "")),
            str(record.get("username", "")),
            str(record.get("source_hashtag", "")),
        ]

    async def check_and_record(self, dedup_key: str, content_digest: str) -> bool:
        """True if this tweet was already seen (exact key or near-duplicate content).
        Records both when it's new."""
        id_is_new = await self._redis.set(f"{self._id_prefix}{dedup_key}", "1", ex=self._ttl, nx=True)
        if not id_is_new:
            return True
        content_is_new = await self._redis.set(
            f"{self._content_prefix}{content_digest}", dedup_key, ex=self._ttl, nx=True
        )
        return not content_is_new

    async def is_duplicate(self, record: dict[str, object]) -> bool:
        """Convenience wrapper: dedup a tweet record on (tweet_id, source_hashtag) plus a
        near-duplicate content hash."""
        key = f"{record['tweet_id']}:{record.get('source_hashtag', '')}"
        digest = content_hash(self.content_key_fields(record))
        return await self.check_and_record(key, digest)

    async def count(self) -> int:
        """Approximate number of distinct tweet ids currently retained (for /metrics)."""
        return len([key async for key in self._redis.scan_iter(match=f"{self._id_prefix}*", count=500)])
