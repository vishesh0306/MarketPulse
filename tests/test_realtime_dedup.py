"""Redis-backed dedup store, exercised against fakeredis (no server needed)."""

from __future__ import annotations

import fakeredis.aioredis
import pytest

from src.realtime.dedup_store import RedisDedupStore


@pytest.fixture
def store() -> RedisDedupStore:
    return RedisDedupStore(fakeredis.aioredis.FakeRedis(decode_responses=True), ttl_seconds=3600)


async def test_first_sighting_is_not_a_duplicate(store: RedisDedupStore) -> None:
    assert await store.check_and_record("t1", "hash-a") is False


async def test_same_id_is_a_duplicate(store: RedisDedupStore) -> None:
    await store.check_and_record("t1", "hash-a")
    assert await store.check_and_record("t1", "hash-a") is True


async def test_same_content_different_id_is_a_duplicate(store: RedisDedupStore) -> None:
    await store.check_and_record("t1", "hash-a")
    assert await store.check_and_record("t2", "hash-a") is True


async def test_different_content_and_id_passes(store: RedisDedupStore) -> None:
    await store.check_and_record("t1", "hash-a")
    assert await store.check_and_record("t2", "hash-b") is False


async def test_is_duplicate_derives_content_hash_from_record(store: RedisDedupStore) -> None:
    first = {"tweet_id": "100", "text_normalized": "nifty breakout confirmed", "username": "trader_a"}
    reposted = {"tweet_id": "101", "text_normalized": "nifty breakout confirmed", "username": "trader_a"}
    genuinely_new = {"tweet_id": "102", "text_normalized": "sensex hits record", "username": "trader_a"}

    assert await store.is_duplicate(first) is False
    assert await store.is_duplicate(reposted) is True  # same text + author
    assert await store.is_duplicate(genuinely_new) is False


async def test_count_tracks_distinct_ids(store: RedisDedupStore) -> None:
    for i in range(5):
        await store.check_and_record(f"t{i}", f"hash-{i}")
    assert await store.count() == 5


async def test_ttl_is_applied_to_recorded_keys(store: RedisDedupStore) -> None:
    await store.check_and_record("t1", "hash-a")
    ttl = await store._redis.ttl("marketpulse:id:t1")
    assert 0 < ttl <= 3600
