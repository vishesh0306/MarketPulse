"""Metrics rollups and the tail supervisor wiring (mock RSS transport + fakeredis)."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import fakeredis.aioredis
import httpx

from src.realtime.dedup_store import RedisDedupStore
from src.realtime.metrics import Metrics
from src.realtime.supervisor import TailSupervisor, build_engine
from src.utils.config_loader import load_settings

_RSS = (Path(__file__).parent / "fixtures" / "nitter_rss.xml").read_text(encoding="utf-8")


# ---- metrics ---------------------------------------------------------------


def test_metrics_snapshot_shape_and_dedup_rate() -> None:
    m = Metrics()
    m.record_dedup(was_duplicate=False)
    m.record_dedup(was_duplicate=True)
    m.record_dedup(was_duplicate=True)
    m.record_poll("good.host", ok=True)
    m.record_poll("bad.host", ok=False)
    m.record_ingest("nifty50", time.time())
    m.record_seal("nifty50", 2)

    snap = m.snapshot()
    assert snap["dedup_hit_rate"] == round(2 / 3, 3)
    assert snap["poll_success_rate_by_host"] == {"good.host": 1.0, "bad.host": 0.0}
    assert snap["sealed_buckets_total"] == 2
    assert "p50" in snap["ingest_lag_seconds"]
    assert snap["tweets_per_minute"]["by_hashtag"]["nifty50"] > 0


# ---- supervisor -----------------------------------------------------------


def _settings_fast():
    s = load_settings().model_copy(deep=True)
    s.realtime.rss.poll_interval_seconds = 0.05
    s.analysis.min_bucket_tweets = 1  # let the 3 fixture tweets produce a real signal
    return s


def _mock_client(handler=None) -> httpx.AsyncClient:
    handler = handler or (
        lambda req: httpx.Response(200, text=_RSS, headers={"content-type": "application/rss+xml"})
    )
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_supervisor_ingests_seals_and_writes_history(tmp_path: Path) -> None:
    settings = _settings_fast()
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    dedup = RedisDedupStore(redis, ttl_seconds=3600)
    engine = build_engine(settings)

    captured: list[dict[str, object]] = []

    async def sink(rows: list[dict[str, object]]) -> None:
        captured.extend(rows)

    sup = TailSupervisor(
        settings, dedup, engine, seal_sinks=[sink], output_dir=tmp_path, metrics=Metrics()
    )

    async with _mock_client() as client:
        task = asyncio.create_task(sup.run(["nifty50"], client=client))
        await asyncio.sleep(0.4)
        sup.stop()
        await asyncio.wait_for(task, timeout=5)

    # fixture has 3 parseable items for one 06:15 bucket; today's clock is well past its
    # seal time, so the shutdown flush seals it.
    assert captured, "expected at least one sealed bucket"
    assert any(r["hashtag"] == "nifty50" for r in captured)

    history = tmp_path / "sealed_signals.jsonl"
    assert history.exists()
    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    assert rows == sup.sealed_history()

    snap = sup.metrics.snapshot()
    assert snap["poll_success_rate_by_host"]  # at least one poll recorded
    assert snap["sealed_buckets_total"] >= 1


async def test_supervisor_dedups_repeated_polls(tmp_path: Path) -> None:
    settings = _settings_fast()
    dedup = RedisDedupStore(fakeredis.aioredis.FakeRedis(decode_responses=True), ttl_seconds=3600)
    engine = build_engine(settings)
    metrics = Metrics()
    sup = TailSupervisor(settings, dedup, engine, output_dir=tmp_path, metrics=metrics)

    async with _mock_client() as client:
        task = asyncio.create_task(sup.run(["nifty50"], client=client))
        await asyncio.sleep(0.4)  # many polls of the same 3-item feed
        sup.stop()
        await asyncio.wait_for(task, timeout=5)

    # The per-hashtag high-water mark plus the Redis store mean the same 3-item feed,
    # polled many times over 0.4s, is ingested exactly once — not once per poll.
    history = sup.sealed_history()
    assert {r["hashtag"] for r in history} == {"nifty50"}
    assert sum(int(r["tweet_count"]) for r in history) == 3


async def test_enqueue_drop_oldest_policy(tmp_path: Path) -> None:
    settings = _settings_fast()
    settings.realtime.queue_maxsize = 2
    settings.realtime.queue_full_policy = "drop_oldest"
    sup = TailSupervisor(
        settings,
        RedisDedupStore(fakeredis.aioredis.FakeRedis(decode_responses=True)),
        build_engine(settings),
        output_dir=tmp_path,
    )
    for i in range(5):
        await sup._enqueue({"tweet_id": str(i)})
    assert sup._queue.qsize() == 2
    assert sup.metrics.snapshot()["queue_drops_total"] == 3
