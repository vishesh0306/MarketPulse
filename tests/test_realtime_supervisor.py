"""Metrics rollups and the tail supervisor wiring (mock RSS transport + fakeredis)."""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import fakeredis.aioredis
import httpx

from src.realtime.dedup_store import RedisDedupStore
from src.realtime.metrics import Metrics
from src.realtime.supervisor import TailSupervisor, build_engine
from src.utils.config_loader import load_settings

_RSS_FIXTURE = (Path(__file__).parent / "fixtures" / "nitter_rss.xml").read_text(encoding="utf-8")


def _with_recent_dates(rss: str, newest_minutes_ago: int = 40) -> str:
    """Re-dates the fixture's items into a bucket that has closed but is still inside the
    backfill window.

    The committed fixture carries fixed pubDates so the RSS *parser* test can assert on
    exact timestamps. The supervisor is a different matter: `sealed_history()` drops
    anything older than `backfill_hours`, so fixed dates make these tests a time bomb —
    they pass until the fixture is more than 24h old, then the supervisor ingests nothing
    and they fail on a day nobody changed anything. They also fail *quietly* first: once
    the history empties out, the file-vs-history assertion just compares two empty lists
    and still passes. Anchor to the clock instead; what these tests care about is
    ingest/dedup/seal behaviour, not which day it is.

    40 minutes back puts every item in a 15-minute bucket that closed well before the seal
    grace period, so the buckets still seal on shutdown exactly as they did before.
    """
    newest = datetime.now(timezone.utc) - timedelta(minutes=newest_minutes_ago)
    stamps = (newest - timedelta(seconds=150 * i) for i in range(rss.count("<pubDate>")))
    # format_datetime is the exact inverse of the parsedate_to_datetime the client uses,
    # so this stays correct regardless of the machine's locale.
    return re.sub(r"<pubDate>[^<]*</pubDate>", lambda _m: f"<pubDate>{format_datetime(next(stamps))}</pubDate>", rss)


_RSS = _with_recent_dates(_RSS_FIXTURE)


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

    # The 3 parseable items are re-dated to a bucket that closed 40 minutes ago, so the
    # shutdown flush seals it and it stays inside the backfill window.
    assert captured, "expected at least one sealed bucket"
    assert any(r["hashtag"] == "nifty50" for r in captured)

    history = tmp_path / "sealed_signals.jsonl"
    assert history.exists()
    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    # Non-empty first: sealed_history() filters by backfill_hours, so if the fixture ages
    # out of the window this comparison silently degrades into [] == [] and passes.
    assert rows, "sealed history file should not be empty"
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


def test_warm_start_ingests_recent_processed_tweets(tmp_path: Path) -> None:
    import pandas as pd

    settings = _settings_fast()
    now = pd.Timestamp.now(tz="UTC")
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(
        [
            {"tweet_id": str(i), "source_hashtag": "nifty50", "username": f"u{i}",
             "text_normalized": "nifty breakout rally", "hashtags": ["nifty50"],
             "created_at": (now - pd.Timedelta(minutes=30)).isoformat()}
            for i in range(6)
        ]
        + [
            {"tweet_id": "old", "source_hashtag": "nifty50", "username": "old",
             "text_normalized": "ancient tweet", "hashtags": ["nifty50"],
             "created_at": (now - pd.Timedelta(days=5)).isoformat()}
        ]
    ).to_parquet(processed / "part-0.parquet")

    sup = TailSupervisor(
        settings,
        RedisDedupStore(fakeredis.aioredis.FakeRedis(decode_responses=True)),
        build_engine(settings),
        output_dir=tmp_path,
    )
    ingested = sup.warm_start(processed_dir=processed)
    assert ingested == 6  # the 5-day-old tweet is outside backfill_hours


def test_warm_start_missing_processed_dir_is_a_noop(tmp_path: Path) -> None:
    sup = TailSupervisor(
        _settings_fast(),
        RedisDedupStore(fakeredis.aioredis.FakeRedis(decode_responses=True)),
        build_engine(_settings_fast()),
        output_dir=tmp_path,
    )
    assert sup.warm_start(processed_dir=tmp_path / "nope") == 0


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
