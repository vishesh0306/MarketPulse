"""The tail loop: per-hashtag RSS collectors feed a bounded queue, one updater drains it
into the incremental signal engine, and a sealer flushes closed buckets to disk and to
subscribers.

Backfill stays the batch scraper's job (Selenium, cursor paging over 24h). This process
picks up from there and keeps the signal current, targeting an ingest lag of tens of
seconds.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.realtime.dedup_store import RedisDedupStore
from src.realtime.incremental_signal import IncrementalSignalEngine
from src.realtime.metrics import Metrics
from src.realtime.rss_client import fetch_hashtag_feed
from src.utils.config_loader import Settings
from src.utils.logger import get_logger

logger = get_logger("realtime_supervisor")

SealSink = Callable[[list[dict[str, object]]], Awaitable[None]]


class TailSupervisor:
    def __init__(
        self,
        settings: Settings,
        dedup: RedisDedupStore,
        engine: IncrementalSignalEngine,
        *,
        metrics: Metrics | None = None,
        seal_sinks: list[SealSink] | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self._settings = settings
        self._rt = settings.realtime
        self._dedup = dedup
        self._engine = engine
        self.metrics = metrics or Metrics()
        self._seal_sinks = seal_sinks or []
        self._output_dir = output_dir or Path(settings.storage.output_dir) / "realtime"
        self._queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=self._rt.queue_maxsize)
        self._sealed_rows: list[dict[str, object]] = []
        self._high_water: dict[str, int] = {}
        self._stop = asyncio.Event()

    # -- collectors -----------------------------------------------------------

    async def _enqueue(self, record: dict[str, object]) -> None:
        if self._rt.queue_full_policy == "block":
            await self._queue.put(record)
            return
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()  # drop_oldest
                self._queue.task_done()
            self.metrics.record_queue_drop()
            self._queue.put_nowait(record)

    async def _collect_hashtag(self, client: httpx.AsyncClient, hashtag: str) -> None:
        template = self._rt.rss.path_template
        hosts = self._settings.scraper.nitter_hosts
        while not self._stop.is_set():
            records, host = await fetch_hashtag_feed(
                client, hashtag, hosts, template, max_items=self._rt.rss.max_items_per_poll
            )
            self.metrics.record_poll(host, ok=bool(records))

            high = self._high_water.get(hashtag, 0)
            fresh = [r for r in records if int(str(r["tweet_id"])) > high]
            for record in fresh:
                is_dup = await self._dedup.is_duplicate(record)
                self.metrics.record_dedup(was_duplicate=is_dup)
                if is_dup:
                    continue
                await self._enqueue(record)
            if records:
                self._high_water[hashtag] = max(int(str(r["tweet_id"])) for r in records)

            self.metrics.set_queue_depth(self._queue.qsize())
            jitter = random.uniform(0, self._rt.rss.poll_interval_seconds * 0.25)
            await self._wait(self._rt.rss.poll_interval_seconds + jitter)

    # -- updater ------------------------------------------------------------

    async def _drain_queue(self) -> None:
        while not self._stop.is_set():
            try:
                record = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                self._engine.ingest(record)
                created = datetime.fromisoformat(str(record["created_at"])).timestamp()
                self.metrics.record_ingest(str(record["source_hashtag"]), created)
            except (KeyError, ValueError) as exc:
                logger.warning("dropped malformed record", extra={"extra_fields": {"error": str(exc)}})
            finally:
                self._queue.task_done()
                self.metrics.set_queue_depth(self._queue.qsize())

    # -- sealer -----------------------------------------------------------

    async def _seal_loop(self) -> None:
        while not self._stop.is_set():
            await self._wait(min(30.0, self._rt.rss.poll_interval_seconds))
            await self._flush_sealed(self._engine.seal_due())

    async def _flush_sealed(self, sealed: list[dict[str, object]]) -> None:
        if not sealed:
            return
        self._sealed_rows.extend(sealed)
        self._prune_history()
        self._write_history()
        for row in sealed:
            self.metrics.record_seal(str(row["hashtag"]))
        for sink in self._seal_sinks:
            with contextlib.suppress(Exception):
                await sink(sealed)
        logger.info("sealed buckets", extra={"extra_fields": {"count": len(sealed)}})

    def _prune_history(self) -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - self._rt.backfill_hours * 3600
        self._sealed_rows = [
            r for r in self._sealed_rows if datetime.fromisoformat(str(r["bucket_start"])).timestamp() >= cutoff
        ]

    def _write_history(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._output_dir / "sealed_signals.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            for row in self._sealed_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(self._output_dir / "sealed_signals.jsonl")

    # -- public -----------------------------------------------------------

    def sealed_history(self) -> list[dict[str, object]]:
        return list(self._sealed_rows)

    def live_snapshot(self) -> list[dict[str, object]]:
        return self._engine.snapshot()

    async def _wait(self, seconds: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)

    async def run(self, hashtags: list[str], client: httpx.AsyncClient | None = None) -> None:
        logger.info("tail supervisor starting", extra={"extra_fields": {"hashtags": hashtags}})
        owns_client = client is None
        if client is None:
            limits = httpx.Limits(max_connections=len(hashtags) + 2)
            client = httpx.AsyncClient(
                timeout=self._rt.rss.request_timeout_seconds, follow_redirects=True, limits=limits
            )
        tasks = [asyncio.create_task(self._drain_queue()), asyncio.create_task(self._seal_loop())]
        tasks += [asyncio.create_task(self._collect_hashtag(client, h)) for h in hashtags]
        try:
            await self._stop.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._flush_sealed(self._engine.seal_due(now=datetime.now(timezone.utc)))
            if owns_client:
                await client.aclose()

    def stop(self) -> None:
        self._stop.set()


def build_engine(settings: Settings) -> IncrementalSignalEngine:
    a = settings.analysis
    return IncrementalSignalEngine(
        bucket_minutes=a.bucket_minutes,
        grace_seconds=settings.realtime.bucket_seal_grace_seconds,
        target_hashtags=set(settings.scraper.all_hashtags),
        sentiment_lexicon_bullish=a.sentiment_lexicon.bullish,
        sentiment_lexicon_bearish=a.sentiment_lexicon.bearish,
        signal_weights=a.signal_weights,
        confidence_level=a.bootstrap.confidence_level,
        n_resamples=a.bootstrap.n_resamples,
        min_bucket_tweets=a.min_bucket_tweets,
        seed=a.bootstrap.random_seed,
    )
