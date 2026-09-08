"""Runtime metrics for the real-time service.

A real-time system that can't state its own lag isn't making a checkable claim, so the
same instinct the batch stage applies to memory (measure it, write it to the run summary)
is applied here to latency and throughput. Everything is cheap, in-process, and exposed
at GET /metrics.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import median


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


@dataclass
class Metrics:
    """Rolling counters and samples. Windows are wall-clock seconds."""

    tweets_per_minute_window: float = 60.0
    lag_sample_size: int = 512

    _ingest_lag_s: deque[float] = field(default_factory=lambda: deque(maxlen=512))
    _ingest_times: deque[float] = field(default_factory=lambda: deque(maxlen=10_000))
    _ingest_by_hashtag: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=5_000)))
    _dedup_seen: int = 0
    _dedup_new: int = 0
    _polls_ok: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _polls_fail: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _last_seal_at: dict[str, float] = field(default_factory=dict)
    _sealed_total: int = 0
    _queue_depth: int = 0
    _queue_drops: int = 0
    started_at: float = field(default_factory=time.time)

    def record_ingest(self, hashtag: str, tweet_created_ts: float) -> None:
        now = time.time()
        self._ingest_lag_s.append(max(0.0, now - tweet_created_ts))
        self._ingest_times.append(now)
        self._ingest_by_hashtag[hashtag].append(now)

    def record_dedup(self, *, was_duplicate: bool) -> None:
        if was_duplicate:
            self._dedup_seen += 1
        else:
            self._dedup_new += 1

    def record_poll(self, host: str | None, *, ok: bool) -> None:
        key = host or "none"
        (self._polls_ok if ok else self._polls_fail)[key] += 1

    def record_seal(self, hashtag: str, count: int = 1) -> None:
        self._last_seal_at[hashtag] = time.time()
        self._sealed_total += count

    def set_queue_depth(self, depth: int) -> None:
        self._queue_depth = depth

    def record_queue_drop(self) -> None:
        self._queue_drops += 1

    def _rate_per_minute(self, times: deque[float]) -> float:
        cutoff = time.time() - self.tweets_per_minute_window
        recent = sum(1 for t in times if t >= cutoff)
        return round(recent / (self.tweets_per_minute_window / 60.0), 2)

    def snapshot(self) -> dict[str, object]:
        lag = list(self._ingest_lag_s)
        total_dedup = self._dedup_seen + self._dedup_new
        return {
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "ingest_lag_seconds": {
                "p50": round(median(lag), 2) if lag else 0.0,
                "p95": round(_percentile(lag, 95), 2),
                "samples": len(lag),
            },
            "tweets_per_minute": {
                "total": self._rate_per_minute(self._ingest_times),
                "by_hashtag": {h: self._rate_per_minute(ts) for h, ts in self._ingest_by_hashtag.items()},
            },
            "dedup_hit_rate": round(self._dedup_seen / total_dedup, 3) if total_dedup else 0.0,
            "poll_success_rate_by_host": {
                host: round(self._polls_ok.get(host, 0) / total, 3)
                for host in {*self._polls_ok, *self._polls_fail}
                if (total := self._polls_ok.get(host, 0) + self._polls_fail.get(host, 0)) > 0
            },
            "signal_staleness_seconds": {
                h: round(time.time() - t, 1) for h, t in self._last_seal_at.items()
            },
            "sealed_buckets_total": self._sealed_total,
            "queue_depth": self._queue_depth,
            "queue_drops_total": self._queue_drops,
        }
