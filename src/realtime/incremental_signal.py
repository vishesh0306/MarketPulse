"""Incremental per-bucket signal for the tail loop.

The batch signal re-reads the whole processed dataset into one DataFrame every run, which
a live loop can't do. Here each (hashtag, 15-minute bucket) keeps a small running
accumulator updated in O(1) per tweet — Welford for the mean/variance of per-tweet
sentiment, plus counters for hashtag co-occurrence and lexicon coverage. When a bucket's
window closes (plus a grace period for stragglers) it is sealed: the composite signal,
interval, and coverage are finalized once and the bucket becomes immutable.

Signal shape matches the batch composite (see analysis/signal_generator): sentiment is the
signed term, hashtag momentum an unsigned multiplier. Engagement virality is absent from
this path because the RSS feed carries no counts — for a tweet seconds old it is ~0 anyway.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.analysis.feature_engineering import sentiment_lexicon_hit, sentiment_score
from src.analysis.signal_generator import bootstrap_confidence_interval

_RESERVOIR_SIZE = 256


def bucket_start_for(ts: datetime, bucket_minutes: int) -> datetime:
    """Floor a timestamp to its bucket boundary (UTC)."""
    ts = ts.astimezone(timezone.utc)
    floored_minute = (ts.minute // bucket_minutes) * bucket_minutes
    return ts.replace(minute=floored_minute, second=0, microsecond=0)


@dataclass
class BucketAccumulator:
    """Running state for one (hashtag, bucket). Not thread-safe; the updater owns it."""

    hashtag: str
    bucket_start: datetime
    bucket_minutes: int
    n: int = 0
    _mean: float = 0.0
    _m2: float = 0.0
    matched: int = 0
    _co_occurrence: int = 0
    _occurrence: int = 0
    _reservoir: list[float] = field(default_factory=list)
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add(self, sentiment: float, lexicon_hit: bool, other_target_tags: int) -> None:
        self.n += 1
        delta = sentiment - self._mean
        self._mean += delta / self.n
        self._m2 += delta * (sentiment - self._mean)
        if lexicon_hit:
            self.matched += 1
        self._occurrence += 1
        self._co_occurrence += other_target_tags
        if len(self._reservoir) < _RESERVOIR_SIZE:
            self._reservoir.append(sentiment)
        else:
            j = random.randint(0, self.n - 1)
            if j < _RESERVOIR_SIZE:
                self._reservoir[j] = sentiment
        self.last_update = datetime.now(timezone.utc)

    @property
    def ends_at(self) -> datetime:
        return self.bucket_start + timedelta(minutes=self.bucket_minutes)

    def is_sealable(self, now: datetime, grace_seconds: int) -> bool:
        return now >= self.ends_at + timedelta(seconds=grace_seconds)

    def _momentum_norm(self, target_hashtag_count: int) -> float:
        if self._occurrence == 0:
            return 0.0
        raw = self._co_occurrence / self._occurrence
        return min(1.0, raw / max(1, target_hashtag_count - 1))

    def seal(
        self,
        *,
        sentiment_weight: float,
        momentum_weight: float,
        confidence_level: float,
        n_resamples: int,
        target_hashtag_count: int,
        min_bucket_tweets: int,
        seed: int,
    ) -> dict[str, object]:
        """Finalize the bucket into an output row. Thin buckets are marked suppressed —
        volume only, no signal — exactly as the batch stage does."""
        coverage = self.matched / self.n if self.n else 0.0
        base = {
            "hashtag": self.hashtag,
            "bucket_start": self.bucket_start.isoformat(),
            "tweet_count": self.n,
            "sentiment_coverage": round(coverage, 4),
            "sealed_at": datetime.now(timezone.utc).isoformat(),
        }
        if self.n < min_bucket_tweets:
            return {**base, "composite_signal": None, "ci_lower": None, "ci_upper": None, "suppressed": True}

        multiplier = sentiment_weight * (1 + momentum_weight * self._momentum_norm(target_hashtag_count))
        lo, hi = bootstrap_confidence_interval(
            self._reservoir, n_resamples=n_resamples, confidence_level=confidence_level, seed=seed,
            fallback_std=float(self._std()),
        )
        return {
            **base,
            "composite_signal": round(self._mean * multiplier, 6),
            "ci_lower": round(lo * multiplier, 6),
            "ci_upper": round(hi * multiplier, 6),
            "suppressed": False,
        }

    def _std(self) -> float:
        return math.sqrt(self._m2 / (self.n - 1)) if self.n > 1 else 0.0


class IncrementalSignalEngine:
    """Holds the live accumulators and hands back sealed rows when buckets close."""

    def __init__(
        self,
        *,
        bucket_minutes: int,
        grace_seconds: int,
        target_hashtags: set[str],
        sentiment_lexicon_bullish: list[str],
        sentiment_lexicon_bearish: list[str],
        signal_weights: dict[str, float],
        confidence_level: float,
        n_resamples: int,
        min_bucket_tweets: int,
        seed: int,
    ) -> None:
        self._bucket_minutes = bucket_minutes
        self._grace = grace_seconds
        self._targets = target_hashtags
        self._bullish = sentiment_lexicon_bullish
        self._bearish = sentiment_lexicon_bearish
        self._weights = signal_weights
        self._confidence_level = confidence_level
        self._n_resamples = n_resamples
        self._min_bucket_tweets = min_bucket_tweets
        self._seed = seed
        self._live: dict[tuple[str, datetime], BucketAccumulator] = {}

    @property
    def live_bucket_count(self) -> int:
        return len(self._live)

    def ingest(self, record: dict[str, object]) -> None:
        hashtag = str(record["source_hashtag"])
        created_at = datetime.fromisoformat(str(record["created_at"]))
        start = bucket_start_for(created_at, self._bucket_minutes)
        key = (hashtag, start)
        acc = self._live.get(key)
        if acc is None:
            acc = BucketAccumulator(hashtag, start, self._bucket_minutes)
            self._live[key] = acc

        text = str(record.get("text_normalized") or record.get("text", ""))
        sentiment = sentiment_score(text, self._bullish, self._bearish)
        hit = sentiment_lexicon_hit(text, self._bullish, self._bearish)
        raw_tags = record.get("hashtags") or []
        tags = {str(t).lower() for t in raw_tags} & self._targets if isinstance(raw_tags, (list, tuple, set)) else set()
        other = max(0, len(tags | {hashtag}) - 1)
        acc.add(sentiment, hit, other)

    def seal_due(self, now: datetime | None = None) -> list[dict[str, object]]:
        """Seal and remove every bucket whose window (plus grace) has passed."""
        now = now or datetime.now(timezone.utc)
        due = [key for key, acc in self._live.items() if acc.is_sealable(now, self._grace)]
        sealed: list[dict[str, object]] = []
        for key in due:
            acc = self._live.pop(key)
            sealed.append(
                acc.seal(
                    sentiment_weight=self._weights["sentiment"],
                    momentum_weight=self._weights["hashtag_momentum"],
                    confidence_level=self._confidence_level,
                    n_resamples=self._n_resamples,
                    target_hashtag_count=len(self._targets),
                    min_bucket_tweets=self._min_bucket_tweets,
                    seed=self._seed,
                )
            )
        return sealed

    def snapshot(self) -> list[dict[str, object]]:
        """Provisional (un-sealed) rows for every live bucket — what `GET /signals?live=1`
        serves. Same shape as a sealed row but `sealed_at` is null."""
        rows: list[dict[str, object]] = []
        for acc in sorted(self._live.values(), key=lambda a: (a.hashtag, a.bucket_start)):
            row = acc.seal(
                sentiment_weight=self._weights["sentiment"],
                momentum_weight=self._weights["hashtag_momentum"],
                confidence_level=self._confidence_level,
                n_resamples=self._n_resamples,
                target_hashtag_count=len(self._targets),
                min_bucket_tweets=self._min_bucket_tweets,
                seed=self._seed,
            )
            row["sealed_at"] = None
            rows.append(row)
        return rows
