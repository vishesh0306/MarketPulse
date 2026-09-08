"""Incremental per-bucket signal: Welford accuracy, sealing, suppression, momentum."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.realtime.incremental_signal import BucketAccumulator, IncrementalSignalEngine, bucket_start_for

_BULLISH = ["breakout", "rally", "tezi"]
_BEARISH = ["breakdown", "crash", "mandi"]
_WEIGHTS = {"sentiment": 0.4, "virality": 0.35, "hashtag_momentum": 0.25}


def _engine(**overrides: object) -> IncrementalSignalEngine:
    kwargs: dict[str, object] = dict(
        bucket_minutes=15,
        grace_seconds=120,
        target_hashtags={"nifty50", "banknifty", "sensex"},
        sentiment_lexicon_bullish=_BULLISH,
        sentiment_lexicon_bearish=_BEARISH,
        signal_weights=_WEIGHTS,
        confidence_level=0.90,
        n_resamples=200,
        min_bucket_tweets=5,
        seed=42,
    )
    kwargs.update(overrides)
    return IncrementalSignalEngine(**kwargs)  # type: ignore[arg-type]


def _record(text: str, hashtag: str = "nifty50", minute: int = 20, hashtags: list[str] | None = None) -> dict[str, object]:
    return {
        "tweet_id": text[:8],
        "source_hashtag": hashtag,
        "created_at": datetime(2026, 9, 8, 6, minute, 0, tzinfo=timezone.utc).isoformat(),
        "text_normalized": text,
        "hashtags": hashtags if hashtags is not None else [hashtag],
    }


def test_bucket_start_for_floors_to_window() -> None:
    ts = datetime(2026, 9, 8, 6, 23, 41, tzinfo=timezone.utc)
    assert bucket_start_for(ts, 15) == datetime(2026, 9, 8, 6, 15, 0, tzinfo=timezone.utc)


def test_welford_mean_matches_naive_mean() -> None:
    acc = BucketAccumulator("nifty50", datetime(2026, 9, 8, 6, 15, tzinfo=timezone.utc), 15)
    values = [1.0, -1.0, 0.5, 0.0, -0.25, 1.0, -0.5]
    for v in values:
        acc.add(v, lexicon_hit=True, other_target_tags=0)
    assert acc._mean == pytest.approx(sum(values) / len(values))
    assert acc.n == len(values)


def test_thin_bucket_is_suppressed_on_seal() -> None:
    engine = _engine()
    for i in range(3):  # below min_bucket_tweets=5
        engine.ingest(_record(f"nifty breakout number {i}"))
    sealed = engine.seal_due(now=datetime(2026, 9, 8, 7, 0, tzinfo=timezone.utc))
    assert len(sealed) == 1
    assert sealed[0]["suppressed"] is True
    assert sealed[0]["composite_signal"] is None
    assert sealed[0]["tweet_count"] == 3


def test_bullish_bucket_seals_positive_signal_with_coverage() -> None:
    engine = _engine()
    for i in range(8):
        engine.ingest(_record(f"nifty breakout rally continues {i}"))
    sealed = engine.seal_due(now=datetime(2026, 9, 8, 7, 0, tzinfo=timezone.utc))
    row = sealed[0]
    assert row["suppressed"] is False
    assert isinstance(row["composite_signal"], float) and row["composite_signal"] > 0
    assert row["ci_lower"] <= row["composite_signal"] <= row["ci_upper"]
    assert row["sentiment_coverage"] == 1.0  # every tweet matched a lexicon term


def test_coverage_distinguishes_unreadable_from_balanced() -> None:
    engine = _engine()
    for i in range(8):
        engine.ingest(_record(f"just some market commentary today {i}"))  # no lexicon terms
    row = engine.seal_due(now=datetime(2026, 9, 8, 7, 0, tzinfo=timezone.utc))[0]
    assert row["composite_signal"] == 0.0
    assert row["sentiment_coverage"] == 0.0


def test_momentum_multiplier_lifts_magnitude_for_cross_hashtag_chatter() -> None:
    lonely = _engine()
    crossed = _engine()
    for i in range(8):
        lonely.ingest(_record(f"nifty breakout {i}", hashtags=["nifty50"]))
        crossed.ingest(_record(f"nifty breakout {i}", hashtags=["nifty50", "banknifty", "sensex"]))
    lonely_row = lonely.seal_due(now=datetime(2026, 9, 8, 7, 0, tzinfo=timezone.utc))[0]
    crossed_row = crossed.seal_due(now=datetime(2026, 9, 8, 7, 0, tzinfo=timezone.utc))[0]
    assert abs(crossed_row["composite_signal"]) > abs(lonely_row["composite_signal"])


def test_seal_due_leaves_buckets_still_inside_grace_window() -> None:
    engine = _engine()
    for i in range(8):
        engine.ingest(_record(f"nifty breakout {i}", minute=20))
    # bucket 06:15-06:30, grace 120s -> not sealable at 06:31
    assert engine.seal_due(now=datetime(2026, 9, 8, 6, 31, tzinfo=timezone.utc)) == []
    assert engine.live_bucket_count == 1
    # sealable at 06:32:01
    assert len(engine.seal_due(now=datetime(2026, 9, 8, 6, 32, 1, tzinfo=timezone.utc))) == 1
    assert engine.live_bucket_count == 0


def test_snapshot_reports_live_buckets_without_removing_them() -> None:
    engine = _engine()
    for i in range(6):
        engine.ingest(_record(f"nifty rally {i}"))
    snap = engine.snapshot()
    assert len(snap) == 1
    assert snap[0]["sealed_at"] is None
    assert engine.live_bucket_count == 1  # not consumed


def test_reservoir_is_bounded() -> None:
    acc = BucketAccumulator("nifty50", datetime(2026, 9, 8, 6, 15, tzinfo=timezone.utc), 15)
    for _ in range(1000):
        acc.add(0.5, lexicon_hit=False, other_target_tags=0)
    assert len(acc._reservoir) <= 256
    assert acc.n == 1000
