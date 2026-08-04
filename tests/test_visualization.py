"""Tests for src/visualization: reservoir sampling and plot generation."""

from __future__ import annotations

import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.visualization.streaming_plots import (
    iter_plot_all,
    plot_hashtag_engagement_distribution,
    plot_signal_with_ci,
    plot_volume_over_time,
    reservoir_sample,
)

# ---- reservoir_sample ---------------------------------------------------


def test_reservoir_sample_returns_full_stream_when_smaller_than_sample_size() -> None:
    result = reservoir_sample(range(10), sample_size=100)
    assert sorted(result) == list(range(10))


def test_reservoir_sample_bounds_output_size() -> None:
    result = reservoir_sample(range(100_000), sample_size=500)
    assert len(result) == 500


def test_reservoir_sample_covers_late_stream_items() -> None:
    """A naive head-N sample would never see items past index `sample_size` — reservoir
    sampling must give every item, including late ones, a chance of being retained."""
    result = reservoir_sample(range(10_000), sample_size=200)
    assert max(result) > 500  # extremely unlikely under correct uniform sampling


# ---- plots (structural: files are created, not visual correctness) ------


@pytest.fixture
def bucketed_signals() -> pd.DataFrame:
    base = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
    rows = []
    for hashtag in ["nifty50", "sensex"]:
        for i in range(5):
            rows.append(
                {
                    "bucket_start": base + timedelta(minutes=15 * i),
                    "hashtag": hashtag,
                    "composite_signal": 0.1 * i,
                    "ci_lower": 0.1 * i - 0.05,
                    "ci_upper": 0.1 * i + 0.05,
                    "tweet_count": 10 + i,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tweet_id": [str(i) for i in range(20)],
            "source_hashtag": (["nifty50"] * 10 + ["sensex"] * 10),
            "likes": list(range(20)),
            "retweets": [0] * 20,
            "replies": [0] * 20,
        }
    )


def test_plot_volume_over_time_writes_file(tmp_path: Path, bucketed_signals: pd.DataFrame) -> None:
    output_path = tmp_path / "volume.png"
    plot_volume_over_time(bucketed_signals, output_path, dpi=72)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_signal_with_ci_writes_file(tmp_path: Path, bucketed_signals: pd.DataFrame) -> None:
    output_path = tmp_path / "signal.png"
    plot_signal_with_ci(bucketed_signals, output_path, dpi=72)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_hashtag_engagement_distribution_writes_file(tmp_path: Path, raw_sample: pd.DataFrame) -> None:
    output_path = tmp_path / "engagement.png"
    plot_hashtag_engagement_distribution(raw_sample, output_path, dpi=72)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# ---- memory bound --------------------------------------------------------


def test_iter_plot_all_peak_memory_bounded_by_sample_size(tmp_path: Path, bucketed_signals: pd.DataFrame) -> None:
    """Peak memory during plotting should scale with the (bounded) sample size, not with
    the size of the underlying processed dataset."""
    signals_path = tmp_path / "signals.parquet"
    bucketed_signals.to_parquet(signals_path)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    n_rows = 20_000
    big_df = pd.DataFrame(
        {
            "tweet_id": [str(i) for i in range(n_rows)],
            "source_hashtag": ["nifty50"] * n_rows,
            "likes": [1] * n_rows,
            "retweets": [0] * n_rows,
            "replies": [0] * n_rows,
        }
    )
    big_df.to_parquet(processed_dir / "part-0.parquet")

    output_dir = tmp_path / "plots"
    sample_size = 500

    tracemalloc.start()
    written = list(iter_plot_all(signals_path, processed_dir, output_dir, sample_size, dpi=72))
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(written) == 3
    # Loose bound: peak memory should be a small multiple of the sample size, nowhere
    # near proportional to n_rows (20,000 rows would blow well past this if unbounded).
    assert peak_bytes < 60 * 1024 * 1024  # 60MB
