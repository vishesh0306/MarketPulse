"""Memory-efficient plots: pre-aggregated buckets for time series, reservoir sampling for
raw-tweet-level distribution views. Plotting memory is bounded by sample size, not dataset size.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import pandas as pd

T = TypeVar("T")


def reservoir_sample(items: Iterable[T], sample_size: int) -> list[T]:
    """Single-pass reservoir sampling — O(sample_size) memory regardless of stream length."""
    raise NotImplementedError


def plot_volume_over_time(bucketed: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Tweet-volume-over-time plot from aggregated bucket data (not raw tweets)."""
    raise NotImplementedError


def plot_signal_with_ci(bucketed: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Composite-signal-over-time plot with the confidence interval shaded."""
    raise NotImplementedError


def plot_hashtag_engagement_distribution(sample: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Hashtag/engagement distribution plot built from a bounded reservoir sample."""
    raise NotImplementedError


def iter_plot_all(input_dir: Path, output_dir: Path, sample_size: int, dpi: int) -> Iterator[Path]:
    """Generates all plots, yielding each output path as it's written."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate memory-efficient plots from signal output.")
    parser.add_argument("--input", type=str, default="data/output")
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
