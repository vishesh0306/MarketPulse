"""Memory-efficient plots: pre-aggregated buckets for time series, reservoir sampling for
raw-tweet-level views. Plotting memory is bounded by sample size, not dataset size.
"""

from __future__ import annotations

import argparse
import random
import tracemalloc
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

import matplotlib

matplotlib.use("Agg")  # headless: no display backend needed to write PNGs

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.dataset as ds

from src.utils.config_loader import load_settings
from src.utils.logger import get_logger, set_level, write_run_summary

logger = get_logger("streaming_plots")

T = TypeVar("T")

# Fixed categorical color order (first four slots of the validated default palette),
# assigned to hashtags in a stable order so the same hashtag is always the same color
# across all three plots — never re-cycled per axes.
_CATEGORICAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"


def _hashtag_colors(hashtags: list[str]) -> dict[str, str]:
    return {tag: _CATEGORICAL_COLORS[i % len(_CATEGORICAL_COLORS)] for i, tag in enumerate(sorted(hashtags))}


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_BASELINE)
    ax.spines["bottom"].set_color(_BASELINE)
    ax.tick_params(colors=_INK_MUTED, labelsize=8)
    ax.title.set_color(_INK_PRIMARY)
    ax.xaxis.label.set_color(_INK_SECONDARY)
    ax.yaxis.label.set_color(_INK_SECONDARY)


def reservoir_sample(items: Iterable[T], sample_size: int) -> list[T]:
    """Single-pass reservoir sampling — O(sample_size) memory regardless of stream length."""
    reservoir: list[T] = []
    for index, item in enumerate(items):
        if index < sample_size:
            reservoir.append(item)
        else:
            j = random.randint(0, index)
            if j < sample_size:
                reservoir[j] = item
    return reservoir


def _iter_processed_rows(processed_dir: Path) -> Iterator[dict[str, Any]]:
    """Streams processed-tweet rows batch by batch via pyarrow, never materializing the
    full dataset as a pandas DataFrame — the source reservoir_sample draws from."""
    dataset = ds.dataset(processed_dir, format="parquet", partitioning="hive")
    for batch in dataset.to_batches():
        yield from batch.to_pylist()


def plot_volume_over_time(bucketed: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Tweet-volume-over-time plot from aggregated bucket data (not raw tweets)."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_SURFACE)
    _style_axes(ax)

    colors = _hashtag_colors(bucketed["hashtag"].unique().tolist())
    for hashtag in sorted(colors):
        series = bucketed[bucketed["hashtag"] == hashtag].sort_values("bucket_start")
        ax.plot(
            series["bucket_start"], series["tweet_count"],
            label=f"#{hashtag}", color=colors[hashtag], linewidth=2,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Tweets per 15-min bucket")
    ax.set_title("Tweet Volume Over Time", color=_INK_PRIMARY, fontsize=13, fontweight="bold")
    ax.legend(frameon=False, labelcolor=_INK_SECONDARY)
    fig.autofmt_xdate()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_signal_with_ci(bucketed: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Composite-signal-over-time plot with the confidence interval shaded.

    One subplot per hashtag (small multiples) rather than overlapping filled CI bands on
    a single axes — four semi-transparent bands on top of each other is unreadable.
    """
    hashtags = sorted(bucketed["hashtag"].unique().tolist())
    colors = _hashtag_colors(hashtags)

    fig, axes_arr = plt.subplots(
        len(hashtags), 1, figsize=(10, 3 * len(hashtags)), sharex=True, facecolor=_SURFACE, squeeze=False
    )
    axes: list[plt.Axes] = list(axes_arr[:, 0])

    for ax, hashtag in zip(axes, hashtags):
        _style_axes(ax)
        series = bucketed[bucketed["hashtag"] == hashtag].sort_values("bucket_start")
        color = colors[hashtag]
        ax.plot(series["bucket_start"], series["composite_signal"], color=color, linewidth=2)
        ax.fill_between(series["bucket_start"], series["ci_lower"], series["ci_upper"], color=color, alpha=0.2)
        ax.axhline(0, color=_BASELINE, linewidth=0.8)
        ax.set_ylabel(f"#{hashtag}", color=_INK_SECONDARY, fontsize=10)

    axes[0].set_title("Composite Signal Over Time (shaded = confidence interval)", color=_INK_PRIMARY, fontsize=13, fontweight="bold")
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_hashtag_engagement_distribution(sample: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Hashtag/engagement distribution plot built from a bounded reservoir sample."""
    colors = _hashtag_colors(sample["source_hashtag"].unique().tolist())
    hashtags = sorted(colors)

    engagement = sample.assign(total_engagement=sample["likes"] + sample["retweets"] + sample["replies"])
    counts = engagement.groupby("source_hashtag")["tweet_id"].count().reindex(hashtags).fillna(0)
    means = engagement.groupby("source_hashtag")["total_engagement"].mean().reindex(hashtags).fillna(0)

    fig, axes_arr = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=_SURFACE, squeeze=False)
    ax1, ax2 = axes_arr[0, 0], axes_arr[0, 1]
    for ax in (ax1, ax2):
        _style_axes(ax)

    bar_colors = [colors[h] for h in hashtags]
    ax1.bar([f"#{h}" for h in hashtags], counts.values, color=bar_colors)
    ax1.set_ylabel("Tweets in sample")
    ax1.set_title("Sample Composition", color=_INK_PRIMARY, fontsize=11)

    ax2.bar([f"#{h}" for h in hashtags], means.values, color=bar_colors)
    ax2.set_ylabel("Mean engagement (likes + retweets + replies)")
    ax2.set_title("Average Engagement", color=_INK_PRIMARY, fontsize=11)

    fig.suptitle(
        f"Hashtag / Engagement Distribution (reservoir sample, n={len(sample)})",
        color=_INK_PRIMARY, fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


def iter_plot_all(
    signals_path: Path, processed_dir: Path, output_dir: Path, sample_size: int, dpi: int
) -> Iterator[Path]:
    """Generates all plots, yielding each output path as it's written.

    One plot's data lives in memory at a time — the bucketed signal frame (already small,
    pre-aggregated) or the bounded reservoir sample — never the full raw dataset.
    """
    bucketed = pd.read_parquet(signals_path)
    if not bucketed.empty:
        volume_path = output_dir / "volume_over_time.png"
        plot_volume_over_time(bucketed, volume_path, dpi)
        yield volume_path

        signal_path = output_dir / "signal_with_ci.png"
        plot_signal_with_ci(bucketed, signal_path, dpi)
        yield signal_path
    del bucketed

    sampled_rows = reservoir_sample(_iter_processed_rows(processed_dir), sample_size)
    if sampled_rows:
        sample_df = pd.DataFrame(sampled_rows)
        distribution_path = output_dir / "hashtag_engagement_distribution.png"
        plot_hashtag_engagement_distribution(sample_df, distribution_path, dpi)
        yield distribution_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate memory-efficient plots from signal output.")
    parser.add_argument("--input", type=str, default="data/output", help="Directory containing signals/signals.parquet")
    parser.add_argument("--processed", type=str, default="data/processed", help="Processed tweets dir for the reservoir sample")
    args = parser.parse_args()

    settings = load_settings()
    set_level(logger, settings.logging.level)
    signals_path = Path(args.input) / "signals" / "signals.parquet"
    plots_dir = Path(settings.storage.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    written = list(
        iter_plot_all(signals_path, Path(args.processed), plots_dir, settings.visualization.reservoir_sample_size, settings.visualization.dpi)
    )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    summary = {
        "plots_written": [str(p) for p in written],
        "peak_memory_mb": round(peak_bytes / (1024 * 1024), 2),
        "reservoir_sample_size": settings.visualization.reservoir_sample_size,
    }
    summary_path = write_run_summary(settings.logging.dir, "visualization", summary)
    logger.info("plotting complete", extra={"extra_fields": {**summary, "summary_path": str(summary_path)}})


if __name__ == "__main__":
    main()
