"""Buckets tweets into time windows and computes a weighted composite signal with a
bootstrapped confidence interval per bucket. The bootstrap step is the only per-bucket
loop and is parallelized with joblib — everything else is vectorized pandas/numpy.
"""

from __future__ import annotations

import argparse
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from src.analysis.feature_engineering import build_feature_frame, fit_tfidf, hashtag_momentum, tfidf_summary
from src.utils.config_loader import Settings, load_settings
from src.utils.logger import get_logger, write_run_summary

logger = get_logger("signal_generator")

OUTPUT_COLUMNS = ["bucket_start", "hashtag", "composite_signal", "ci_lower", "ci_upper", "tweet_count"]


def bucket_tweets(df: pd.DataFrame, bucket_minutes: int) -> pd.DataFrame:
    """Assigns each tweet to a fixed-width time bucket (bucket_start column)."""
    out = df.copy()
    out["bucket_start"] = out["created_at"].dt.floor(f"{bucket_minutes}min")
    return out


def _per_tweet_contribution(features: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return (
        weights["sentiment"] * features["sentiment"]
        + weights["virality"] * features["virality_norm"]
        + weights["hashtag_momentum"] * features["momentum_norm"]
    )


def composite_signal(features: pd.DataFrame, weights: dict[str, float]) -> float:
    """Weighted sum of engineered features (weights sourced from config, never hardcoded)."""
    if features.empty:
        return 0.0
    return float(_per_tweet_contribution(features, weights).mean())


def bootstrap_confidence_interval(
    values: list[float],
    n_resamples: int,
    confidence_level: float,
    seed: int | None = None,
    fallback_std: float = 0.0,
) -> tuple[float, float]:
    """Bootstrap CI over the values in a bucket; wider when the bucket has fewer tweets.

    Pure resampling degenerates to zero width for a single-tweet bucket — there's nothing
    to resample, so it would otherwise report the *most* confidence exactly where there
    should be the *least*. A standard-error-based margin (using `fallback_std`, typically
    the dataset-wide std, as a prior when the bucket has too few points to estimate its
    own variance) is combined with the bootstrap margin via max(), so the interval is
    never narrower than sample size alone would justify.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return 0.0, 0.0

    point_estimate = float(arr.mean())
    alpha = 1 - confidence_level
    z = float(stats.norm.ppf(1 - alpha / 2))

    bootstrap_margin = 0.0
    if n >= 2:
        rng = np.random.default_rng(seed)
        resample_means = np.array(
            [rng.choice(arr, size=n, replace=True).mean() for _ in range(n_resamples)]
        )
        lower_pct = float(np.percentile(resample_means, 100 * alpha / 2))
        upper_pct = float(np.percentile(resample_means, 100 * (1 - alpha / 2)))
        bootstrap_margin = max(point_estimate - lower_pct, upper_pct - point_estimate)

    std = float(arr.std(ddof=1)) if n >= 2 else fallback_std
    std = max(std, fallback_std)
    analytical_margin = z * std / np.sqrt(n)

    margin = max(bootstrap_margin, analytical_margin)
    return point_estimate - margin, point_estimate + margin


def _bucket_result(
    key: tuple[str, Any],
    group: pd.DataFrame,
    weights: dict[str, float],
    n_resamples: int,
    confidence_level: float,
    fallback_std: float,
) -> dict[str, Any]:
    hashtag, bucket_start = key
    contributions = _per_tweet_contribution(group, weights).tolist()
    signal = float(np.mean(contributions)) if contributions else 0.0
    ci_lower, ci_upper = bootstrap_confidence_interval(
        contributions, n_resamples, confidence_level, fallback_std=fallback_std
    )
    return {
        "bucket_start": bucket_start,
        "hashtag": hashtag,
        "composite_signal": signal,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "tweet_count": len(group),
    }


def generate_signals(processed_dir: Path, config: Settings) -> pd.DataFrame:
    """Runs bucketing -> composite signal -> CI over the processed dataset.

    Output columns: bucket_start, composite_signal, ci_lower, ci_upper, tweet_count, hashtag.
    """
    df = pd.read_parquet(processed_dir)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = build_feature_frame(df, config.analysis)
    df = bucket_tweets(df, config.analysis.bucket_minutes)

    # TF-IDF is fit once, batch-style, over the whole available corpus — the "text-to-
    # vector" deliverable. It is not fed into the composite signal (see module docstring
    # in feature_engineering.py), so there is no lookahead/leakage concern for the signal
    # itself; a summary is logged as evidence it ran.
    vectorizer, matrix = fit_tfidf(
        df["text_normalized"].tolist(),
        config.analysis.tfidf.max_features,
        config.analysis.tfidf.ngram_range,
        config.analysis.tfidf.min_df,
        config.analysis.tfidf.stopwords_extra,
    )
    logger.info("tfidf fit complete", extra={"extra_fields": tfidf_summary(vectorizer, matrix)})

    # Hashtag momentum is bucket-level (co-occurrence across tweets in a time window), so
    # it's computed per (hashtag, bucket) group, dataset-wide min-max normalized, then
    # merged back as a per-row column — a bootstrap resample of a bucket's rows leaves
    # this column constant, which is intentional: momentum isn't attributable to any one
    # tweet within the bucket.
    momentum_rows = []
    for (hashtag, bucket_start), group in df.groupby(["source_hashtag", "bucket_start"], observed=True):
        momentum_map = hashtag_momentum(group["hashtags"].tolist())
        momentum_rows.append(
            {"source_hashtag": hashtag, "bucket_start": bucket_start, "momentum_raw": momentum_map.get(hashtag, 0.0)}
        )
    momentum_df = pd.DataFrame(momentum_rows)
    m_min, m_max = momentum_df["momentum_raw"].min(), momentum_df["momentum_raw"].max()
    m_range = (m_max - m_min) or 1.0
    momentum_df["momentum_norm"] = (momentum_df["momentum_raw"] - m_min) / m_range
    df = df.merge(momentum_df[["source_hashtag", "bucket_start", "momentum_norm"]], on=["source_hashtag", "bucket_start"], how="left")

    weights = config.analysis.signal_weights
    n_resamples = config.analysis.bootstrap.n_resamples
    confidence_level = config.analysis.bootstrap.confidence_level

    # Dataset-wide std of per-tweet contributions, used as the small-bucket CI fallback.
    fallback_std = float(_per_tweet_contribution(df, weights).std(ddof=1)) if len(df) >= 2 else 0.0

    groups = list(df.groupby(["source_hashtag", "bucket_start"], observed=True))
    results = Parallel(n_jobs=-1)(
        delayed(_bucket_result)(key, group, weights, n_resamples, confidence_level, fallback_std)
        for key, group in groups
    )

    return pd.DataFrame(results, columns=OUTPUT_COLUMNS).sort_values(["hashtag", "bucket_start"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate composite trading signals from processed tweets.")
    parser.add_argument("--input", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="data/output")
    args = parser.parse_args()

    settings = load_settings()
    output_dir = Path(args.output) / "signals"
    output_dir.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    signals = generate_signals(Path(args.input), settings)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    output_path = output_dir / "signals.parquet"
    signals.to_parquet(output_path, compression=settings.storage.parquet_compression)

    summary = {
        "buckets": len(signals),
        "hashtags": sorted(signals["hashtag"].unique().tolist()) if not signals.empty else [],
        "output_path": str(output_path),
        "peak_memory_mb": round(peak_bytes / (1024 * 1024), 2),
    }
    summary_path = write_run_summary(settings.logging.dir, "signals", summary)
    logger.info("signal generation complete", extra={"extra_fields": {**summary, "summary_path": str(summary_path)}})


if __name__ == "__main__":
    main()
