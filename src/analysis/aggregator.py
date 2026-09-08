"""Rolls bucketed signals up to coarser windows (hourly/daily) for the visualization layer
and any downstream consumer.
"""

from __future__ import annotations

import pandas as pd


def rollup(bucketed_signals: pd.DataFrame, window: str) -> pd.DataFrame:
    """Aggregates bucket-level signals to a coarser window (e.g. '1h', '1d').

    Coarser-window values are tweet-count-weighted averages of the finer buckets, so a
    15-minute bucket with 200 tweets influences the daily rollup more than one with 2.
    """
    if bucketed_signals.empty:
        return bucketed_signals.copy()

    df = bucketed_signals.copy()
    # pandas 2.2 deprecated the uppercase offset aliases ('H', 'D') in favour of lowercase;
    # normalize here so an old-style config value keeps working without the warning.
    df["bucket_start"] = df["bucket_start"].dt.floor(window.lower())

    weighted = df.assign(
        weighted_signal=df["composite_signal"] * df["tweet_count"],
        weighted_ci_lower=df["ci_lower"] * df["tweet_count"],
        weighted_ci_upper=df["ci_upper"] * df["tweet_count"],
    )
    grouped = weighted.groupby(["hashtag", "bucket_start"], observed=True).agg(
        weighted_signal=("weighted_signal", "sum"),
        weighted_ci_lower=("weighted_ci_lower", "sum"),
        weighted_ci_upper=("weighted_ci_upper", "sum"),
        tweet_count=("tweet_count", "sum"),
    )

    grouped["composite_signal"] = grouped["weighted_signal"] / grouped["tweet_count"]
    grouped["ci_lower"] = grouped["weighted_ci_lower"] / grouped["tweet_count"]
    grouped["ci_upper"] = grouped["weighted_ci_upper"] / grouped["tweet_count"]

    out = grouped[["composite_signal", "ci_lower", "ci_upper", "tweet_count"]].reset_index()
    return out.sort_values(["hashtag", "bucket_start"]).reset_index(drop=True)
