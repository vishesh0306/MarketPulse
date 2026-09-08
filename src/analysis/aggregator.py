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

    keys = ["hashtag", "bucket_start"]

    # Volume rolls up over every fine bucket, including the ones whose signal was
    # suppressed for being too thin — the coarse bucket's tweet_count is still the full
    # count.
    volume = df.groupby(keys, observed=True)["tweet_count"].sum().rename("tweet_count")

    # Signal and interval roll up as a tweet-count-weighted average over only the fine
    # buckets that carry a signal, so a run of suppressed buckets doesn't drag the coarse
    # value toward zero or leave it undefined.
    scored = df.dropna(subset=["composite_signal"]).copy()
    weighted = scored.assign(
        _w_signal=scored["composite_signal"] * scored["tweet_count"],
        _w_ci_lower=scored["ci_lower"] * scored["tweet_count"],
        _w_ci_upper=scored["ci_upper"] * scored["tweet_count"],
    )
    agg = weighted.groupby(keys, observed=True).agg(
        _w_signal=("_w_signal", "sum"),
        _w_ci_lower=("_w_ci_lower", "sum"),
        _w_ci_upper=("_w_ci_upper", "sum"),
        _scored_tweets=("tweet_count", "sum"),
    )
    agg["composite_signal"] = agg["_w_signal"] / agg["_scored_tweets"]
    agg["ci_lower"] = agg["_w_ci_lower"] / agg["_scored_tweets"]
    agg["ci_upper"] = agg["_w_ci_upper"] / agg["_scored_tweets"]

    out = (
        volume.to_frame()
        .join(agg[["composite_signal", "ci_lower", "ci_upper"]], how="left")
        .reset_index()
    )
    return out[["hashtag", "bucket_start", "composite_signal", "ci_lower", "ci_upper", "tweet_count"]].sort_values(
        keys
    ).reset_index(drop=True)
