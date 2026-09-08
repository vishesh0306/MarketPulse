"""Rolls bucketed signals up to coarser windows (hourly/daily) for the visualization layer
and any downstream consumer.
"""

from __future__ import annotations

import pandas as pd

_KEYS = ["hashtag", "bucket_start"]


def _weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> pd.Series:
    """Per-group weighted mean of value_col by weight_col, over rows where value_col is
    not null. Groups with no non-null rows come back as NaN."""
    rows = df.dropna(subset=[value_col])
    num = (rows[value_col] * rows[weight_col]).groupby([rows["hashtag"], rows["bucket_start"]]).sum()
    den = rows[weight_col].groupby([rows["hashtag"], rows["bucket_start"]]).sum()
    return (num / den).rename(value_col)


def rollup(bucketed_signals: pd.DataFrame, window: str) -> pd.DataFrame:
    """Aggregates bucket-level signals to a coarser window (e.g. '1h', '1d').

    Coarser-window values are tweet-count-weighted averages of the finer buckets, so a
    15-minute bucket with 200 tweets influences the daily rollup more than one with 2.
    Volume (tweet_count) sums over every fine bucket; the signal, interval and sentiment
    coverage are weighted means over the fine buckets that carry a value, so a run of
    suppressed thin buckets neither drags the coarse signal toward zero nor blanks it.
    """
    if bucketed_signals.empty:
        return bucketed_signals.copy()

    df = bucketed_signals.copy()
    # pandas 2.2 deprecated the uppercase offset aliases ('H', 'D') in favour of lowercase;
    # normalize here so an old-style config value keeps working without the warning.
    df["bucket_start"] = df["bucket_start"].dt.floor(window.lower())

    parts: list[pd.Series] = [df.groupby(_KEYS, observed=True)["tweet_count"].sum()]
    for col in ("composite_signal", "ci_lower", "ci_upper", "sentiment_coverage"):
        if col in df.columns:
            parts.append(_weighted_mean(df, col, "tweet_count"))

    out = pd.concat(parts, axis=1).reset_index()
    ordered = ["hashtag", "bucket_start", "composite_signal", "ci_lower", "ci_upper", "tweet_count"]
    if "sentiment_coverage" in out.columns:
        ordered.append("sentiment_coverage")
    return out[ordered].sort_values(_KEYS).reset_index(drop=True)
