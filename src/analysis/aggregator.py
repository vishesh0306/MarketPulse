"""Rolls bucketed signals up to coarser windows (hourly/daily) for visualization/downstream use."""

from __future__ import annotations

import pandas as pd


def rollup(bucketed_signals: pd.DataFrame, window: str) -> pd.DataFrame:
    """Aggregates bucket-level signals to a coarser window (e.g. '1H', '1D')."""
    raise NotImplementedError
