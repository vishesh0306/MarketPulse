"""Tests for src/analysis/market_hours.py."""

from __future__ import annotations

from datetime import datetime, timezone

from src.analysis.market_hours import is_market_hours


def test_is_market_hours_true_during_session() -> None:
    # Tue 2026-08-04 06:00 UTC = 11:30 IST, inside 09:15-15:30.
    assert is_market_hours(datetime(2026, 8, 4, 6, 0, tzinfo=timezone.utc)) is True


def test_is_market_hours_false_before_open() -> None:
    # 03:00 UTC = 08:30 IST, before 09:15 open.
    assert is_market_hours(datetime(2026, 8, 4, 3, 0, tzinfo=timezone.utc)) is False


def test_is_market_hours_false_after_close() -> None:
    # 10:30 UTC = 16:00 IST, after 15:30 close.
    assert is_market_hours(datetime(2026, 8, 4, 10, 30, tzinfo=timezone.utc)) is False


def test_is_market_hours_false_on_weekend() -> None:
    # Sat 2026-08-08 06:00 UTC = 11:30 IST, would be inside session hours on a weekday.
    assert is_market_hours(datetime(2026, 8, 8, 6, 0, tzinfo=timezone.utc)) is False


def test_is_market_hours_true_at_exact_open_and_close_boundaries() -> None:
    assert is_market_hours(datetime(2026, 8, 4, 3, 45, tzinfo=timezone.utc)) is True  # 09:15 IST
    assert is_market_hours(datetime(2026, 8, 4, 10, 0, tzinfo=timezone.utc)) is True  # 15:30 IST
