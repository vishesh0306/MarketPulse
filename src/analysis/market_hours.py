"""NSE market-hours check, for filtering tweet buckets to the trading session.

The scraper and storage layers work entirely in UTC (correctly — timestamps should be
unambiguous), but nothing downstream converts to IST or knows when NSE is actually open.
Off-hours chatter is real data and stays in data/processed, but a *trading* signal bucketed
across a day when 57% of the buckets fall outside the 09:15-15:30 IST session is mostly
measuring conversation volume, not market-hours sentiment — and those off-hours buckets
are exactly the ones with too few tweets for a meaningful confidence interval.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
NSE_OPEN = (9, 15)
NSE_CLOSE = (15, 30)


def is_market_hours(timestamp_utc: datetime) -> bool:
    """True if timestamp_utc falls within NSE's Monday-Friday 09:15-15:30 IST session."""
    local = timestamp_utc.astimezone(IST)
    if local.weekday() >= 5:
        return False
    open_time = local.replace(hour=NSE_OPEN[0], minute=NSE_OPEN[1], second=0, microsecond=0)
    close_time = local.replace(hour=NSE_CLOSE[0], minute=NSE_CLOSE[1], second=0, microsecond=0)
    return open_time <= local <= close_time
