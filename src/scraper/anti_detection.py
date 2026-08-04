"""User-agent rotation, randomized scroll timing/distance, and soft-block detection."""

from __future__ import annotations


def random_user_agent() -> str:
    """Returns a randomized, realistic desktop browser user-agent string."""
    raise NotImplementedError


def random_scroll_amount(min_pixels: int, max_pixels: int) -> int:
    """Returns a randomized scroll distance in pixels."""
    raise NotImplementedError


def is_soft_blocked(page_source: str, indicators: list[str]) -> bool:
    """Returns True if the page source matches a known soft-block/error indicator."""
    raise NotImplementedError
