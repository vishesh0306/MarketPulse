"""User-agent rotation, randomized scroll timing/distance, and soft-block detection."""

from __future__ import annotations

import random

# Recent, realistic desktop Chrome UAs across common OSes — rotated rather than generated,
# since generated/fake UAs are themselves a detection signal.
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]


def random_user_agent() -> str:
    """Returns a randomized, realistic desktop browser user-agent string."""
    return random.choice(_USER_AGENTS)


def random_scroll_amount(min_pixels: int, max_pixels: int) -> int:
    """Returns a randomized scroll distance in pixels."""
    return random.randint(min_pixels, max_pixels)


def is_soft_blocked(page_source: str, indicators: list[str]) -> bool:
    """Returns True if the page source matches a known soft-block/error indicator."""
    lowered = page_source.lower()
    return any(indicator.lower() in lowered for indicator in indicators)
