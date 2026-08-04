"""Per-hashtag Twitter/X search scraping: scroll-based pagination + field extraction.

No official/paid Twitter API and no tweepy — Selenium against the public search UI only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterator

from selenium.webdriver.chrome.webdriver import WebDriver

from src.scraper.rate_limiter import RateLimitedError  # re-exported for callers


class ScrapeTimeoutError(Exception):
    """Raised when a page element does not appear within the configured explicit wait."""


class ParseError(Exception):
    """Raised when a tweet card's expected fields cannot be extracted from the DOM."""


def build_search_url(hashtag: str, url_template: str) -> str:
    """Renders the search URL for a given hashtag from the configured template."""
    raise NotImplementedError


def extract_tweet_fields(card_html: str) -> dict[str, Any]:
    """Parses one tweet-card's HTML into username, timestamp, text, engagement, mentions, hashtags.

    Pure function so it's testable offline against tests/fixtures/search_results.html.
    """
    raise NotImplementedError


def scrape_hashtag(
    driver: WebDriver,
    hashtag: str,
    hours_lookback: int,
    min_tweets: int,
    output_path: Path,
) -> dict[str, Any]:
    """Scrolls the search UI for one hashtag, streaming deduplicated tweets to output_path as JSONL.

    Returns a per-hashtag summary dict (collected count, errors) for the run-summary log.
    """
    raise NotImplementedError


def iter_scroll_batches(driver: WebDriver, max_scrolls: int) -> Iterator[list[str]]:
    """Yields batches of newly-rendered tweet-card HTML as the page is scrolled."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Indian market-hashtag tweets via Selenium.")
    parser.add_argument("--hashtags", type=str, required=True, help="Comma-separated hashtags, no '#'.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours.")
    parser.add_argument("--min-tweets", type=int, default=2000, help="Minimum total tweets to collect.")
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
