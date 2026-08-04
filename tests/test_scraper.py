"""Tests for src/scraper — extraction logic verified offline against tests/fixtures/,
plus a simulated rate-limit/backoff scenario that doesn't require a live network call.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from selenium.common.exceptions import WebDriverException

from src.scraper.anti_detection import is_soft_blocked
from src.scraper.rate_limiter import TokenBucketRateLimiter
from src.scraper.twitter_scraper import (
    ParseError,
    _extract_engagement,  # testing this internal directly is the point of the regression test below
    _scrape_one_hashtag,
    build_search_url,
    extract_tweet_fields,
)
from src.utils.config_loader import load_settings

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "search_results.html"


@pytest.fixture(scope="module")
def fixture_cards() -> list[str]:
    html = FIXTURE_PATH.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    return [str(card) for card in soup.find_all(class_="timeline-item")]


def test_fixture_has_tweet_cards(fixture_cards: list[str]) -> None:
    assert len(fixture_cards) > 0


def test_extract_tweet_fields_all_cards_parse(fixture_cards: list[str]) -> None:
    for card_html in fixture_cards:
        record = extract_tweet_fields(card_html, "nifty50")
        assert record["tweet_id"].isdigit()
        assert record["username"]
        assert record["created_at"].endswith("+00:00")
        assert isinstance(record["likes"], int)
        assert isinstance(record["retweets"], int)
        assert isinstance(record["replies"], int)
        assert isinstance(record["mentions"], list)
        assert isinstance(record["hashtags"], list)
        assert record["source_hashtag"] == "nifty50"


def test_extract_tweet_fields_first_card_values(fixture_cards: list[str]) -> None:
    record = extract_tweet_fields(fixture_cards[0], "nifty50")
    assert record["tweet_id"] == "2084625182103343125"
    assert record["username"] == "AshishD39029716"
    assert record["created_at"] == "2026-08-04T12:59:00+00:00"
    assert "SEBI_India" in record["mentions"]
    assert "nifty50" in record["hashtags"]
    assert "RahulGandhiPlease" not in record["mentions"]


def test_extract_tweet_fields_mentions_split_across_adjacent_links(fixture_cards: list[str]) -> None:
    """A mention link immediately followed by text (no source whitespace) must not merge
    into one token — regression test for the get_text(strip=True) fragment-merging bug."""
    record = extract_tweet_fields(fixture_cards[0], "nifty50")
    assert "RahulGandhi" in record["mentions"]


def test_extract_tweet_fields_raises_parse_error_on_empty_card() -> None:
    with pytest.raises(ParseError):
        extract_tweet_fields("<div class='timeline-item'></div>", "nifty50")


def test_extract_engagement_finds_icon_span_not_wrapper_div(fixture_cards: list[str]) -> None:
    """Regression test: the icon lookup must land on <span class="icon-comment"> etc.,
    not the wrapping <div class="icon-container"> (which also starts with "icon-" and,
    being the shallower match, was silently winning the lookup and zeroing every count)."""
    soup = BeautifulSoup(fixture_cards[11], "html.parser")
    engagement = _extract_engagement(soup)
    assert engagement == {"replies": 26, "retweets": 0, "likes": 8}


def test_build_search_url() -> None:
    url = build_search_url("nifty50", "nitter.example.com", "https://{host}/search?f=tweets&q=%23{hashtag}")
    assert url == "https://nitter.example.com/search?f=tweets&q=%23nifty50"


def test_scrape_one_hashtag_handles_driver_failure_gracefully(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker-pool task (one hashtag) that fails to even start a browser session must
    return a well-formed summary dict, not raise — otherwise one hashtag's failure would
    take down the whole ProcessPoolExecutor.map/as_completed loop in main()."""

    def _raise_driver_error(headless: bool, driver_path: str):  # noqa: ARG001
        raise WebDriverException("could not start browser")

    monkeypatch.setattr("src.scraper.twitter_scraper.get_driver", _raise_driver_error)

    settings = load_settings()
    summary = _scrape_one_hashtag("nifty50", 24, 10, tmp_path / "nifty50.jsonl", settings, "fake-driver-path")

    assert summary["hashtag"] == "nifty50"
    assert summary["collected"] == 0
    assert summary["errors"]


def test_is_soft_blocked_detects_known_indicators() -> None:
    indicators = ["Instance has been rate limited", "Verifying your browser"]
    assert is_soft_blocked("<html>Instance has been rate limited</html>", indicators)
    assert not is_soft_blocked("<html>normal search results</html>", indicators)


def test_rate_limiter_backoff_increases_with_attempt() -> None:
    """Simulates a 429/soft-block scenario: each retry attempt should back off longer,
    capped at max_seconds, without raising or hanging."""
    limiter = TokenBucketRateLimiter(capacity=5, refill_rate_per_second=100.0)
    delay_1 = limiter.backoff(attempt=1, base_seconds=0.01, max_seconds=1.0, multiplier=2.0)
    delay_3 = limiter.backoff(attempt=3, base_seconds=0.01, max_seconds=1.0, multiplier=2.0)
    assert delay_1 > 0
    assert delay_3 > delay_1
    assert delay_3 <= 1.0 * 1.2  # respects max_seconds cap plus jitter headroom
