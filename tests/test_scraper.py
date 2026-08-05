"""Tests for src/scraper — extraction logic against tests/fixtures/, plus a simulated
rate-limit/backoff scenario that doesn't require a live network call.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bs4 import BeautifulSoup

from selenium.common.exceptions import TimeoutException, WebDriverException
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError

from src.scraper.anti_detection import is_soft_blocked
from src.scraper.rate_limiter import RateLimitedError, TokenBucketRateLimiter
from src.scraper.twitter_scraper import (
    ParseError,
    ScrapeTimeoutError,
    _attempt_host,
    _extract_engagement,
    _scrape_one_hashtag,
    build_search_url,
    extract_tweet_fields,
    iter_result_pages,
    scrape_hashtag,
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
    into one token."""
    record = extract_tweet_fields(fixture_cards[0], "nifty50")
    assert "RahulGandhi" in record["mentions"]


def test_extract_tweet_fields_raises_parse_error_on_empty_card() -> None:
    with pytest.raises(ParseError):
        extract_tweet_fields("<div class='timeline-item'></div>", "nifty50")


def test_extract_engagement_finds_icon_span_not_wrapper_div(fixture_cards: list[str]) -> None:
    """The icon lookup must land on <span class="icon-comment"> etc., not the wrapping
    <div class="icon-container"> (which also starts with "icon-" and, being the shallower
    match, would otherwise win and zero every count)."""
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
    """Each retry attempt should back off longer, capped at max_seconds."""
    limiter = TokenBucketRateLimiter(capacity=5, refill_rate_per_second=100.0)
    delay_1 = limiter.backoff(attempt=1, base_seconds=0.01, max_seconds=1.0, multiplier=2.0)
    delay_3 = limiter.backoff(attempt=3, base_seconds=0.01, max_seconds=1.0, multiplier=2.0)
    assert delay_1 > 0
    assert delay_3 > delay_1
    assert delay_3 <= 1.0 * 1.2  # respects max_seconds cap plus jitter headroom


class _AlwaysTimesOut:
    """Stands in for WebDriverWait: .until() always times out, like a page that never
    renders .timeline."""

    def __init__(self, driver: object, timeout: float) -> None:
        pass

    def until(self, condition: object) -> None:
        raise TimeoutException("no such element: .timeline")


def _fake_pagination_config() -> SimpleNamespace:
    return SimpleNamespace(page_render_timeout_seconds=1, min_pause_seconds=0, max_pause_seconds=0)


def test_iter_result_pages_classifies_challenge_page_as_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full-page anti-bot challenge never renders .timeline, so it should still be
    classified as retryable rather than discarding the host outright."""
    monkeypatch.setattr("src.scraper.twitter_scraper.WebDriverWait", _AlwaysTimesOut)

    driver = MagicMock()
    driver.page_source = "<html><title>Making sure you're not a bot!</title></html>"
    rate_limiter = TokenBucketRateLimiter(capacity=5, refill_rate_per_second=100.0)

    with pytest.raises(RateLimitedError):
        list(iter_result_pages(driver, 1, _fake_pagination_config(), rate_limiter, ["not a bot"]))


def test_iter_result_pages_genuinely_dead_host_stays_scrape_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A render timeout with no recognizable challenge page should still raise
    ScrapeTimeoutError (no retry) so a dead host fails over quickly."""
    monkeypatch.setattr("src.scraper.twitter_scraper.WebDriverWait", _AlwaysTimesOut)

    driver = MagicMock()
    driver.page_source = "<html><body>totally unrelated content</body></html>"
    rate_limiter = TokenBucketRateLimiter(capacity=5, refill_rate_per_second=100.0)

    with pytest.raises(ScrapeTimeoutError):
        list(iter_result_pages(driver, 1, _fake_pagination_config(), rate_limiter, ["not a bot"]))


def test_attempt_host_classifies_urllib3_read_timeout_as_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raw urllib3 ReadTimeoutError from driver.get() (the socket-level timeout on the
    webdriver connection, distinct from Selenium's own TimeoutException) should back off
    and exhaust the host like any other stall, not propagate and crash the whole scrape."""

    def _raising_page_iter(*args: object, **kwargs: object):
        raise URLLib3ReadTimeoutError(None, "http://example.test", "read timed out")
        yield []  # pragma: no cover

    monkeypatch.setattr("src.scraper.twitter_scraper._new_page_iter", lambda *a, **k: _raising_page_iter())

    settings = load_settings().model_copy(deep=True)
    settings.scraper.rate_limiter.backoff_base_seconds = 0.001
    settings.scraper.rate_limiter.backoff_max_seconds = 0.01

    driver = MagicMock()
    result = _attempt_host(
        driver, "nifty50", "nitter.example.com", datetime.now(timezone.utc), 10, io.StringIO(), set(), settings
    )

    assert result["host_exhausted"] is True
    assert result["backoff_triggered"] is True


def test_scrape_hashtag_tries_next_host_even_when_previous_not_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host that returns cleanly (ran out of pages, not rate-limited) but with far
    fewer tweets than min_tweets must still fail over to the next configured host,
    rather than treating "not exhausted" as "done, stop trying more hosts"."""
    settings = load_settings()
    calls: list[str] = []

    def _fake_attempt_host(driver, hashtag, host, cutoff, remaining_target, out_file, seen_ids, settings):  # noqa: ARG001
        calls.append(host)
        return {"collected": 1, "parse_errors": 0, "errors": [], "backoff_triggered": False, "host_exhausted": False}

    monkeypatch.setattr("src.scraper.twitter_scraper._attempt_host", _fake_attempt_host)

    summary = scrape_hashtag(MagicMock(), "nifty50", 24, 10, tmp_path / "nifty50.jsonl", settings)

    assert calls == settings.scraper.nitter_hosts
    assert summary["collected"] == len(settings.scraper.nitter_hosts)
