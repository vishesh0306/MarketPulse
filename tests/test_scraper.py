"""Tests for src/scraper — extraction logic against tests/fixtures/, plus a simulated
rate-limit/backoff scenario that doesn't require a live network call.
"""

from __future__ import annotations

import io
import sys
import threading
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bs4 import BeautifulSoup

from selenium.common.exceptions import TimeoutException, WebDriverException
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError

from src.scraper.anti_detection import _USER_AGENTS, is_soft_blocked
from src.scraper.rate_limiter import RateLimitedError, TokenBucketRateLimiter
from src.scraper.selenium_driver import rotate_user_agent
from src.scraper.twitter_scraper import (
    ParseError,
    ScrapeTimeoutError,
    _attempt_host,
    _extract_engagement,
    _scrape_one_hashtag,
    _SharedProgress,
    build_search_url,
    extract_tweet_fields,
    iter_result_pages,
    main,
    scrape_hashtag,
)
from src.utils.config_loader import ScraperConfig, load_settings

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


def test_rotate_user_agent_issues_cdp_override_with_known_ua() -> None:
    driver = MagicMock()
    ua = rotate_user_agent(driver)
    assert ua in _USER_AGENTS
    driver.execute_cdp_cmd.assert_called_once_with("Network.setUserAgentOverride", {"userAgent": ua})


def test_all_hashtags_is_core_then_related_deduped() -> None:
    cfg = ScraperConfig.model_construct(
        hashtags=["nifty50", "sensex"],
        related_hashtags=["nifty", "sensex", "nse"],  # 'sensex' overlaps the core list
    )
    assert cfg.all_hashtags == ["nifty50", "sensex", "nifty", "nse"]


def test_default_config_keeps_the_four_assignment_hashtags() -> None:
    core = load_settings().scraper.hashtags
    assert set(core) == {"nifty50", "sensex", "intraday", "banknifty"}


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


class _AlwaysReady:
    """Stands in for WebDriverWait: .until() succeeds immediately."""

    def __init__(self, driver: object, timeout: float) -> None:
        pass

    def until(self, condition: object) -> None:
        return None


def test_iter_result_pages_extracts_cards_from_page_source_without_per_card_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Card HTML comes from parsing the single page_source fetch (shared with the
    soft-block check) — not a separate find_elements() plus a per-card
    get_attribute("outerHTML") WebDriver round trip for every tweet card on the page."""
    monkeypatch.setattr("src.scraper.twitter_scraper.WebDriverWait", _AlwaysReady)

    driver = MagicMock()
    driver.page_source = FIXTURE_PATH.read_text(encoding="utf-8")
    driver.find_elements.return_value = []  # no "Load more" link -> stop after this page
    rate_limiter = TokenBucketRateLimiter(capacity=5, refill_rate_per_second=100.0)

    pages = list(iter_result_pages(driver, 1, _fake_pagination_config(), rate_limiter, ["not a real indicator"]))

    assert len(pages) == 1
    assert len(pages[0]) > 0
    # find_elements is only called once, for the "Load more" link check — never for
    # timeline-item cards, and never followed by a per-card get_attribute round trip.
    assert driver.find_elements.call_count == 1


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

    def _fake_attempt_host(driver, hashtag, host, cutoff, remaining_target, out_file, seen_ids, settings, progress=None):  # noqa: ARG001
        calls.append(host)
        return {"collected": 1, "parse_errors": 0, "errors": [], "backoff_triggered": False, "host_exhausted": False}

    monkeypatch.setattr("src.scraper.twitter_scraper._attempt_host", _fake_attempt_host)

    summary = scrape_hashtag(MagicMock(), "nifty50", 24, 10, tmp_path / "nifty50.jsonl", settings)

    assert calls == settings.scraper.nitter_hosts
    assert summary["collected"] == len(settings.scraper.nitter_hosts)


def _local_progress(target: int) -> _SharedProgress:
    """A _SharedProgress backed by plain in-process objects, for tests that don't need a
    real multiprocessing Manager."""
    return _SharedProgress(SimpleNamespace(value=0), threading.Lock(), target)


def test_shared_progress_add_and_remaining() -> None:
    progress = _local_progress(target=10)
    assert progress.remaining() == 10
    progress.add(4)
    assert progress.remaining() == 6
    progress.add(0)
    progress.add(-3)
    assert progress.remaining() == 6
    progress.add(20)
    assert progress.remaining() == 0


def test_scrape_hashtag_stops_at_run_wide_target_not_its_own_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a shared counter, a hashtag stops as soon as the *combined* run total hits
    the target — it doesn't keep failing over to more hosts chasing its own min_tweets."""
    settings = load_settings()
    progress = _local_progress(target=5)
    progress.add(3)  # another worker already collected 3

    calls: list[tuple[str, int]] = []

    def _fake_attempt_host(
        driver, hashtag, host, cutoff, remaining_target, out_file, seen_ids, settings, progress=None
    ):  # noqa: ARG001
        calls.append((host, remaining_target))
        progress.add(2)  # this host yields the final 2 -> run total 5
        return {"collected": 2, "parse_errors": 0, "errors": [], "backoff_triggered": False, "host_exhausted": False}

    monkeypatch.setattr("src.scraper.twitter_scraper._attempt_host", _fake_attempt_host)

    summary = scrape_hashtag(MagicMock(), "nifty50", 24, 100, tmp_path / "nifty50.jsonl", settings, progress)

    assert len(calls) == 1  # stopped after one host, target met run-wide
    assert calls[0][1] == 2  # remaining passed in = target(5) - already(3)
    assert summary["collected"] == 2


class _SerialExecutor:
    """Stand-in for ProcessPoolExecutor that runs submitted callables inline, so main()
    can be exercised without spawning (unpicklable) worker processes in a test."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> "_SerialExecutor":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def submit(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 — mirror Executor: exceptions land on the future
            future.set_exception(exc)
        return future


def _run_main_with_collected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, per_hashtag_collected: int, argv: list[str]
) -> int:
    """Runs main() with the worker pool stubbed to a fixed per-hashtag collected count.
    Returns the process exit code (0 when main() returns normally)."""

    def _fake_scrape_one(hashtag, hours, min_tweets, output_path, settings, driver_path):  # noqa: ARG001
        return {
            "hashtag": hashtag,
            "collected": per_hashtag_collected,
            "unique_ids": per_hashtag_collected,
            "parse_errors": 0,
            "errors": [],
            "backoff_triggered": False,
            "hosts_tried": ["nitter.example.com"],
        }

    monkeypatch.setattr("src.scraper.twitter_scraper.ProcessPoolExecutor", _SerialExecutor)
    monkeypatch.setattr("src.scraper.twitter_scraper.resolve_driver_path", lambda: "fake-driver-path")
    monkeypatch.setattr("src.scraper.twitter_scraper._scrape_one_hashtag", _fake_scrape_one)
    monkeypatch.setattr("src.scraper.twitter_scraper.write_run_summary", lambda *a, **k: tmp_path / "summary.json")
    monkeypatch.setattr(sys, "argv", ["twitter_scraper", *argv])

    try:
        main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_main_exits_nonzero_on_tweet_shortfall(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A run that collects fewer than --min-tweets must exit non-zero, so run_pipeline.sh
    stops instead of carrying a thin corpus through to the signal stage."""
    code = _run_main_with_collected(
        monkeypatch, tmp_path, per_hashtag_collected=1, argv=["--hashtags", "nifty50,sensex", "--min-tweets", "100"]
    )
    assert code == 1


def test_main_allow_shortfall_flag_exits_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--allow-shortfall opts out of the hard minimum for exploratory runs."""
    code = _run_main_with_collected(
        monkeypatch,
        tmp_path,
        per_hashtag_collected=1,
        argv=["--hashtags", "nifty50,sensex", "--min-tweets", "100", "--allow-shortfall"],
    )
    assert code == 0


def test_main_exits_zero_when_target_met(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    code = _run_main_with_collected(
        monkeypatch, tmp_path, per_hashtag_collected=50, argv=["--hashtags", "nifty50,sensex", "--min-tweets", "100"]
    )
    assert code == 0


def test_main_survives_one_worker_crashing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If one hashtag's worker raises an unexpected exception, the run must still record
    a summary for the hashtags that succeeded rather than aborting with a traceback."""
    written: dict[str, object] = {}

    def _flaky_scrape_one(hashtag, hours, min_tweets, output_path, settings, driver_path):  # noqa: ARG001
        if hashtag == "sensex":
            raise RuntimeError("worker blew up")
        return {
            "hashtag": hashtag,
            "collected": 40,
            "unique_ids": 40,
            "parse_errors": 0,
            "errors": [],
            "backoff_triggered": False,
            "hosts_tried": ["nitter.example.com"],
        }

    monkeypatch.setattr("src.scraper.twitter_scraper.ProcessPoolExecutor", _SerialExecutor)
    monkeypatch.setattr("src.scraper.twitter_scraper.resolve_driver_path", lambda: "fake-driver-path")
    monkeypatch.setattr("src.scraper.twitter_scraper._scrape_one_hashtag", _flaky_scrape_one)
    monkeypatch.setattr(
        "src.scraper.twitter_scraper.write_run_summary",
        lambda log_dir, phase, payload: written.update(payload) or (tmp_path / "summary.json"),
    )
    monkeypatch.setattr(sys, "argv", ["twitter_scraper", "--hashtags", "nifty50,sensex", "--min-tweets", "100"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1  # 40 < 100 -> still a shortfall
    assert written["total_collected"] == 40
    per_hashtag = {entry["hashtag"]: entry for entry in written["per_hashtag"]}
    assert per_hashtag["nifty50"]["collected"] == 40
    assert per_hashtag["sensex"]["collected"] == 0
    assert per_hashtag["sensex"]["errors"]
