"""Per-hashtag X/Twitter search scraping via Nitter — an open-source, login-free HTML
front-end for X.com's public content. x.com's own search UI redirects unauthenticated
requests to a login wall (verified 2026-08-04), so this targets Nitter instances instead,
with automatic failover across configured hosts when one is down or soft-blocked.

No official/paid Twitter API and no tweepy — Selenium against a public search UI only.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from bs4 import BeautifulSoup
from bs4.element import Tag
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.scraper.anti_detection import is_soft_blocked
from src.scraper.rate_limiter import RateLimitedError, TokenBucketRateLimiter
from src.scraper.selenium_driver import get_driver, resolve_driver_path
from src.utils.config_loader import Settings, load_settings
from src.utils.logger import get_logger, write_run_summary

logger = get_logger("twitter_scraper")

_STATUS_LINK_RE = re.compile(r"^/([A-Za-z0-9_]{1,15})/status/(\d+)")
_MENTION_RE = re.compile(r"@(\w+)")
_HASHTAG_RE = re.compile(r"#(\w+)")
_COUNT_SUFFIX = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
_WHITESPACE_RE = re.compile(r"\s+")
_MAX_RATE_LIMIT_RETRIES = 3
_STAT_KINDS = {"comment": "replies", "retweet": "retweets", "heart": "likes"}


class ScrapeTimeoutError(Exception):
    """Raised when a page element does not appear within the configured explicit wait."""


class ParseError(Exception):
    """Raised when a tweet card's expected fields cannot be extracted from the DOM."""


class AllHostsExhaustedError(Exception):
    """Raised when every configured Nitter host is down or soft-blocked for a hashtag."""


def build_search_url(hashtag: str, host: str, path_template: str) -> str:
    """Renders the search URL for a given hashtag and Nitter host from the configured template."""
    return path_template.format(host=host, hashtag=hashtag)


def _parse_count(text: str) -> int:
    """Converts an engagement count string ('1.2K', '3,401', '', '34') into an int."""
    text = text.strip().replace(",", "")
    if not text:
        return 0
    match = re.match(r"^([\d.]+)([KMB]?)$", text.upper())
    if not match:
        digits = re.sub(r"[^\d]", "", text)
        return int(digits) if digits else 0
    number, suffix = match.groups()
    value = float(number)
    if suffix in _COUNT_SUFFIX:
        value *= _COUNT_SUFFIX[suffix]
    return int(value)


def _parse_nitter_timestamp(title: str) -> datetime:
    """Parses Nitter's tweet-date title attribute, e.g. 'Aug 4, 2026 · 12:59 PM UTC'."""
    normalized = _WHITESPACE_RE.sub(" ", title.replace("·", " ")).strip()
    normalized = normalized.removesuffix(" UTC").strip()
    naive = datetime.strptime(normalized, "%b %d, %Y %I:%M %p")
    return naive.replace(tzinfo=timezone.utc)


def _extract_engagement(card: Tag) -> dict[str, int]:
    counts = {"replies": 0, "retweets": 0, "likes": 0}
    for stat in card.find_all(class_="tweet-stat"):
        # Must be restricted to <span> tags: the wrapping <div class="icon-container">
        # also starts with "icon-" and, being the shallower match, would otherwise be
        # returned first — silently defeating every lookup in _STAT_KINDS below.
        icon = stat.find("span", class_=re.compile(r"^icon-"))
        if icon is None:
            continue
        icon_class = next((c for c in icon.get("class", []) if c.startswith("icon-")), None)
        if icon_class is None:
            continue
        kind = _STAT_KINDS.get(icon_class.removeprefix("icon-"))
        if kind is None:
            continue
        counts[kind] = _parse_count(stat.get_text(strip=True))
    return counts


def extract_tweet_fields(card_html: str, source_hashtag: str) -> dict[str, Any]:
    """Parses one Nitter timeline-item's HTML into username, timestamp, text, engagement,
    mentions, and hashtags. Pure function so it's testable offline against
    tests/fixtures/search_results.html.
    """
    soup = BeautifulSoup(card_html, "html.parser")
    card = soup.find(class_="timeline-item") or soup

    permalink = card.find("a", class_="tweet-link", href=True)
    if permalink is None:
        permalink = next(
            (a for a in card.find_all("a", href=True) if _STATUS_LINK_RE.match(a["href"])), None
        )
    match = _STATUS_LINK_RE.match(permalink["href"]) if permalink else None
    if match is None:
        raise ParseError("no status permalink (username/tweet_id) found in tweet card")
    username, tweet_id = match.group(1), match.group(2)

    date_span = card.find(class_="tweet-date")
    time_link = date_span.find("a") if date_span else None
    if time_link is None or not time_link.get("title"):
        raise ParseError(f"no timestamp found for tweet {tweet_id}")
    try:
        created_at = _parse_nitter_timestamp(time_link["title"])
    except ValueError as exc:
        raise ParseError(f"unparseable timestamp for tweet {tweet_id}: {time_link['title']!r}") from exc

    content_node = card.find(class_="tweet-content")
    if content_node:
        # separator=" " prevents adjacent inline tags (e.g. a mention link followed
        # directly by text) from merging into one token when strip=True trims each
        # fragment individually — see extraction fixture test for the regression case.
        text = _WHITESPACE_RE.sub(" ", content_node.get_text(separator=" ", strip=True)).strip()
    else:
        text = ""

    engagement = _extract_engagement(card)
    mentions = sorted(set(_MENTION_RE.findall(text)))
    hashtags = sorted({tag.lower() for tag in _HASHTAG_RE.findall(text)})

    return {
        "tweet_id": tweet_id,
        "username": username,
        "created_at": created_at.isoformat(),
        "text": text,
        "likes": engagement["likes"],
        "retweets": engagement["retweets"],
        "replies": engagement["replies"],
        "mentions": mentions,
        "hashtags": hashtags,
        "source_hashtag": source_hashtag,
    }


def iter_result_pages(
    driver: WebDriver,
    max_pages: int,
    pagination_config: Any,
    rate_limiter: TokenBucketRateLimiter,
    soft_block_indicators: list[str],
) -> Iterator[list[str]]:
    """Yields batches of tweet-card HTML, one batch per Nitter results page, following the
    cursor-based 'Load more' link until pages run out or max_pages is reached.
    """
    for _ in range(max_pages):
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "timeline")))
        except TimeoutException as exc:
            raise ScrapeTimeoutError("timed out waiting for search results to render") from exc

        if is_soft_blocked(driver.page_source, soft_block_indicators):
            raise RateLimitedError("soft-block/anti-bot indicator detected in page source")

        cards = driver.find_elements(By.CLASS_NAME, "timeline-item")
        cards_html = [html for card in cards if (html := card.get_attribute("outerHTML")) is not None]
        if cards_html:
            yield cards_html

        next_href = None
        for link in driver.find_elements(By.CSS_SELECTOR, "div.show-more a"):
            href = link.get_attribute("href")
            if href and "cursor=" in href:
                next_href = href
                break
        if next_href is None:
            return

        rate_limiter.acquire(pagination_config.min_pause_seconds, pagination_config.max_pause_seconds)
        driver.get(next_href)


def _process_card_batch(
    batch: list[str],
    hashtag: str,
    out_file: Any,
    seen_ids: set[str],
    cutoff: datetime,
) -> tuple[int, int, bool, list[str]]:
    """Extracts, dedupes, and writes one page's worth of tweet cards.

    Returns (newly_collected, parse_errors, reached_cutoff, errors).
    """
    collected = 0
    parse_errors = 0
    reached_cutoff = False
    errors: list[str] = []
    for card_html in batch:
        try:
            record = extract_tweet_fields(card_html, hashtag)
        except ParseError as exc:
            parse_errors += 1
            errors.append(str(exc))
            continue

        if record["tweet_id"] in seen_ids:
            continue
        seen_ids.add(record["tweet_id"])

        created_at = datetime.fromisoformat(record["created_at"])
        if created_at < cutoff:
            # Results are newest-first, so one tweet past the lookback window means
            # everything after it is too — stop paging instead of fruitlessly fetching
            # pages that can't count.
            reached_cutoff = True
            break

        record["collected_at"] = datetime.now(timezone.utc).isoformat()
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_file.flush()
        collected += 1
    return collected, parse_errors, reached_cutoff, errors


def _new_page_iter(driver: WebDriver, rate_limiter: TokenBucketRateLimiter, settings: Settings) -> Iterator[list[str]]:
    return iter_result_pages(
        driver,
        settings.scraper.pagination.max_pages_per_session,
        settings.scraper.pagination,
        rate_limiter,
        settings.scraper.anti_detection.soft_block_indicators,
    )


def _handle_host_stall(
    exc: Exception, hashtag: str, host: str, retry: int, rate_limiter: TokenBucketRateLimiter, settings: Settings
) -> bool:
    """Logs and backs off for a rate-limit/stall exception (a stalled page load, e.g. a
    slow third-party image/media subresource, is treated the same as a soft-block).

    Returns True if the host should be considered exhausted (retries used up, caller
    should move on to the next host), False if the caller should retry this host.
    """
    if retry > _MAX_RATE_LIMIT_RETRIES:
        logger.warning(
            "host exhausted after max retries, trying next host",
            extra={"extra_fields": {"hashtag": hashtag, "host": host, "retry": retry}},
        )
        return True
    delay = rate_limiter.backoff(
        retry,
        settings.scraper.rate_limiter.backoff_base_seconds,
        settings.scraper.rate_limiter.backoff_max_seconds,
        settings.scraper.rate_limiter.backoff_multiplier,
    )
    logger.warning(
        "rate limited or page load stalled; backed off and resuming",
        extra={
            "extra_fields": {
                "hashtag": hashtag,
                "host": host,
                "retry": retry,
                "delay_seconds": delay,
                "cause": type(exc).__name__,
            }
        },
    )
    return False


def _attempt_host(
    driver: WebDriver,
    hashtag: str,
    host: str,
    cutoff: datetime,
    remaining_target: int,
    out_file: Any,
    seen_ids: set[str],
    settings: Settings,
) -> dict[str, Any]:
    """Pages through one host's search results for a hashtag until remaining_target
    tweets are collected, the 24h cutoff is hit, or the host proves unhealthy (retries
    exhausted on repeated soft-blocks/stalls).
    """
    url = build_search_url(hashtag, host, settings.scraper.search_path_template)
    rate_limiter = TokenBucketRateLimiter(
        settings.scraper.rate_limiter.bucket_capacity,
        settings.scraper.rate_limiter.refill_rate_per_second,
    )

    try:
        driver.get(url)
    except WebDriverException as exc:
        return {
            "collected": 0,
            "parse_errors": 0,
            "errors": [f"{host}: navigation failed: {exc}"],
            "backoff_triggered": False,
            "host_exhausted": False,
        }

    collected = 0
    parse_errors = 0
    errors: list[str] = []
    backoff_triggered = False
    host_exhausted = False
    retry = 0
    page_iter = _new_page_iter(driver, rate_limiter, settings)

    while True:
        try:
            for batch in page_iter:
                batch_collected, batch_parse_errors, reached_cutoff, batch_errors = _process_card_batch(
                    batch, hashtag, out_file, seen_ids, cutoff
                )
                collected += batch_collected
                parse_errors += batch_parse_errors
                errors.extend(batch_errors)
                if collected >= remaining_target or reached_cutoff:
                    break
            break
        except (RateLimitedError, TimeoutException) as exc:
            backoff_triggered = True
            retry += 1
            errors.append(f"{host}: {exc}")
            host_exhausted = _handle_host_stall(exc, hashtag, host, retry, rate_limiter, settings)
            if host_exhausted:
                break
            page_iter = _new_page_iter(driver, rate_limiter, settings)
        except ScrapeTimeoutError as exc:
            errors.append(f"{host}: {exc}")
            host_exhausted = True
            break

    return {
        "collected": collected,
        "parse_errors": parse_errors,
        "errors": errors,
        "backoff_triggered": backoff_triggered,
        "host_exhausted": host_exhausted,
    }


def scrape_hashtag(
    driver: WebDriver,
    hashtag: str,
    hours_lookback: int,
    min_tweets: int,
    output_path: Path,
    settings: Settings,
) -> dict[str, Any]:
    """Pages through Nitter search results for one hashtag, streaming deduplicated tweets to
    output_path as JSONL, failing over to the next configured host on a soft-block.

    Returns a per-hashtag summary dict (collected count, errors) for the run-summary log.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_lookback)
    seen_ids: set[str] = set()
    collected = 0
    parse_errors = 0
    errors: list[str] = []
    backoff_triggered = False
    hosts_tried: list[str] = []

    with output_path.open("a", encoding="utf-8") as out_file:
        for host in settings.scraper.nitter_hosts:
            hosts_tried.append(host)
            result = _attempt_host(
                driver, hashtag, host, cutoff, min_tweets - collected, out_file, seen_ids, settings
            )
            collected += result["collected"]
            parse_errors += result["parse_errors"]
            errors.extend(result["errors"])
            backoff_triggered = backoff_triggered or result["backoff_triggered"]

            if collected >= min_tweets or not result["host_exhausted"]:
                break
            logger.info(
                "falling back to next Nitter host",
                extra={"extra_fields": {"hashtag": hashtag, "exhausted_host": host}},
            )

    return {
        "hashtag": hashtag,
        "collected": collected,
        "unique_ids": len(seen_ids),
        "parse_errors": parse_errors,
        "errors": errors,
        "backoff_triggered": backoff_triggered,
        "hosts_tried": hosts_tried,
    }


def _scrape_one_hashtag(
    hashtag: str, hours: int, min_tweets: int, output_path: Path, settings: Settings, driver_path: str
) -> dict[str, Any]:
    """Runs one hashtag's full scrape in its own browser session — the unit of work
    submitted to the worker pool. Defined at module level (not nested/lambda) so it's
    picklable for ProcessPoolExecutor under Windows' 'spawn' start method.

    Safe to run concurrently with other hashtags: each call gets its own Selenium
    session, its own rate limiter, its own in-memory dedup sets (scoped inside
    scrape_hashtag), and writes only to its own hashtag-specific output file — there is
    no mutable state shared across hashtags that would need locking. driver_path is
    pre-resolved once by the caller (see main()) rather than re-resolved here, since
    ChromeDriverManager's cache isn't safe under several workers touching it at once.
    """
    logger.info("starting hashtag scrape", extra={"extra_fields": {"hashtag": hashtag}})
    try:
        with get_driver(headless=settings.scraper.headless, driver_path=driver_path) as driver:
            return scrape_hashtag(driver, hashtag, hours, min_tweets, output_path, settings)
    except (ScrapeTimeoutError, WebDriverException) as exc:
        logger.error(
            "hashtag scrape failed",
            extra={"extra_fields": {"hashtag": hashtag, "error": str(exc)}},
        )
        return {
            "hashtag": hashtag,
            "collected": 0,
            "unique_ids": 0,
            "parse_errors": 0,
            "errors": [str(exc)],
            "backoff_triggered": False,
            "hosts_tried": [],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Indian market-hashtag tweets via Nitter/Selenium.")
    parser.add_argument("--hashtags", type=str, required=True, help="Comma-separated hashtags, no '#'.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours.")
    parser.add_argument("--min-tweets", type=int, default=2000, help="Minimum total tweets to collect.")
    parser.add_argument("--workers", type=int, default=None, help="Worker pool size (default: config value).")
    args = parser.parse_args()

    settings = load_settings()
    hashtags = [tag.strip().lstrip("#") for tag in args.hashtags.split(",") if tag.strip()]
    per_hashtag_target = max(1, args.min_tweets // len(hashtags))

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = Path(settings.storage.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    worker_count = max(1, min(args.workers or settings.scraper.worker_pool_size, len(hashtags)))

    # Resolved once, here, before any worker starts — see get_driver()'s docstring for
    # why concurrent workers each resolving their own driver path is unsafe.
    driver_path = resolve_driver_path()

    summaries: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _scrape_one_hashtag,
                hashtag,
                args.hours,
                per_hashtag_target,
                raw_dir / f"{hashtag}_{run_timestamp}.jsonl",
                settings,
                driver_path,
            ): hashtag
            for hashtag in hashtags
        }
        for future in as_completed(futures):
            summaries.append(future.result())

    total_collected = sum(summary["collected"] for summary in summaries)

    summary_path = write_run_summary(
        settings.logging.dir,
        "scraper",
        {
            "run_timestamp": run_timestamp,
            "hashtags": hashtags,
            "min_tweets_target": args.min_tweets,
            "worker_pool_size": worker_count,
            "total_collected": total_collected,
            "per_hashtag": summaries,
        },
    )
    logger.info(
        "scrape run complete",
        extra={"extra_fields": {"total_collected": total_collected, "summary_path": str(summary_path)}},
    )


if __name__ == "__main__":
    main()
