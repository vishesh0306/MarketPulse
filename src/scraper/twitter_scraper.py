"""Per-hashtag X/Twitter search scraping via Nitter — an open-source, login-free HTML
front-end for X.com's public content, used since x.com's own search UI requires a login.
Automatic failover across configured hosts when one is down or soft-blocked.

No official/paid Twitter API and no tweepy — Selenium against a public search UI only.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import re
import sys
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
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError

from src.scraper.anti_detection import is_soft_blocked
from src.scraper.rate_limiter import RateLimitedError, TokenBucketRateLimiter
from src.scraper.selenium_driver import get_driver, resolve_driver_path, rotate_user_agent
from src.utils.config_loader import Settings, load_settings
from src.utils.logger import get_logger, set_level, write_run_summary

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


class _SharedProgress:
    """A run-wide view of how many tweets have been collected so far, shared across the
    per-hashtag worker processes.

    Without this, each worker chases a fixed 1/N slice of --min-tweets and a dense
    hashtag stops at its slice while a sparse one falls short, so the run total lands
    under target with headroom left unused. With it, every worker keeps collecting until
    the *combined* total reaches the target. Backed by a multiprocessing Manager so the
    proxies survive being passed to workers under the 'spawn' start method.
    """

    def __init__(self, counter: Any, lock: Any, target: int) -> None:
        self._counter = counter
        self._lock = lock
        self.target = target

    def add(self, n: int) -> None:
        if n <= 0:
            return
        # += on a manager proxy is read-modify-write across a socket, so it needs the
        # lock even though each worker only ever adds.
        with self._lock:
            self._counter.value += n

    def remaining(self) -> int:
        return max(0, self.target - self._counter.value)


# Set once per worker process by the pool initializer; None in the parent and in any
# direct (non-pool) call, where scrape_hashtag falls back to its local min_tweets cap.
_WORKER_PROGRESS: _SharedProgress | None = None


def _init_worker(progress: _SharedProgress | None) -> None:
    global _WORKER_PROGRESS
    _WORKER_PROGRESS = progress


def _target_reached(local_collected: int, local_target: int, progress: _SharedProgress | None) -> bool:
    """True once enough tweets are in hand — run-wide when a shared counter is in play,
    otherwise for this hashtag alone."""
    if progress is not None:
        return progress.remaining() <= 0
    return local_collected >= local_target


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


def _as_tag(node: Any) -> Tag | None:
    """Narrows a bs4 find() result (Tag | NavigableString | None) down to Tag | None.

    A class/tag-name search should never actually match a bare NavigableString (bs4 only
    returns those for text-content searches), but the stubs type find() as if it could —
    and NavigableString is itself a str subclass, so an un-narrowed result silently
    resolves to str.find()'s signature instead of Tag's. Treated as no match either way.
    """
    return node if isinstance(node, Tag) else None


def extract_tweet_fields(card_html: str, source_hashtag: str) -> dict[str, Any]:
    """Parses one Nitter timeline-item's HTML into username, timestamp, text, engagement,
    mentions, and hashtags. Pure function so it's testable offline against
    tests/fixtures/search_results.html.
    """
    soup = BeautifulSoup(card_html, "html.parser")
    card = _as_tag(soup.find(class_="timeline-item")) or soup

    permalink = _as_tag(card.find("a", class_="tweet-link", href=True))
    if permalink is None:
        permalink = next(
            (a for a in card.find_all("a", href=True) if isinstance(a, Tag) and _STATUS_LINK_RE.match(str(a["href"]))),
            None,
        )
    match = _STATUS_LINK_RE.match(str(permalink["href"])) if permalink else None
    if match is None:
        raise ParseError("no status permalink (username/tweet_id) found in tweet card")
    username, tweet_id = match.group(1), match.group(2)

    date_span = _as_tag(card.find(class_="tweet-date"))
    time_link = _as_tag(date_span.find("a")) if date_span else None
    if time_link is None or not time_link.get("title"):
        raise ParseError(f"no timestamp found for tweet {tweet_id}")
    try:
        created_at = _parse_nitter_timestamp(str(time_link["title"]))
    except ValueError as exc:
        raise ParseError(f"unparseable timestamp for tweet {tweet_id}: {time_link['title']!r}") from exc

    content_node = _as_tag(card.find(class_="tweet-content"))
    if content_node:
        # separator=" " prevents adjacent inline tags (e.g. a mention link followed
        # directly by text) from merging into one token when strip=True trims each
        # fragment individually.
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
            WebDriverWait(driver, pagination_config.page_render_timeout_seconds).until(
                EC.presence_of_element_located((By.CLASS_NAME, "timeline"))
            )
        except TimeoutException as exc:
            # A full-page anti-bot challenge never renders .timeline, so it times out here
            # rather than reaching the is_soft_blocked() check below. Inspect the page before
            # giving up: a recognized challenge page is retryable, a dead host is not.
            if is_soft_blocked(driver.page_source, soft_block_indicators):
                raise RateLimitedError(
                    "soft-block/anti-bot indicator detected after page-render timeout"
                ) from exc
            raise ScrapeTimeoutError("timed out waiting for search results to render") from exc

        # One page_source fetch serves both the soft-block check and card extraction —
        # find_elements() + a per-card get_attribute("outerHTML") would cost a separate
        # WebDriver round-trip per tweet card on the page; parsing the same already-
        # fetched HTML with BeautifulSoup costs nothing extra.
        page_source = driver.page_source
        if is_soft_blocked(page_source, soft_block_indicators):
            raise RateLimitedError("soft-block/anti-bot indicator detected in page source")

        cards_html = [str(card) for card in BeautifulSoup(page_source, "html.parser").find_all(class_="timeline-item")]
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

        created_at = datetime.fromisoformat(record["created_at"])
        if created_at < cutoff:
            # Results are newest-first, so one tweet past the lookback window means
            # everything after it is too — stop paging instead of fruitlessly fetching
            # pages that can't count. Recorded *before* seen_ids so this tweet, which is
            # never written, doesn't inflate the run summary's unique_ids over the row
            # count actually on disk.
            reached_cutoff = True
            break

        seen_ids.add(record["tweet_id"])

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
    progress: _SharedProgress | None = None,
) -> dict[str, Any]:
    """Pages through one host's search results for a hashtag until the target is reached
    (run-wide when `progress` is given, otherwise `remaining_target` for this hashtag),
    the 24h cutoff is hit, or the host proves unhealthy (retries exhausted on repeated
    soft-blocks/stalls).
    """
    url = build_search_url(hashtag, host, settings.scraper.search_path_template)
    rate_limiter = TokenBucketRateLimiter(
        settings.scraper.rate_limiter.bucket_capacity,
        settings.scraper.rate_limiter.refill_rate_per_second,
    )

    # Fresh user-agent per host so one worker doesn't present the same fingerprint to
    # every mirror in the failover list. Best-effort — a driver that can't take the CDP
    # command just keeps the UA it was built with.
    try:
        rotate_user_agent(driver)
    except WebDriverException as exc:
        logger.debug("user-agent rotation failed", extra={"extra_fields": {"host": host, "error": str(exc)}})

    try:
        driver.get(url)
    except (WebDriverException, URLLib3ReadTimeoutError) as exc:
        return {
            "collected": 0,
            "parse_errors": 0,
            "errors": [f"{host}: navigation failed: {exc}"],
            "backoff_triggered": False,
            "host_exhausted": True,
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
                if progress is not None:
                    progress.add(batch_collected)
                if _target_reached(collected, remaining_target, progress) or reached_cutoff:
                    break
            break
        except (RateLimitedError, TimeoutException, URLLib3ReadTimeoutError) as exc:
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
    progress: _SharedProgress | None = None,
) -> dict[str, Any]:
    """Pages through Nitter search results for one hashtag, streaming deduplicated tweets to
    output_path as JSONL, failing over to the next configured host on a soft-block.

    Stops when the target is reached — the run-wide total when `progress` is supplied
    (workers share one target), otherwise this hashtag's own `min_tweets`.

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
            remaining = progress.remaining() if progress is not None else min_tweets - collected
            result = _attempt_host(
                driver, hashtag, host, cutoff, remaining, out_file, seen_ids, settings, progress
            )
            collected += result["collected"]
            parse_errors += result["parse_errors"]
            errors.extend(result["errors"])
            backoff_triggered = backoff_triggered or result["backoff_triggered"]

            if _target_reached(collected, min_tweets, progress):
                break
            # A host can stop short of min_tweets without being "exhausted" (rate-limited)
            # at all — it can just legitimately run out of pages, or hit the lookback
            # cutoff, before yielding enough tweets. Either way, still under target means
            # still worth trying the next configured host rather than stopping here.
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
    scrape_hashtag), and writes only to its own hashtag-specific output file. The one
    piece of shared state is the run-wide collected counter (_WORKER_PROGRESS, set by
    the pool initializer), which is internally locked. driver_path is pre-resolved once
    by the caller (see main()) rather than re-resolved here, since ChromeDriverManager's
    cache isn't safe under several workers touching it at once.
    """
    # Nothing left to do — a worker that only starts after the shared target is already
    # met shouldn't spin up a whole browser to fetch one page and immediately stop.
    if _WORKER_PROGRESS is not None and _WORKER_PROGRESS.remaining() <= 0:
        return {
            "hashtag": hashtag,
            "collected": 0,
            "unique_ids": 0,
            "parse_errors": 0,
            "errors": [],
            "backoff_triggered": False,
            "hosts_tried": [],
        }

    logger.info("starting hashtag scrape", extra={"extra_fields": {"hashtag": hashtag}})
    try:
        with get_driver(headless=settings.scraper.headless, driver_path=driver_path) as driver:
            return scrape_hashtag(driver, hashtag, hours, min_tweets, output_path, settings, _WORKER_PROGRESS)
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
    settings = load_settings()
    set_level(logger, settings.logging.level)

    parser = argparse.ArgumentParser(description="Scrape Indian market-hashtag tweets via Nitter/Selenium.")
    parser.add_argument(
        "--hashtags",
        type=str,
        default=",".join(settings.scraper.all_hashtags),
        help="Comma-separated hashtags, no '#'. Defaults to the assignment's four plus config related_hashtags.",
    )
    parser.add_argument("--hours", type=int, default=settings.scraper.hours_lookback, help="Lookback window in hours.")
    parser.add_argument(
        "--min-tweets", type=int, default=settings.scraper.min_tweets_target, help="Minimum total tweets to collect."
    )
    parser.add_argument("--workers", type=int, default=None, help="Worker pool size (default: config value).")
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="Exit 0 even if fewer than --min-tweets were collected (default: exit non-zero on a shortfall).",
    )
    args = parser.parse_args()

    hashtags = [tag.strip().lstrip("#") for tag in args.hashtags.split(",") if tag.strip()]

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = Path(settings.storage.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    worker_count = max(1, min(args.workers or settings.scraper.worker_pool_size, len(hashtags)))

    # Resolved once, here, before any worker starts — see get_driver()'s docstring for
    # why concurrent workers each resolving their own driver path is unsafe.
    driver_path = resolve_driver_path()

    # One shared target for the whole run rather than args.min_tweets // len(hashtags)
    # per worker: a hashtag with plenty of supply keeps collecting to cover one that
    # runs dry, instead of both stopping at their fixed slice. Each worker is still
    # capped at args.min_tweets on its own so a bug in the shared counter can't make a
    # single hashtag run away.
    summaries: list[dict[str, Any]] = []
    with multiprocessing.Manager() as manager:
        progress = _SharedProgress(manager.Value("i", 0), manager.Lock(), args.min_tweets)
        with ProcessPoolExecutor(
            max_workers=worker_count, initializer=_init_worker, initargs=(progress,)
        ) as executor:
            futures = {
                executor.submit(
                    _scrape_one_hashtag,
                    hashtag,
                    args.hours,
                    args.min_tweets,
                    raw_dir / f"{hashtag}_{run_timestamp}.jsonl",
                    settings,
                    driver_path,
                ): hashtag
                for hashtag in hashtags
            }
            for future in as_completed(futures):
                hashtag = futures[future]
                try:
                    summaries.append(future.result())
                except Exception as exc:
                    # One worker crashing (a bug, a killed process, a broken pool) must
                    # not discard the summaries for the hashtags that succeeded — record
                    # it and carry on so the run summary and the shortfall check still run.
                    logger.exception(
                        "hashtag worker crashed", extra={"extra_fields": {"hashtag": hashtag, "error": str(exc)}}
                    )
                    summaries.append(
                        {
                            "hashtag": hashtag,
                            "collected": 0,
                            "unique_ids": 0,
                            "parse_errors": 0,
                            "errors": [f"worker crashed: {exc}"],
                            "backoff_triggered": False,
                            "hosts_tried": [],
                        }
                    )

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

    # The 2,000-tweet minimum is a hard requirement, not a best-effort goal: a run that
    # falls short must fail loudly so the pipeline stops here instead of carrying a
    # thin corpus through to signals that can't support the statistic. --allow-shortfall
    # opts out for exploratory runs.
    if total_collected < args.min_tweets and not args.allow_shortfall:
        shortfall = args.min_tweets - total_collected
        logger.error(
            "collected fewer tweets than required",
            extra={
                "extra_fields": {
                    "total_collected": total_collected,
                    "min_tweets": args.min_tweets,
                    "shortfall": shortfall,
                    "summary_path": str(summary_path),
                }
            },
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
