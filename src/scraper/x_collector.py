"""Collection straight from x.com's own web endpoints, via twscrape.

Why this exists alongside the Nitter/Selenium scraper: Nitter stopped being a usable
source. X removed anonymous access in 2024, so every public mirror now needs real X
account tokens, X bans those tokens, and the instances answer 403/410 to everyone. A run
against the seven configured mirrors currently collects zero tweets.

twscrape authenticates as an ordinary logged-in browser session (the `auth_token`/`ct0`
cookies you can copy out of DevTools) and calls the same JSON endpoints x.com's own
front-end calls. That is scraping a logged-in session — not the paid/developer API, no
API key, nothing purchased. It also means no HTML parsing, so none of the usual
brittleness of scraping a virtualised React timeline applies, and engagement counts come
through as real numbers rather than the zeros an RSS feed gives.

Output is the same JSONL shape the Selenium scraper writes, so processing, dedup, signal
generation, plots and the real-time warm start all consume it unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

from src.processing.cleaner import clean_text, normalize_for_features
from src.processing.deduplicator import content_hash
from src.utils.config_loader import Settings, load_settings
from src.utils.logger import get_logger, set_level, write_run_summary

logger = get_logger("x_collector")

# x.com caps how deep a single search query will paginate, so the collector walks each
# hashtag and stops on the run-wide target rather than trying to drain one query.
_DEFAULT_PER_HASHTAG_CAP = 2000

# How often to emit a progress line while collecting a hashtag.
_PROGRESS_EVERY = 100


def route_twscrape_logs() -> None:
    """Forwards twscrape's loguru output into this project's structured logger.

    twscrape logs the thing you most need to see during a long run — "No account
    available for queue SearchTimeline. Next available at 23:20:56" — but to its own
    loguru sink, in a different format, and nothing else prints while it waits. Routing it
    here means one stream and one format shows both our progress and why a pause is
    happening, so a rate-limit wait is distinguishable from a hang.
    """
    try:
        from loguru import logger as loguru_logger
    except ImportError:  # twscrape absent (pure-function use); nothing to route
        return

    def _sink(message: Any) -> None:
        record = message.record
        logger.log(
            logging.getLevelName(record["level"].name),
            record["message"],
            extra={"extra_fields": {"source": "twscrape"}},
        )

    loguru_logger.remove()
    loguru_logger.add(_sink, level="INFO")


class MissingCredentialsError(RuntimeError):
    """Raised when no X session cookies are available to authenticate with."""


def window_bounds(
    hours: float, offset_hours: float = 0.0, *, now: datetime | None = None
) -> tuple[datetime, datetime]:
    """The [start, end] the collector should keep, as a window ending `offset_hours` ago
    and spanning `hours` back from there.

    offset_hours=0 gives the plain "last N hours". A non-zero offset targets an earlier
    slice — e.g. the NSE cash session, which is a fixed band of the day rather than the
    most recent N hours. Paging blindly from "now" spends the rate-limit budget on
    post-close chatter before it ever reaches the session.
    """
    now = now or datetime.now(timezone.utc)
    end = now - timedelta(hours=offset_hours)
    return end - timedelta(hours=hours), end


def build_query(hashtag: str, hours: float, offset_hours: float = 0.0, *, now: datetime | None = None) -> str:
    """Builds an x.com search query for one hashtag over the requested window.

    Uses `since_time`/`until_time` (unix seconds) rather than `since:`/`until:` — the date
    form has day granularity, which can't express a rolling sub-day window.

    Note these operators are a *hint*, not a guarantee: x.com still slips pinned and
    popular older tweets into results (an observed run came back with a 2023 tweet). The
    window is therefore enforced again client-side in `within_window`, which is what
    actually keeps the output inside the range the assignment asks for.
    """
    start, end = window_bounds(hours, offset_hours, now=now)
    return f"#{hashtag} since_time:{int(start.timestamp())} until_time:{int(end.timestamp())}"


def within_window(
    created_at_iso: str, hours: float, offset_hours: float = 0.0, *, now: datetime | None = None
) -> bool:
    """True if an ISO timestamp falls inside the requested window. The authoritative
    check — see build_query for why x.com's own operators can't be trusted alone."""
    now = now or datetime.now(timezone.utc)
    try:
        created = datetime.fromisoformat(created_at_iso)
    except ValueError:
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    start, end = window_bounds(hours, offset_hours, now=now)
    # Small forward tolerance absorbs clock skew between x.com and this machine.
    return start <= created <= (end + timedelta(minutes=5))


def tweet_to_record(tweet: Any, source_hashtag: str, *, collected_at: datetime | None = None) -> dict[str, Any]:
    """Maps a twscrape Tweet onto the raw-JSONL record the rest of the pipeline expects.

    A pure function of the tweet object, so it's testable without a network call or an
    account — see tests/test_x_collector.py.
    """
    collected_at = collected_at or datetime.now(timezone.utc)
    created = tweet.date
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)

    mentions = sorted({str(u.username) for u in (tweet.mentionedUsers or [])})
    hashtags = sorted({str(tag).lower().lstrip("#") for tag in (tweet.hashtags or [])})

    return {
        "tweet_id": str(tweet.id_str or tweet.id),
        "username": str(tweet.user.username),
        "created_at": created.astimezone(timezone.utc).isoformat(),
        "text": tweet.rawContent or "",
        "likes": int(tweet.likeCount or 0),
        "retweets": int(tweet.retweetCount or 0),
        "replies": int(tweet.replyCount or 0),
        "mentions": mentions,
        "hashtags": hashtags,
        "source_hashtag": source_hashtag,
        "collected_at": collected_at.isoformat(),
    }


def _near_duplicate_key(record: dict[str, Any]) -> str:
    """The processing stage's near-duplicate hash, computed here so the collector's target
    counts tweets that will actually survive into the dataset.

    Must stay in step with `processing.near_duplicate_hash_fields` — text_normalized +
    username + source_hashtag. The collector only has raw text, so it applies the same
    cleaner the processing stage would before hashing. Without this the run stopped at N
    distinct ids and processing then dropped a slice of them as near-duplicates, so a
    2,000-tweet target delivered 1,963.
    """
    normalized = normalize_for_features(clean_text(str(record.get("text", ""))))
    return content_hash([normalized, str(record.get("username", "")), str(record.get("source_hashtag", ""))])


def _load_dotenv_once() -> None:
    """Best-effort load of a local .env so the collector runs without exporting vars by
    hand. python-dotenv is optional; missing it just means you export them yourself."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def resolve_cookies(explicit: str | None = None) -> str:
    """Finds X session cookies from the CLI flag, X_COOKIES, or X_AUTH_TOKEN + X_CT0.

    Never logged and never written to disk by this function — these are a live credential
    for the account they came from.
    """
    if explicit:
        return explicit
    _load_dotenv_once()
    combined = os.environ.get("X_COOKIES")
    if combined:
        return combined
    auth_token, ct0 = os.environ.get("X_AUTH_TOKEN"), os.environ.get("X_CT0")
    if auth_token and ct0:
        return f"auth_token={auth_token}; ct0={ct0}"
    raise MissingCredentialsError(
        "No X session cookies found. Set X_AUTH_TOKEN and X_CT0 (or X_COOKIES) in your "
        "environment/.env, or pass --cookies. Copy them from a logged-in x.com session: "
        "DevTools -> Application -> Cookies -> https://x.com."
    )


async def build_api(cookies: str, db_path: str, account_label: str) -> Any:
    """Creates a twscrape API bound to a single cookie-authenticated account.

    twscrape is imported lazily so this module's pure functions stay importable — and
    testable — without it installed.
    """
    from twscrape import API

    # After the import, not before: twscrape installs its own loguru handler at import
    # time, which would otherwise replace ours again.
    route_twscrape_logs()

    api = API(db_path)
    await api.pool.add_account_cookies(account_label, cookies)
    return api


def _search_stream(
    api: Any, hashtag: str, hours: float, offset_hours: float, limit: int, *, now: datetime | None = None
) -> AsyncGenerator[Any, None]:
    """The raw twscrape search generator for one hashtag.

    Deliberately *not* an `async def` wrapper that re-yields: the collector stops early
    once the run-wide target is met, and breaking out of a generator that re-yields from
    another one closes the inner generator while it is still awaiting a page, which raises
    `RuntimeError: aclose(): asynchronous generator is already running` and leaves pending
    tasks behind. Returning the underlying generator directly means there is exactly one
    object to close, and `aclosing` in the caller closes it deterministically.
    """
    # product=Latest forces the chronological tab; the "Top" tab reorders by engagement
    # and pulls in older popular tweets, which is the opposite of what a 24h window wants.
    return api.search(
        build_query(hashtag, hours, offset_hours, now=now), limit=limit, kv={"product": "Latest"}
    )


async def collect_hashtag(
    api: Any,
    hashtag: str,
    hours: float,
    output_path: Path,
    *,
    remaining: Callable[[], int],
    offset_hours: float = 0.0,
    per_hashtag_cap: int = _DEFAULT_PER_HASHTAG_CAP,
    run_unique_ids: set[str] | None = None,
    run_content_hashes: set[str] | None = None,
    window_end: datetime | None = None,
) -> dict[str, Any]:
    """Streams one hashtag's tweets to JSONL, stopping when the run-wide target is met.

    `run_unique_ids` is the run's set of *distinct* tweet ids, shared across hashtags and
    updated live, and `remaining` is read from it. That distinction matters: a tweet
    carrying two tracked hashtags is legitimately collected under each (storage partitions
    by hashtag, and each hashtag's signal should count it), so rows exceed distinct
    tweets — an observed run wrote 2,000 rows covering only 1,559 tweets. Gating on rows
    would let `--min-tweets 2000` pass on a corpus well short of 2,000 tweets.

    `window_end` pins the collection window for the whole run. Left to default, both the
    query and the filter would call `datetime.now()` afresh, so over a 30-45 minute run
    the cutoff slides forward and tweets accepted early get rejected later — worst at the
    oldest edge, which the Latest tab reaches last.
    """
    collected = 0
    out_of_window = 0
    near_duplicates = 0
    seen_ids: set[str] = set()
    run_unique_ids = run_unique_ids if run_unique_ids is not None else set()
    run_content_hashes = run_content_hashes if run_content_hashes is not None else set()
    window_end = window_end or datetime.now(timezone.utc)
    errors: list[str] = []
    oldest: str | None = None
    newest: str | None = None

    logger.info(
        "hashtag collection started",
        extra={"extra_fields": {"hashtag": hashtag, "still_needed": remaining(), "output": str(output_path)}},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = _search_stream(api, hashtag, hours, offset_hours, per_hashtag_cap, now=window_end)
        # aclosing() so an early break shuts the search generator down deterministically
        # rather than leaving the event loop to finalize it mid-await at interpreter exit.
        with output_path.open("a", encoding="utf-8") as out_file:
            async with contextlib.aclosing(stream) as tweets:
                async for tweet in tweets:
                    if remaining() <= 0:
                        break
                    record = tweet_to_record(tweet, hashtag)
                    if record["tweet_id"] in seen_ids:
                        continue
                    seen_ids.add(record["tweet_id"])
                    if not within_window(str(record["created_at"]), hours, offset_hours, now=window_end):
                        out_of_window += 1
                        continue
                    # The same near-duplicate rule processing applies, so the target counts
                    # tweets that will actually survive into the dataset. Without it the run
                    # stopped at N distinct ids and processing then removed a slice of them,
                    # delivering 1,963 against a target of 2,000.
                    digest = _near_duplicate_key(record)
                    if digest in run_content_hashes:
                        near_duplicates += 1
                        continue
                    run_content_hashes.add(digest)
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_file.flush()
                    collected += 1
                    # Registered after the write, so the run-wide gate counts distinct
                    # tweets actually on disk rather than rows.
                    run_unique_ids.add(str(record["tweet_id"]))
                    created = str(record["created_at"])
                    oldest = created if oldest is None or created < oldest else oldest
                    newest = created if newest is None or created > newest else newest
                    # Without this the run can sit silent for 15 minutes at a stretch
                    # while it pages and waits out rate limits, with no way to tell a
                    # working run from a hung one.
                    if collected % _PROGRESS_EVERY == 0:
                        logger.info(
                            "collecting",
                            extra={
                                "extra_fields": {
                                    "hashtag": hashtag,
                                    "collected": collected,
                                    "run_unique": len(run_unique_ids),
                                    "still_needed": remaining(),
                                    "newest": newest,
                                    "oldest": oldest,
                                }
                            },
                        )
    except Exception as exc:  # noqa: BLE001 — one hashtag failing must not sink the run
        logger.exception("hashtag collection failed", extra={"extra_fields": {"hashtag": hashtag}})
        errors.append(f"{type(exc).__name__}: {exc}")

    logger.info(
        "hashtag collection complete",
        extra={
            "extra_fields": {
                "hashtag": hashtag,
                "collected": collected,
                "dropped_out_of_window": out_of_window,
                "dropped_near_duplicate": near_duplicates,
                "oldest": oldest,
                "newest": newest,
            }
        },
    )
    return {
        "hashtag": hashtag,
        "collected": collected,
        "unique_ids": len(seen_ids),
        "dropped_out_of_window": out_of_window,
        "dropped_near_duplicate": near_duplicates,
        "parse_errors": 0,
        "errors": errors,
        "oldest_tweet": oldest,
        "newest_tweet": newest,
        "backoff_triggered": False,
        "hosts_tried": ["x.com"],
    }


def _empty_summary(hashtag: str) -> dict[str, Any]:
    return {
        "hashtag": hashtag,
        "collected": 0,
        "unique_ids": 0,
        "dropped_out_of_window": 0,
        "dropped_near_duplicate": 0,
        "parse_errors": 0,
        "errors": [],
        "oldest_tweet": None,
        "newest_tweet": None,
        "backoff_triggered": False,
        "hosts_tried": [],
    }


async def collect(
    hashtags: list[str],
    hours: float,
    min_tweets: int,
    raw_dir: Path,
    run_timestamp: str,
    *,
    api: Any,
    offset_hours: float = 0.0,
) -> tuple[list[dict[str, Any]], int]:
    """Runs every hashtag against x.com until the run-wide target is met.

    The target counts *distinct tweets*, not rows. A tweet carrying two tracked hashtags
    is written once per hashtag, so rows run well ahead of tweets — gating on rows would
    let `--min-tweets 2000` succeed on a corpus of far fewer than 2,000 tweets.

    The window is pinned once here so every hashtag queries and filters against the same
    instant, rather than each re-deriving "now" as the run stretches over rate-limit waits.
    """
    run_unique_ids: set[str] = set()
    run_content_hashes: set[str] = set()
    window_end = datetime.now(timezone.utc)
    rows = 0

    def remaining() -> int:
        return max(0, min_tweets - len(run_unique_ids))

    summaries: list[dict[str, Any]] = []
    for hashtag in hashtags:
        if remaining() <= 0:
            summaries.append(_empty_summary(hashtag))
            continue
        summary = await collect_hashtag(
            api,
            hashtag,
            hours,
            raw_dir / f"{hashtag}_{run_timestamp}.jsonl",
            remaining=remaining,
            offset_hours=offset_hours,
            run_unique_ids=run_unique_ids,
            run_content_hashes=run_content_hashes,
            window_end=window_end,
        )
        rows += int(summary["collected"])
        summaries.append(summary)
        logger.info(
            "run progress",
            extra={
                "extra_fields": {
                    "unique_tweets": len(run_unique_ids),
                    "rows_written": rows,
                    "target": min_tweets,
                }
            },
        )
    return summaries, len(run_unique_ids)


async def _run(
    args: argparse.Namespace, settings: Settings, hashtags: list[str], run_timestamp: str
) -> tuple[list[dict[str, Any]], int]:
    cookies = resolve_cookies(args.cookies)
    api = await build_api(cookies, args.accounts_db, args.account_label)
    return await collect(
        hashtags,
        args.hours,
        args.min_tweets,
        Path(settings.storage.raw_dir),
        run_timestamp,
        api=api,
        offset_hours=args.offset_hours,
    )


def main() -> None:
    settings = load_settings()
    set_level(logger, settings.logging.level)

    parser = argparse.ArgumentParser(description="Collect Indian market tweets from x.com via twscrape.")
    parser.add_argument("--hashtags", type=str, default=",".join(settings.scraper.all_hashtags))
    parser.add_argument("--hours", type=float, default=settings.scraper.hours_lookback)
    parser.add_argument(
        "--offset-hours",
        type=float,
        default=0.0,
        help="End the window this many hours before now (0 = last --hours). Lets a run "
        "target an earlier slice such as the NSE session instead of paging from now.",
    )
    parser.add_argument("--min-tweets", type=int, default=settings.scraper.min_tweets_target)
    parser.add_argument("--cookies", type=str, default=None, help="auth_token=...; ct0=... (else read from env)")
    parser.add_argument("--accounts-db", type=str, default="accounts.db", help="twscrape account store path")
    parser.add_argument("--account-label", type=str, default="marketpulse")
    parser.add_argument(
        "--allow-shortfall", action="store_true", help="Exit 0 even if fewer than --min-tweets were collected."
    )
    args = parser.parse_args()

    hashtags = [tag.strip().lstrip("#") for tag in args.hashtags.split(",") if tag.strip()]
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    Path(settings.storage.raw_dir).mkdir(parents=True, exist_ok=True)

    try:
        summaries, unique_tweets = asyncio.run(_run(args, settings, hashtags, run_timestamp))
    except MissingCredentialsError as exc:
        logger.error("missing credentials", extra={"extra_fields": {"error": str(exc)}})
        sys.exit(2)

    rows_written = sum(int(s["collected"]) for s in summaries)
    summary_path = write_run_summary(
        settings.logging.dir,
        "scraper",
        {
            "run_timestamp": run_timestamp,
            "source": "x.com (twscrape)",
            "hashtags": hashtags,
            "hours_lookback": args.hours,
            "offset_hours": args.offset_hours,
            "min_tweets_target": args.min_tweets,
            # The target counts distinct tweets, so that is what gets reported and
            # checked. rows_written runs ahead of it because a tweet carrying two
            # tracked hashtags is written once under each.
            "total_collected": unique_tweets,
            "unique_tweets": unique_tweets,
            "rows_written": rows_written,
            "per_hashtag": summaries,
        },
    )
    logger.info(
        "collection run complete",
        extra={
            "extra_fields": {
                "unique_tweets": unique_tweets,
                "rows_written": rows_written,
                "summary_path": str(summary_path),
            }
        },
    )

    if unique_tweets < args.min_tweets and not args.allow_shortfall:
        logger.error(
            "collected fewer tweets than required",
            extra={
                "extra_fields": {
                    "unique_tweets": unique_tweets,
                    "rows_written": rows_written,
                    "min_tweets": args.min_tweets,
                    "shortfall": args.min_tweets - unique_tweets,
                }
            },
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
