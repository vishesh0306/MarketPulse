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
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from src.utils.config_loader import Settings, load_settings
from src.utils.logger import get_logger, set_level, write_run_summary

logger = get_logger("x_collector")

# x.com caps how deep a single search query will paginate, so the collector walks each
# hashtag and stops on the run-wide target rather than trying to drain one query.
_DEFAULT_PER_HASHTAG_CAP = 2000


class MissingCredentialsError(RuntimeError):
    """Raised when no X session cookies are available to authenticate with."""


def build_query(hashtag: str, hours: int, *, now: datetime | None = None) -> str:
    """Builds an x.com search query for one hashtag over the last `hours`.

    Uses `since_time`/`until_time` (unix seconds) rather than `since:`/`until:` — the date
    form has day granularity, which can't express a 24-hour rolling window.

    Note these operators are a *hint*, not a guarantee: x.com still slips pinned and
    popular older tweets into results (an observed run came back with a 2023 tweet). The
    lookback is therefore enforced again client-side in `within_window`, which is what
    actually keeps the output inside the 24 hours the assignment asks for.
    """
    now = now or datetime.now(timezone.utc)
    since = int((now - timedelta(hours=hours)).timestamp())
    until = int(now.timestamp())
    return f"#{hashtag} since_time:{since} until_time:{until}"


def within_window(created_at_iso: str, hours: int, *, now: datetime | None = None) -> bool:
    """True if an ISO timestamp falls inside the last `hours`. The authoritative lookback
    check — see build_query for why x.com's own operators can't be trusted alone."""
    now = now or datetime.now(timezone.utc)
    try:
        created = datetime.fromisoformat(created_at_iso)
    except ValueError:
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (now - timedelta(hours=hours)) <= created <= (now + timedelta(minutes=5))


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

    api = API(db_path)
    await api.pool.add_account_cookies(account_label, cookies)
    return api


async def _iter_hashtag(api: Any, hashtag: str, hours: int, limit: int) -> AsyncIterator[Any]:
    # product=Latest forces the chronological tab; the "Top" tab reorders by engagement
    # and pulls in older popular tweets, which is the opposite of what a 24h window wants.
    async for tweet in api.search(build_query(hashtag, hours), limit=limit, kv={"product": "Latest"}):
        yield tweet


async def collect_hashtag(
    api: Any,
    hashtag: str,
    hours: int,
    output_path: Path,
    *,
    remaining: Callable[[], int],
    per_hashtag_cap: int = _DEFAULT_PER_HASHTAG_CAP,
) -> dict[str, Any]:
    """Streams one hashtag's tweets to JSONL, stopping when the run-wide target is met.

    `remaining` returns how many more tweets the run still needs *excluding* what this
    call has collected so far — the caller only folds a hashtag's total in once it
    finishes — so the loop compares its own running count against it. Same shared-target
    idea as the Selenium scraper's _SharedProgress: a dense hashtag covers for a sparse
    one instead of both stopping at a fixed slice.
    """
    collected = 0
    out_of_window = 0
    seen_ids: set[str] = set()
    errors: list[str] = []
    oldest: str | None = None
    newest: str | None = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("a", encoding="utf-8") as out_file:
            async for tweet in _iter_hashtag(api, hashtag, hours, per_hashtag_cap):
                if collected >= remaining():
                    break
                record = tweet_to_record(tweet, hashtag)
                if record["tweet_id"] in seen_ids:
                    continue
                seen_ids.add(record["tweet_id"])
                if not within_window(str(record["created_at"]), hours):
                    out_of_window += 1
                    continue
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()
                collected += 1
                created = str(record["created_at"])
                oldest = created if oldest is None or created < oldest else oldest
                newest = created if newest is None or created > newest else newest
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
        "parse_errors": 0,
        "errors": [],
        "oldest_tweet": None,
        "newest_tweet": None,
        "backoff_triggered": False,
        "hosts_tried": [],
    }


async def collect(
    hashtags: list[str],
    hours: int,
    min_tweets: int,
    raw_dir: Path,
    run_timestamp: str,
    *,
    api: Any,
) -> list[dict[str, Any]]:
    """Runs every hashtag against x.com until the run-wide target is met."""
    total = 0

    def remaining() -> int:
        return max(0, min_tweets - total)

    summaries: list[dict[str, Any]] = []
    for hashtag in hashtags:
        if remaining() <= 0:
            summaries.append(_empty_summary(hashtag))
            continue
        summary = await collect_hashtag(
            api, hashtag, hours, raw_dir / f"{hashtag}_{run_timestamp}.jsonl", remaining=remaining
        )
        total += int(summary["collected"])
        summaries.append(summary)
        logger.info("run progress", extra={"extra_fields": {"total_collected": total, "target": min_tweets}})
    return summaries


async def _run(args: argparse.Namespace, settings: Settings, hashtags: list[str], run_timestamp: str) -> list[dict[str, Any]]:
    cookies = resolve_cookies(args.cookies)
    api = await build_api(cookies, args.accounts_db, args.account_label)
    return await collect(
        hashtags,
        args.hours,
        args.min_tweets,
        Path(settings.storage.raw_dir),
        run_timestamp,
        api=api,
    )


def main() -> None:
    settings = load_settings()
    set_level(logger, settings.logging.level)

    parser = argparse.ArgumentParser(description="Collect Indian market tweets from x.com via twscrape.")
    parser.add_argument("--hashtags", type=str, default=",".join(settings.scraper.all_hashtags))
    parser.add_argument("--hours", type=int, default=settings.scraper.hours_lookback)
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
        summaries = asyncio.run(_run(args, settings, hashtags, run_timestamp))
    except MissingCredentialsError as exc:
        logger.error("missing credentials", extra={"extra_fields": {"error": str(exc)}})
        sys.exit(2)

    total_collected = sum(int(s["collected"]) for s in summaries)
    summary_path = write_run_summary(
        settings.logging.dir,
        "scraper",
        {
            "run_timestamp": run_timestamp,
            "source": "x.com (twscrape)",
            "hashtags": hashtags,
            "hours_lookback": args.hours,
            "min_tweets_target": args.min_tweets,
            "total_collected": total_collected,
            "per_hashtag": summaries,
        },
    )
    logger.info(
        "collection run complete",
        extra={"extra_fields": {"total_collected": total_collected, "summary_path": str(summary_path)}},
    )

    if total_collected < args.min_tweets and not args.allow_shortfall:
        logger.error(
            "collected fewer tweets than required",
            extra={
                "extra_fields": {
                    "total_collected": total_collected,
                    "min_tweets": args.min_tweets,
                    "shortfall": args.min_tweets - total_collected,
                }
            },
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
