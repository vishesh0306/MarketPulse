"""x.com/twscrape collector: query construction, record mapping, shared-target collection.

No network and no twscrape account needed — the API is faked, and tweet_to_record is a
pure function of a tweet-shaped object.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.processing.schema import validate
from src.scraper.x_collector import (
    MissingCredentialsError,
    build_query,
    collect,
    collect_hashtag,
    resolve_cookies,
    tweet_to_record,
    within_window,
)

_NOW = datetime(2026, 9, 8, 12, 0, 0, tzinfo=timezone.utc)


def _tweet(
    tweet_id: int = 1234567890,
    *,
    minutes_ago: int = 5,
    text: str = "Nifty breakout above 24000 @rbi #nifty50 #BankNifty",
    likes: int = 12,
    retweets: int = 3,
    replies: int = 4,
    username: str = "trader_raj",
    hashtags: list[str] | None = None,
    mentions: list[str] | None = None,
    naive_date: bool = False,
) -> SimpleNamespace:
    date = _NOW - timedelta(minutes=minutes_ago)
    return SimpleNamespace(
        id=tweet_id,
        id_str=str(tweet_id),
        date=date.replace(tzinfo=None) if naive_date else date,
        user=SimpleNamespace(username=username),
        rawContent=text,
        likeCount=likes,
        retweetCount=retweets,
        replyCount=replies,
        hashtags=["nifty50", "BankNifty"] if hashtags is None else hashtags,
        mentionedUsers=[SimpleNamespace(username=m) for m in (["rbi"] if mentions is None else mentions)],
    )


class _FakeAPI:
    """Stands in for twscrape's API: yields a fixed list of tweets per search call.

    Mirrors the real signature (including `kv`) so a change to how we call search shows
    up here rather than only in production."""

    def __init__(self, tweets: list[SimpleNamespace]) -> None:
        self._tweets = tweets
        self.queries: list[str] = []
        self.kvs: list[dict | None] = []

    async def search(self, query: str, limit: int = 0, kv: dict | None = None):  # noqa: ARG002
        self.queries.append(query)
        self.kvs.append(kv)
        for tweet in self._tweets:
            yield tweet


# ---- build_query ----------------------------------------------------------


def test_build_query_uses_unix_window_not_day_granularity() -> None:
    query = build_query("nifty50", 24, now=_NOW)
    since = int((_NOW - timedelta(hours=24)).timestamp())
    until = int(_NOW.timestamp())
    assert query == f"#nifty50 since_time:{since} until_time:{until}"


def test_build_query_window_width_matches_hours() -> None:
    q = build_query("sensex", 6, now=_NOW)
    since = int(q.split("since_time:")[1].split()[0])
    until = int(q.split("until_time:")[1])
    assert until - since == 6 * 3600


def test_build_query_offset_targets_an_earlier_slice() -> None:
    """A window ending `offset_hours` ago — used to hit the NSE session directly instead
    of burning rate-limit budget paging back through post-close chatter."""
    q = build_query("nifty50", 6.25, 8.4, now=_NOW)
    since = int(q.split("since_time:")[1].split()[0])
    until = int(q.split("until_time:")[1])
    assert until == int((_NOW - timedelta(hours=8.4)).timestamp())
    assert until - since == int(6.25 * 3600)


def test_within_window_respects_offset() -> None:
    inside = (_NOW - timedelta(hours=10)).isoformat()      # within the 8.4h-14.65h band
    too_recent = (_NOW - timedelta(hours=2)).isoformat()   # after the window closed
    too_old = (_NOW - timedelta(hours=20)).isoformat()     # before it opened
    assert within_window(inside, 6.25, 8.4, now=_NOW) is True
    assert within_window(too_recent, 6.25, 8.4, now=_NOW) is False
    assert within_window(too_old, 6.25, 8.4, now=_NOW) is False


# ---- within_window --------------------------------------------------------


def test_within_window_accepts_recent_and_rejects_old() -> None:
    recent = (_NOW - timedelta(hours=3)).isoformat()
    stale = (_NOW - timedelta(days=400)).isoformat()  # the 2023 tweet x.com actually returned
    assert within_window(recent, 24, now=_NOW) is True
    assert within_window(stale, 24, now=_NOW) is False


def test_within_window_boundary_and_clock_skew() -> None:
    just_inside = (_NOW - timedelta(hours=23, minutes=59)).isoformat()
    just_outside = (_NOW - timedelta(hours=24, minutes=1)).isoformat()
    slight_future = (_NOW + timedelta(minutes=2)).isoformat()  # tolerated: clock skew
    far_future = (_NOW + timedelta(hours=2)).isoformat()
    assert within_window(just_inside, 24, now=_NOW) is True
    assert within_window(just_outside, 24, now=_NOW) is False
    assert within_window(slight_future, 24, now=_NOW) is True
    assert within_window(far_future, 24, now=_NOW) is False


def test_within_window_rejects_unparseable() -> None:
    assert within_window("not-a-date", 24, now=_NOW) is False


async def test_collect_hashtag_drops_out_of_window_tweets(tmp_path: Path) -> None:
    """x.com leaks pinned/popular older tweets into results even with since_time set, so
    the collector must not write them — the 24h requirement depends on this."""
    fresh = _tweet(1, minutes_ago=30)
    ancient = _tweet(2, minutes_ago=60 * 24 * 400)
    api = _FakeAPI([fresh, ancient])
    out = tmp_path / "n.jsonl"

    summary = await collect_hashtag(api, "nifty50", 24, out, remaining=lambda: 100)

    assert summary["collected"] == 1
    assert summary["dropped_out_of_window"] == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["tweet_id"] for r in rows] == ["1"]


# ---- tweet_to_record ------------------------------------------------------


def test_tweet_to_record_maps_every_required_field() -> None:
    record = tweet_to_record(_tweet(), "nifty50", collected_at=_NOW)
    assert record["tweet_id"] == "1234567890"
    assert record["username"] == "trader_raj"
    assert record["created_at"] == "2026-09-08T11:55:00+00:00"
    assert record["text"].startswith("Nifty breakout")
    assert (record["likes"], record["retweets"], record["replies"]) == (12, 3, 4)
    assert record["mentions"] == ["rbi"]
    assert record["hashtags"] == ["banknifty", "nifty50"]  # lowercased + sorted
    assert record["source_hashtag"] == "nifty50"
    assert record["collected_at"] == _NOW.isoformat()


def test_tweet_to_record_output_passes_the_pipeline_schema() -> None:
    """The whole point of this collector is that downstream stages consume it unchanged,
    so a mapped record plus the two fields the cleaner adds must validate."""
    record = tweet_to_record(_tweet(), "nifty50", collected_at=_NOW)
    record["text_normalized"] = record["text"].lower()
    record["lang_hint"] = "en"
    is_valid, reason = validate(record)
    assert is_valid, reason


def test_tweet_to_record_assumes_utc_for_naive_timestamps() -> None:
    record = tweet_to_record(_tweet(naive_date=True), "nifty50", collected_at=_NOW)
    assert record["created_at"].endswith("+00:00")


def test_tweet_to_record_handles_empty_hashtags_and_mentions() -> None:
    record = tweet_to_record(_tweet(hashtags=[], mentions=[]), "sensex", collected_at=_NOW)
    assert record["hashtags"] == []
    assert record["mentions"] == []


def test_tweet_to_record_strips_leading_hash_and_dedupes() -> None:
    record = tweet_to_record(_tweet(hashtags=["#Nifty50", "nifty50", "SENSEX"]), "nifty50", collected_at=_NOW)
    assert record["hashtags"] == ["nifty50", "sensex"]


# ---- resolve_cookies ------------------------------------------------------


def test_resolve_cookies_prefers_explicit_argument() -> None:
    assert resolve_cookies("auth_token=a; ct0=b") == "auth_token=a; ct0=b"


@pytest.fixture
def no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stops resolve_cookies() reading a developer's real local .env mid-test."""
    monkeypatch.setattr("src.scraper.x_collector._load_dotenv_once", lambda: None)


def test_resolve_cookies_from_split_env_vars(monkeypatch: pytest.MonkeyPatch, no_dotenv: None) -> None:
    monkeypatch.delenv("X_COOKIES", raising=False)
    monkeypatch.setenv("X_AUTH_TOKEN", "tok")
    monkeypatch.setenv("X_CT0", "csrf")
    assert resolve_cookies() == "auth_token=tok; ct0=csrf"


def test_resolve_cookies_raises_with_actionable_message(monkeypatch: pytest.MonkeyPatch, no_dotenv: None) -> None:
    for var in ("X_COOKIES", "X_AUTH_TOKEN", "X_CT0"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(MissingCredentialsError, match="DevTools"):
        resolve_cookies()


# ---- collect_hashtag / collect --------------------------------------------


async def test_collect_hashtag_requests_the_latest_tab(tmp_path: Path) -> None:
    """The Top tab reorders by engagement and drags in old popular tweets — a 24h window
    needs the chronological tab."""
    api = _FakeAPI([_tweet(1)])
    await collect_hashtag(api, "nifty50", 24, tmp_path / "n.jsonl", remaining=lambda: 10)
    assert api.kvs == [{"product": "Latest"}]


async def test_collect_hashtag_writes_jsonl_and_dedupes(tmp_path: Path) -> None:
    tweets = [_tweet(1), _tweet(2), _tweet(1)]  # id 1 repeated
    api = _FakeAPI(tweets)
    out = tmp_path / "nifty50.jsonl"

    summary = await collect_hashtag(api, "nifty50", 24, out, remaining=lambda: 100)

    assert summary["collected"] == 2
    assert summary["unique_ids"] == 2
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["tweet_id"] for r in rows] == ["1", "2"]
    assert all(r["source_hashtag"] == "nifty50" for r in rows)


async def test_collect_hashtag_stops_at_run_wide_remaining(tmp_path: Path) -> None:
    """`remaining` reads the live run-wide set, so it decrements as tweets are written."""
    api = _FakeAPI([_tweet(i) for i in range(1, 11)])
    run_ids: set[str] = set()
    summary = await collect_hashtag(
        api, "nifty50", 24, tmp_path / "n.jsonl",
        remaining=lambda: max(0, 3 - len(run_ids)), run_unique_ids=run_ids,
    )
    assert summary["collected"] == 3
    assert run_ids == {"1", "2", "3"}


async def test_target_counts_distinct_tweets_not_rows(tmp_path: Path) -> None:
    """A tweet carrying two tracked hashtags is written once per hashtag, so rows run
    ahead of distinct tweets. Gating on rows would let --min-tweets pass on a corpus of
    far fewer actual tweets — an observed run wrote 2,000 rows covering only 1,559."""
    shared = [_tweet(1), _tweet(2), _tweet(3)]  # same three tweets under every hashtag
    api = _FakeAPI(shared)

    summaries = await collect(["nifty50", "sensex", "intraday"], 24, 5, tmp_path, "20260908T120000Z", api=api)

    rows = sum(int(s["collected"]) for s in summaries)
    distinct = len({json.loads(line)["tweet_id"]
                    for f in tmp_path.glob("*.jsonl")
                    for line in f.read_text(encoding="utf-8").splitlines()})
    # All three hashtags run, because 3 distinct tweets never reaches the target of 5 —
    # even though 9 rows get written. Under a row-based gate it would have stopped at 5.
    assert distinct == 3
    assert rows == 9
    assert all(int(s["collected"]) == 3 for s in summaries)


async def test_collect_pins_the_window_for_the_whole_run(tmp_path: Path) -> None:
    """Every hashtag must query and filter against the same instant. If each re-derived
    'now', a 30-45 minute run would slide its cutoff forward and start rejecting tweets
    at the oldest edge that it would have accepted earlier."""
    api = _FakeAPI([_tweet(1)])
    await collect(["nifty50", "sensex"], 24, 100, tmp_path, "20260908T120000Z", api=api)

    windows = {q.split("since_time:")[1] for q in api.queries}
    assert len(api.queries) == 2
    assert len(windows) == 1, f"window drifted between hashtags: {windows}"


async def test_collect_hashtag_records_oldest_and_newest(tmp_path: Path) -> None:
    api = _FakeAPI([_tweet(1, minutes_ago=60), _tweet(2, minutes_ago=5)])
    summary = await collect_hashtag(api, "nifty50", 24, tmp_path / "n.jsonl", remaining=lambda: 100)
    assert summary["oldest_tweet"] < summary["newest_tweet"]


async def test_collect_hashtag_survives_a_failing_search(tmp_path: Path) -> None:
    class _Boom:
        async def search(self, query: str, limit: int = 0, kv: dict | None = None):  # noqa: ARG002
            raise RuntimeError("rate limited")
            yield  # pragma: no cover

    summary = await collect_hashtag(_Boom(), "nifty50", 24, tmp_path / "n.jsonl", remaining=lambda: 100)
    assert summary["collected"] == 0
    assert summary["errors"] and "rate limited" in summary["errors"][0]


async def test_collect_shares_the_target_across_hashtags(tmp_path: Path) -> None:
    """A dense hashtag should cover for the rest: once the run-wide target is met, later
    hashtags are skipped rather than each chasing its own slice."""
    api = _FakeAPI([_tweet(i) for i in range(1, 21)])
    summaries = await collect(["nifty50", "sensex", "intraday"], 24, 5, tmp_path, "20260908T120000Z", api=api)

    assert sum(int(s["collected"]) for s in summaries) == 5
    assert summaries[0]["collected"] == 5
    assert summaries[1]["collected"] == 0 and summaries[2]["collected"] == 0
    assert (tmp_path / "nifty50_20260908T120000Z.jsonl").exists()


async def test_collect_spreads_across_hashtags_when_each_is_thin(tmp_path: Path) -> None:
    api = _FakeAPI([_tweet(1), _tweet(2)])
    summaries = await collect(["nifty50", "sensex"], 24, 10, tmp_path, "20260908T120000Z", api=api)
    assert [s["collected"] for s in summaries] == [2, 2]
