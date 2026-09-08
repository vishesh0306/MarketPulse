"""Nitter RSS collection for the tail loop.

Nitter serves every search as an RSS feed (`/search/rss?...`): one HTTP GET, XML body, no
browser and no JavaScript anti-bot challenge in the common case. That makes it one to two
orders of magnitude cheaper per tweet than driving Selenium, which is what a 20-second
polling loop needs. The trade-off is that the feed carries no engagement counts and only
the most recent ~20 items with no cursor — fine for tailing, which is why Selenium stays
the tool for the deep 24h backfill.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import httpx

from src.utils.logger import get_logger

logger = get_logger("realtime_rss")

_STATUS_RE = re.compile(r"/([A-Za-z0-9_]{1,15})/status/(\d+)")
_MENTION_RE = re.compile(r"@(\w+)")
_HASHTAG_RE = re.compile(r"#(\w+)")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def build_rss_url(hashtag: str, host: str, path_template: str) -> str:
    return path_template.format(host=host, hashtag=hashtag)


def _text(node: ElementTree.Element | None) -> str:
    return (node.text or "").strip() if node is not None else ""


def _strip_markup(value: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", value)).strip()


def parse_rss(xml_body: str, source_hashtag: str) -> list[dict[str, object]]:
    """Parses a Nitter RSS body into tweet records shaped like the scraper's JSONL rows.

    Engagement counts aren't in the feed, so likes/retweets/replies come back as 0 — for a
    tweet seconds old that's also the true value, and a later enrichment pass can revise it.
    Malformed items are skipped, not raised on: a partial feed is still useful.
    """
    try:
        root = ElementTree.fromstring(xml_body)
    except ElementTree.ParseError as exc:
        logger.warning("rss parse failed", extra={"extra_fields": {"hashtag": source_hashtag, "error": str(exc)}})
        return []

    records: list[dict[str, object]] = []
    for item in root.iter("item"):
        link = _text(item.find("link"))
        match = _STATUS_RE.search(link)
        if match is None:
            continue
        username, tweet_id = match.group(1), match.group(2)

        raw_title = _text(item.find("title"))
        description = _strip_markup(_text(item.find("description")))
        text = _strip_markup(raw_title) or description
        if not text:
            continue

        pub = _text(item.find("pubDate"))
        try:
            created_at = parsedate_to_datetime(pub).astimezone(timezone.utc) if pub else datetime.now(timezone.utc)
        except (TypeError, ValueError):
            created_at = datetime.now(timezone.utc)

        records.append(
            {
                "tweet_id": tweet_id,
                "username": username,
                "created_at": created_at.isoformat(),
                "text": text,
                "text_normalized": text.lower(),
                "likes": 0,
                "retweets": 0,
                "replies": 0,
                "mentions": sorted(set(_MENTION_RE.findall(text))),
                "hashtags": sorted({tag.lower() for tag in _HASHTAG_RE.findall(text)}),
                "source_hashtag": source_hashtag,
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return records


async def fetch_hashtag_feed(
    client: httpx.AsyncClient,
    hashtag: str,
    hosts: list[str],
    path_template: str,
    *,
    max_items: int,
) -> tuple[list[dict[str, object]], str | None]:
    """Fetches one hashtag's RSS feed, trying each host until one returns parseable items.

    Returns (records newest-first, host that served them) or ([], None) if every host
    failed. Never raises — a poll that comes back empty just means try again next tick.
    """
    for host in hosts:
        url = build_rss_url(hashtag, host, path_template)
        try:
            response = await client.get(url, headers={"Accept": "application/rss+xml, application/xml"})
        except httpx.HTTPError as exc:
            logger.info("rss host unreachable", extra={"extra_fields": {"hashtag": hashtag, "host": host, "error": str(exc)}})
            continue
        if response.status_code != 200 or "xml" not in response.headers.get("content-type", ""):
            logger.info(
                "rss host returned non-feed",
                extra={"extra_fields": {"hashtag": hashtag, "host": host, "status": response.status_code}},
            )
            continue
        records = parse_rss(response.text, hashtag)
        if records:
            return records[:max_items], host
    return [], None
