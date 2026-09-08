"""Nitter RSS parsing and host failover for the tail loop."""

from __future__ import annotations

from pathlib import Path

import httpx

from src.realtime.rss_client import build_rss_url, fetch_hashtag_feed, parse_rss

_RSS = (Path(__file__).parent / "fixtures" / "nitter_rss.xml").read_text(encoding="utf-8")
_TEMPLATE = "https://{host}/search/rss?f=tweets&q=%23{hashtag}"


def test_build_rss_url() -> None:
    assert build_rss_url("nifty50", "nitter.example.com", _TEMPLATE) == (
        "https://nitter.example.com/search/rss?f=tweets&q=%23nifty50"
    )


def test_parse_rss_extracts_records_and_skips_unparseable_items() -> None:
    records = parse_rss(_RSS, "nifty50")
    # 3 valid items; the 4th has no /status/ link and is skipped.
    assert len(records) == 3
    first = records[0]
    assert first["tweet_id"] == "2097291464778957244"
    assert first["username"] == "trader_raj"
    assert first["source_hashtag"] == "nifty50"
    assert first["created_at"].startswith("2026-09-08T06:15:00")
    assert first["likes"] == first["retweets"] == first["replies"] == 0
    assert first["mentions"] == ["rbi"]
    assert set(first["hashtags"]) == {"nifty50", "banknifty"}
    assert first["text_normalized"] == first["text"].lower()


def test_parse_rss_falls_back_to_description_when_title_empty() -> None:
    records = parse_rss(_RSS, "intraday")
    body_item = [r for r in records if r["tweet_id"] == "2097289476255146298"][0]
    assert "fallback body text" in str(body_item["text"]).lower()


def test_parse_rss_bad_xml_returns_empty() -> None:
    assert parse_rss("<rss><channel><item>", "nifty50") == []


async def test_fetch_hashtag_feed_uses_first_healthy_host() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "down.example" in request.url.host:
            return httpx.Response(503, text="unavailable")
        if "empty.example" in request.url.host:
            return httpx.Response(200, text="<rss><channel></channel></rss>", headers={"content-type": "application/xml"})
        return httpx.Response(200, text=_RSS, headers={"content-type": "application/rss+xml"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        records, host = await fetch_hashtag_feed(
            client, "nifty50", ["down.example", "empty.example", "good.example"], _TEMPLATE, max_items=100
        )

    assert host == "good.example"
    assert len(records) == 3


async def test_fetch_hashtag_feed_all_hosts_fail_returns_empty() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(503))
    async with httpx.AsyncClient(transport=transport) as client:
        records, host = await fetch_hashtag_feed(client, "nifty50", ["a.example", "b.example"], _TEMPLATE, max_items=100)
    assert records == []
    assert host is None


async def test_fetch_hashtag_feed_caps_items() -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, text=_RSS, headers={"content-type": "application/xml"})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        records, _ = await fetch_hashtag_feed(client, "nifty50", ["good.example"], _TEMPLATE, max_items=2)
    assert len(records) == 2
