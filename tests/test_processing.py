"""Tests for src/processing: cleaner, schema, deduplicator, storage."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.processing.cleaner import clean_text, detect_lang_hint, normalize_for_features
from src.processing.deduplicator import content_hash, dedup_records
from src.processing.schema import validate
from src.processing.storage import process_raw_files


# ---- cleaner ----------------------------------------------------------------


def test_clean_text_strips_urls() -> None:
    assert clean_text("Nifty breakout today https://t.co/abc123 great move") == "Nifty breakout today great move"


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("Nifty   is\n\nup   today") == "Nifty is up today"


def test_clean_text_nfc_normalizes_devanagari() -> None:
    # Decomposed (base char + combining vowel sign) vs already-composed form.
    decomposed = "क" + "ा"  # KA + AA vowel sign, byte-distinct from the composed form
    cleaned = clean_text(decomposed)
    assert cleaned == unicodedata_normalize(decomposed)


def unicodedata_normalize(text: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFC", text)


def test_normalize_for_features_lowercases() -> None:
    assert normalize_for_features("Nifty50 BREAKOUT") == "nifty50 breakout"


def test_detect_lang_hint_english() -> None:
    assert detect_lang_hint("Nifty is up today") == "en"


def test_detect_lang_hint_hindi() -> None:
    assert detect_lang_hint("बाजार आज ऊपर है") == "hi"


def test_detect_lang_hint_mixed() -> None:
    assert detect_lang_hint("Market aaj बहुत अच्छा है today") == "mixed"


# ---- schema -------------------------------------------------------------


def _valid_record() -> dict:
    return {
        "tweet_id": "123",
        "username": "trader1",
        "created_at": "2026-08-04T12:00:00+00:00",
        "collected_at": "2026-08-04T12:05:00+00:00",
        "text": "Nifty breakout above resistance",
        "text_normalized": "nifty breakout above resistance",
        "likes": 5,
        "retweets": 2,
        "replies": 1,
        "mentions": [],
        "hashtags": ["nifty50"],
        "source_hashtag": "nifty50",
        "lang_hint": "en",
    }


def test_validate_accepts_well_formed_record() -> None:
    is_valid, reason = validate(_valid_record())
    assert is_valid is True
    assert reason is None


def test_validate_rejects_empty_text_with_reason() -> None:
    record = {**_valid_record(), "text": ""}
    is_valid, reason = validate(record)
    assert is_valid is False
    assert reason is not None and "text" in reason


def test_validate_rejects_missing_tweet_id() -> None:
    record = _valid_record()
    del record["tweet_id"]
    is_valid, reason = validate(record)
    assert is_valid is False
    assert reason is not None and "tweet_id" in reason


def test_validate_rejects_negative_engagement_count() -> None:
    record = {**_valid_record(), "likes": -5}
    is_valid, reason = validate(record)
    assert is_valid is False
    assert reason is not None


# ---- deduplicator ---------------------------------------------------------


def test_content_hash_deterministic() -> None:
    assert content_hash(["a", "b"]) == content_hash(["a", "b"])
    assert content_hash(["a", "b"]) != content_hash(["a", "c"])


def test_dedup_records_removes_exact_tweet_id_duplicate() -> None:
    records = [
        {"tweet_id": "1", "source_hashtag": "nifty50", "text_normalized": "hello", "username": "u1"},
        {"tweet_id": "1", "source_hashtag": "nifty50", "text_normalized": "hello", "username": "u1"},
        {"tweet_id": "2", "source_hashtag": "nifty50", "text_normalized": "world", "username": "u2"},
    ]
    result = list(dedup_records(records, ["text_normalized", "username"]))
    assert [r["tweet_id"] for r in result] == ["1", "2"]


def test_dedup_records_keeps_same_tweet_under_different_source_hashtag() -> None:
    """A tweet mentioning both #nifty50 and #banknifty is collected once per hashtag
    search, same tweet_id, different source_hashtag each time. Keying dedup on tweet_id
    alone would drop the second copy and erase the tweet from one hashtag's partition
    entirely; keying on (tweet_id, source_hashtag) keeps both."""
    records = [
        {"tweet_id": "1", "source_hashtag": "banknifty", "text_normalized": "hello", "username": "u1"},
        {"tweet_id": "1", "source_hashtag": "nifty50", "text_normalized": "hello", "username": "u1"},
    ]
    result = list(dedup_records(records, ["text_normalized", "username", "source_hashtag"]))
    assert [r["source_hashtag"] for r in result] == ["banknifty", "nifty50"]


def test_dedup_records_removes_near_duplicate_content() -> None:
    """Deliberately-inserted duplicate: different tweet_id, identical normalized text
    and author — must be caught by the near-duplicate hash layer, not just exact-ID dedup."""
    records = [
        {"tweet_id": "100", "source_hashtag": "nifty50", "text_normalized": "nifty breakout", "username": "trader1"},
        {"tweet_id": "101", "source_hashtag": "nifty50", "text_normalized": "nifty breakout", "username": "trader1"},
        {"tweet_id": "102", "source_hashtag": "nifty50", "text_normalized": "different content", "username": "trader1"},
    ]
    result = list(dedup_records(records, ["text_normalized", "username"]))
    assert [r["tweet_id"] for r in result] == ["100", "102"]


# ---- storage (integration) ------------------------------------------------


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    records = [
        {
            "tweet_id": "1001",
            "username": "trader1",
            "created_at": "2026-08-04T12:00:00+00:00",
            "text": "Nifty breakout above resistance #nifty50",
            "likes": 5,
            "retweets": 2,
            "replies": 1,
            "mentions": [],
            "hashtags": ["nifty50"],
            "source_hashtag": "nifty50",
            "collected_at": "2026-08-04T12:05:00+00:00",
        },
        {
            # Deliberately malformed: empty text after stripping — must be quarantined,
            # not silently dropped or silently kept, proving the reject path works even
            # though the real scraped dataset happened to produce zero rejects.
            "tweet_id": "1002",
            "username": "trader2",
            "created_at": "2026-08-04T12:01:00+00:00",
            "text": "   ",
            "likes": 1,
            "retweets": 0,
            "replies": 0,
            "mentions": [],
            "hashtags": [],
            "source_hashtag": "nifty50",
            "collected_at": "2026-08-04T12:05:00+00:00",
        },
        {
            # Exact duplicate of tweet 1001.
            "tweet_id": "1001",
            "username": "trader1",
            "created_at": "2026-08-04T12:00:00+00:00",
            "text": "Nifty breakout above resistance #nifty50",
            "likes": 5,
            "retweets": 2,
            "replies": 1,
            "mentions": [],
            "hashtags": ["nifty50"],
            "source_hashtag": "nifty50",
            "collected_at": "2026-08-04T12:05:00+00:00",
        },
    ]
    with (raw / "nifty50_test.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return raw


def test_process_raw_files_counts_reconcile(tmp_path: Path, raw_dir: Path) -> None:
    output_dir = tmp_path / "processed"
    rejects_dir = tmp_path / "processed" / "_rejects"

    summary = process_raw_files(
        raw_dir, output_dir, chunk_size_rows=50, rejects_dir=rejects_dir,
        near_duplicate_fields=["text_normalized", "username"], compression="snappy",
    )

    assert summary["in"] == 3
    assert summary["out"] == 1
    assert summary["rejected"] == 1
    assert summary["deduped"] == 1
    assert summary["in"] == summary["out"] + summary["rejected"] + summary["deduped"]


def test_process_raw_files_is_idempotent_across_reruns(tmp_path: Path, raw_dir: Path) -> None:
    """Re-running against the same data/raw and output_dir must not accumulate duplicate
    rows — each run should clear prior part-*.parquet files rather than append to them."""
    output_dir = tmp_path / "processed"
    rejects_dir = tmp_path / "processed" / "_rejects"

    first = process_raw_files(
        raw_dir, output_dir, chunk_size_rows=50, rejects_dir=rejects_dir,
        near_duplicate_fields=["text_normalized", "username"], compression="snappy",
    )
    second = process_raw_files(
        raw_dir, output_dir, chunk_size_rows=50, rejects_dir=rejects_dir,
        near_duplicate_fields=["text_normalized", "username"], compression="snappy",
    )

    assert first["out"] == second["out"]
    df = pd.read_parquet(output_dir)
    assert len(df) == second["out"]
    assert df["tweet_id"].duplicated().sum() == 0


def test_process_raw_files_quarantines_invalid_record_with_reason(tmp_path: Path, raw_dir: Path) -> None:
    output_dir = tmp_path / "processed"
    rejects_dir = tmp_path / "processed" / "_rejects"

    process_raw_files(
        raw_dir, output_dir, chunk_size_rows=50, rejects_dir=rejects_dir,
        near_duplicate_fields=["text_normalized", "username"], compression="snappy",
    )

    reject_files = list(rejects_dir.glob("*.jsonl"))
    assert len(reject_files) >= 1
    rejected_records = [json.loads(line) for path in reject_files for line in path.read_text(encoding="utf-8").splitlines()]
    assert any(r["tweet_id"] == "1002" for r in rejected_records)
    assert all("_reject_reason" in r for r in rejected_records)


def test_process_raw_files_writes_readable_parquet(tmp_path: Path, raw_dir: Path) -> None:
    output_dir = tmp_path / "processed"
    rejects_dir = tmp_path / "processed" / "_rejects"

    process_raw_files(
        raw_dir, output_dir, chunk_size_rows=50, rejects_dir=rejects_dir,
        near_duplicate_fields=["text_normalized", "username"], compression="snappy",
    )

    df = pd.read_parquet(output_dir)
    assert len(df) == 1
    assert df.iloc[0]["tweet_id"] == "1001"
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])


def test_process_raw_files_honors_configured_chunk_size(tmp_path: Path) -> None:
    """chunk_size_rows should be a load-bearing parameter: a small chunk size against a
    single partition must produce that many separate part files."""
    raw = tmp_path / "raw"
    raw.mkdir()
    n_records = 5000
    chunk_size = 500
    with (raw / "synthetic.jsonl").open("w", encoding="utf-8") as f:
        for i in range(n_records):
            record = {
                "tweet_id": str(i),
                "username": f"user{i}",
                "created_at": "2026-08-04T12:00:00+00:00",
                "text": f"Nifty breakout number {i}",
                "likes": 0, "retweets": 0, "replies": 0,
                "mentions": [], "hashtags": ["nifty50"],
                "source_hashtag": "nifty50",
                "collected_at": "2026-08-04T12:00:00+00:00",
            }
            f.write(json.dumps(record) + "\n")

    output_dir = tmp_path / "processed"
    rejects_dir = output_dir / "_rejects"
    summary = process_raw_files(
        raw, output_dir, chunk_size_rows=chunk_size, rejects_dir=rejects_dir,
        near_duplicate_fields=["text_normalized", "username"], compression="snappy",
    )

    assert summary["out"] == n_records
    part_files = list((output_dir / "date=2026-08-04" / "hashtag=nifty50").glob("*.parquet"))
    assert len(part_files) == n_records // chunk_size


def test_process_raw_files_flushes_rejects_mid_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reject buffer must flush at chunk_size_rows too, not just at end-of-run —
    otherwise a large run with many invalid records would hold them all in memory,
    defeating the point of chunking. Counts _flush_rejects calls directly rather than
    inferring it from file contents, since a single final flush would look identical from
    the output alone."""
    raw = tmp_path / "raw"
    raw.mkdir()
    n_records = 1200
    chunk_size = 500
    with (raw / "synthetic.jsonl").open("w", encoding="utf-8") as f:
        for i in range(n_records):
            record = {
                "tweet_id": str(i),
                "username": f"user{i}",
                "created_at": "2026-08-04T12:00:00+00:00",
                "text": "",  # invalid: empty text is rejected by schema.validate
                "likes": 0, "retweets": 0, "replies": 0,
                "mentions": [], "hashtags": ["nifty50"],
                "source_hashtag": "nifty50",
                "collected_at": "2026-08-04T12:00:00+00:00",
            }
            f.write(json.dumps(record) + "\n")

    flush_calls = []
    import src.processing.storage as storage_module

    original_flush = storage_module._flush_rejects

    def _counting_flush(buffer, rejects_dir):
        flush_calls.append(len(buffer))
        original_flush(buffer, rejects_dir)

    monkeypatch.setattr(storage_module, "_flush_rejects", _counting_flush)

    output_dir = tmp_path / "processed"
    rejects_dir = output_dir / "_rejects"
    summary = process_raw_files(
        raw, output_dir, chunk_size_rows=chunk_size, rejects_dir=rejects_dir,
        near_duplicate_fields=["text_normalized", "username"], compression="snappy",
    )

    assert summary["rejected"] == n_records
    # 1200 records / 500 chunk_size -> flushes at 500, 1000 mid-stream, plus a final
    # flush of the remaining 200 -> at least 2 mid-stream flushes, not one giant one.
    assert len(flush_calls) >= 2
    assert all(count <= chunk_size for count in flush_calls)
