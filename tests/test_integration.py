"""Integration test: raw JSONL -> processed Parquet -> signal output, asserting on
shape/columns/no exceptions across the full processing -> analysis chain.

Uses a small synthetic sample rather than the real fixture HTML, since the scraper's
extraction step is already covered end to end by tests/test_scraper.py — this test's
job is to prove the *pipeline wiring* (storage.py's output really is generate_signals()'s
expected input) works, not to re-test extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis.signal_generator import OUTPUT_COLUMNS, generate_signals
from src.processing.storage import process_raw_files
from src.utils.config_loader import load_settings


def _write_raw_sample(raw_dir: Path) -> None:
    base_ts = "2026-08-04T{:02d}:00:00+00:00"
    records = []
    for hour in range(10, 14):
        for hashtag in ("nifty50", "sensex"):
            for i in range(5):
                sentiment_word = "bullish breakout" if i % 2 == 0 else "bearish breakdown"
                records.append(
                    {
                        "tweet_id": f"{hashtag}-{hour}-{i}",
                        "username": f"trader-{hashtag}-{i}",
                        "created_at": base_ts.format(hour),
                        # Text varies by hashtag/hour/index so near-duplicate dedup
                        # (matched on text_normalized + username) doesn't collapse
                        # genuinely distinct records from different hashtags/times.
                        "text": f"{hashtag} update at {hour}:00, {sentiment_word} move, tweet {i}",
                        "likes": i * 3,
                        "retweets": i,
                        "replies": i,
                        "mentions": [],
                        "hashtags": [hashtag],
                        "source_hashtag": hashtag,
                        "collected_at": base_ts.format(hour),
                    }
                )
    with (raw_dir / "sample.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_raw_to_processed_to_signals_pipeline(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_raw_sample(raw_dir)

    processed_dir = tmp_path / "processed"
    rejects_dir = processed_dir / "_rejects"
    settings = load_settings()

    processing_summary = process_raw_files(
        raw_dir,
        processed_dir,
        chunk_size_rows=settings.storage.chunk_size_rows,
        rejects_dir=rejects_dir,
        near_duplicate_fields=settings.processing.near_duplicate_hash_fields,
        compression=settings.storage.parquet_compression,
    )
    assert processing_summary["out"] > 0
    assert processing_summary["in"] == (
        processing_summary["out"] + processing_summary["rejected"] + processing_summary["deduped"]
    )

    processed_df = pd.read_parquet(processed_dir)
    assert not processed_df.empty
    assert pd.api.types.is_datetime64_any_dtype(processed_df["created_at"])
    assert processed_df["tweet_id"].duplicated().sum() == 0

    signals = generate_signals(processed_dir, settings)
    assert not signals.empty
    assert list(signals.columns) == OUTPUT_COLUMNS
    assert set(signals["hashtag"]) == {"nifty50", "sensex"}
    scored = signals.dropna(subset=["composite_signal"])
    assert not scored.empty
    assert (scored["ci_lower"] <= scored["composite_signal"]).all()
    assert (scored["composite_signal"] <= scored["ci_upper"]).all()
    assert (signals["tweet_count"] > 0).all()
    assert signals["sentiment_coverage"].between(0.0, 1.0).all()
