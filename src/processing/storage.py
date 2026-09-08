"""Reads data/raw/*.jsonl, applies cleaner -> schema validation -> deduplicator, writes
partitioned Parquet to data/processed/date=YYYY-MM-DD/hashtag=X/, chunked to bound memory.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tracemalloc
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from src.processing.cleaner import clean_text, detect_lang_hint, normalize_for_features
from src.processing.deduplicator import dedup_records
from src.processing.schema import TweetRecord, validate
from src.utils.config_loader import load_settings
from src.utils.logger import get_logger, set_level, write_run_summary

logger = get_logger("processing_storage")

PARQUET_SCHEMA = pa.schema(
    [
        ("tweet_id", pa.string()),
        ("username", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("collected_at", pa.timestamp("us", tz="UTC")),
        ("text", pa.string()),
        ("text_normalized", pa.string()),
        ("likes", pa.int32()),
        ("retweets", pa.int32()),
        ("replies", pa.int32()),
        ("mentions", pa.list_(pa.string())),
        ("hashtags", pa.list_(pa.string())),
        ("source_hashtag", pa.string()),
        ("lang_hint", pa.string()),
    ]
)


def _iter_raw_records(input_dir: Path) -> Iterator[tuple[dict[str, Any] | None, str | None]]:
    """Yields (record, error) pairs. A malformed line yields (None, error) instead of
    raising, so it quarantines like any other invalid record instead of aborting the
    whole run partway through — with the previous run's output already gone.
    """
    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line), None
                except json.JSONDecodeError as exc:
                    yield None, f"malformed_json: {exc}"


def _clean_record(raw: dict[str, Any]) -> dict[str, Any]:
    text = clean_text(raw.get("text", ""))
    return {
        **raw,
        "text": text,
        "text_normalized": normalize_for_features(text),
        "lang_hint": detect_lang_hint(text),
    }


def _reject_bucket(reason: str) -> str:
    field = reason.split(":", 1)[0].strip()
    return re.sub(r"[^a-zA-Z0-9_]+", "_", field) or "unknown"


def _write_chunk(chunk: list[dict[str, Any]], output_dir: Path, compression: str) -> None:
    by_partition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in chunk:
        date_str = record["created_at"].strftime("%Y-%m-%d")
        by_partition[(date_str, record["source_hashtag"])].append(record)

    for (date_str, hashtag), records in by_partition.items():
        part_dir = output_dir / f"date={date_str}" / f"hashtag={hashtag}"
        part_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(records, schema=PARQUET_SCHEMA)
        part_path = part_dir / f"part-{uuid.uuid4().hex[:12]}.parquet"
        pq.write_table(table, part_path, compression=compression)


def _tmp_sibling(path: Path) -> Path:
    return path.parent / f".{path.name}.tmp"


def _carry_over_markers(final_dir: Path, tmp_dir: Path) -> None:
    """Copies dotfiles from the outgoing directory into the incoming one.

    The swap below replaces final_dir wholesale, which also throws away anything in it
    that wasn't written by this run — including the tracked `.gitkeep` files that keep
    `data/processed/` and `data/processed/_rejects/` in git. Every run was silently
    deleting them and dirtying the working tree. Pipeline output is only ever
    `date=*/hashtag=*/part-*.parquet` and reject JSONL, so a leading dot is a reliable
    marker of "placed here deliberately, not produced by the pipeline".
    """
    if not final_dir.exists():
        return
    for marker in final_dir.rglob(".*"):
        if not marker.is_file():
            continue
        destination = tmp_dir / marker.relative_to(final_dir)
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(marker, destination)


def _replace_directory(tmp_dir: Path, final_dir: Path) -> None:
    """Swaps tmp_dir into final_dir's place, atomically per directory rename.

    final_dir's previous contents stay intact right up until tmp_dir (the fully-written
    new output) is ready to take its place, so a crash partway through a run — a
    malformed input record that somehow still got past validation, a full disk, anything
    — never leaves less output on disk than there was before the run started, the way
    clearing output_dir up front before writing anything new would.
    """
    _carry_over_markers(final_dir, tmp_dir)
    backup_dir = final_dir.parent / f".{final_dir.name}.bak"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if final_dir.exists():
        final_dir.rename(backup_dir)
    tmp_dir.rename(final_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def _flush_rejects(buffer: list[tuple[str, dict[str, Any]]], rejects_dir: Path) -> None:
    if not buffer:
        return
    rejects_dir.mkdir(parents=True, exist_ok=True)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bucket, record in buffer:
        by_bucket[bucket].append(record)
    for bucket, records in by_bucket.items():
        path = rejects_dir / f"{bucket}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def process_raw_files(
    input_dir: Path,
    output_dir: Path,
    chunk_size_rows: int,
    rejects_dir: Path,
    near_duplicate_fields: list[str],
    compression: str,
) -> dict[str, Any]:
    """Runs the full clean -> validate -> dedup -> write pipeline over all raw JSONL files.

    Processes in chunks (default from config) rather than loading the full raw dataset
    into memory at once. Idempotent: writes to a temporary directory and only swaps it
    into place once the run completes, so re-running against the same data/raw always
    reproduces the same row count rather than accumulating duplicates, and a run that
    fails partway through leaves the previous output untouched rather than deleted.
    Returns a run-summary dict (counts in/out/rejected/deduped).
    """
    tmp_output_dir = _tmp_sibling(output_dir)
    if tmp_output_dir.exists():
        shutil.rmtree(tmp_output_dir)
    tmp_output_dir.mkdir(parents=True)

    # rejects_dir is nested under output_dir by default (output_dir/_rejects), in which
    # case one swap of tmp_output_dir carries it along; otherwise it gets its own.
    rejects_nested = rejects_dir == output_dir or output_dir in rejects_dir.parents
    if rejects_nested:
        tmp_rejects_dir = tmp_output_dir / rejects_dir.relative_to(output_dir)
    else:
        tmp_rejects_dir = _tmp_sibling(rejects_dir)
        if tmp_rejects_dir.exists():
            shutil.rmtree(tmp_rejects_dir)
        tmp_rejects_dir.mkdir(parents=True)

    counts = {"in": 0, "out": 0, "rejected": 0}
    reject_reason_counts: dict[str, int] = defaultdict(int)
    reject_buffer: list[tuple[str, dict[str, Any]]] = []

    def validated_stream() -> Iterator[dict[str, Any]]:
        for raw, parse_error in _iter_raw_records(input_dir):
            counts["in"] += 1
            if parse_error is not None:
                counts["rejected"] += 1
                bucket = _reject_bucket(parse_error)
                reject_reason_counts[bucket] += 1
                reject_buffer.append((bucket, {"_reject_reason": parse_error}))
                if len(reject_buffer) >= chunk_size_rows:
                    _flush_rejects(reject_buffer, tmp_rejects_dir)
                    reject_buffer.clear()
                continue
            assert raw is not None  # parse_error is None, so json.loads succeeded
            cleaned = _clean_record(raw)
            is_valid, reason = validate(cleaned)
            if not is_valid:
                counts["rejected"] += 1
                bucket = _reject_bucket(reason or "unknown")
                reject_reason_counts[bucket] += 1
                reject_buffer.append((bucket, {**cleaned, "_reject_reason": reason}))
                if len(reject_buffer) >= chunk_size_rows:
                    _flush_rejects(reject_buffer, tmp_rejects_dir)
                    reject_buffer.clear()
                continue
            yield cleaned

    chunk: list[dict[str, Any]] = []
    for record in dedup_records(validated_stream(), near_duplicate_fields):
        typed_record = TweetRecord.model_validate(record).model_dump()
        chunk.append(typed_record)
        counts["out"] += 1
        if len(chunk) >= chunk_size_rows:
            _write_chunk(chunk, tmp_output_dir, compression)
            chunk.clear()

    _write_chunk(chunk, tmp_output_dir, compression)
    _flush_rejects(reject_buffer, tmp_rejects_dir)

    _replace_directory(tmp_output_dir, output_dir)
    if not rejects_nested:
        _replace_directory(tmp_rejects_dir, rejects_dir)

    # Derived rather than separately tracked, so the in = out + rejected + deduped
    # invariant holds by construction instead of by two counters staying in sync.
    counts["deduped"] = counts["in"] - counts["out"] - counts["rejected"]

    return {**counts, "reject_reasons": dict(reject_reason_counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, dedupe, and store raw tweets as Parquet.")
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    args = parser.parse_args()

    settings = load_settings()
    set_level(logger, settings.logging.level)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    rejects_dir = Path(settings.storage.rejects_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    summary = process_raw_files(
        input_dir,
        output_dir,
        settings.storage.chunk_size_rows,
        rejects_dir,
        settings.processing.near_duplicate_hash_fields,
        settings.storage.parquet_compression,
    )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    summary["peak_memory_mb"] = round(peak_bytes / (1024 * 1024), 2)

    summary_path = write_run_summary(settings.logging.dir, "processing", summary)
    logger.info(
        "processing run complete",
        extra={"extra_fields": {**summary, "summary_path": str(summary_path)}},
    )


if __name__ == "__main__":
    main()
