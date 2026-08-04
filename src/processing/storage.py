"""Reads data/raw/*.jsonl, applies cleaner -> schema validation -> deduplicator, writes
partitioned Parquet to data/processed/date=YYYY-MM-DD/hashtag=X/, chunked to bound memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def process_raw_files(input_dir: Path, output_dir: Path, chunk_size_rows: int) -> dict[str, Any]:
    """Runs the full clean -> validate -> dedup -> write pipeline over all raw JSONL files.

    Processes in chunks (default from config) rather than loading the full raw dataset at
    once. Returns a run-summary dict (counts in/out/rejected/deduped) for logging.
    """
    raise NotImplementedError


def write_partitioned_parquet(records: list[dict[str, Any]], output_dir: Path, compression: str) -> None:
    """Writes records to date=YYYY-MM-DD/hashtag=X/part-*.parquet using pyarrow."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, dedupe, and store raw tweets as Parquet.")
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
