"""Two-tier dedup: exact dedup on tweet_id, near-duplicate dedup on a content hash."""

from __future__ import annotations

from typing import Any, Iterable, Iterator


def content_hash(text_normalized: str, username: str) -> str:
    """Returns a SHA-256 hash of normalized text + author, for near-duplicate detection."""
    raise NotImplementedError


def dedup_records(records: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yields records with exact (tweet_id) and near-duplicate (content_hash) dupes removed."""
    raise NotImplementedError
