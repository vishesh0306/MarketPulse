"""Two-tier dedup: exact dedup on tweet_id, near-duplicate dedup on a content hash."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Iterator

_FIELD_SEPARATOR = "\x1f"


def content_hash(field_values: list[str]) -> str:
    """Returns a SHA-256 hash over the given field values, for near-duplicate detection."""
    payload = _FIELD_SEPARATOR.join(field_values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dedup_records(
    records: Iterable[dict[str, Any]], near_duplicate_fields: list[str]
) -> Iterator[dict[str, Any]]:
    """Yields records with exact (tweet_id) and near-duplicate (content_hash) dupes removed.

    Uses in-memory sets of seen IDs/hashes for O(1) average-case lookup per record, per
    ARCHITECTURE.md section 3.1's data-structure choice — O(n) memory in unique records,
    not the full dataset held twice.
    """
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for record in records:
        tweet_id = record["tweet_id"]
        if tweet_id in seen_ids:
            continue
        seen_ids.add(tweet_id)

        chash = content_hash([str(record.get(field, "")) for field in near_duplicate_fields])
        if chash in seen_hashes:
            continue
        seen_hashes.add(chash)

        yield record
