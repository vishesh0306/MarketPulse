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
    """Yields records with exact ((tweet_id, source_hashtag)) and near-duplicate
    (content_hash) dupes removed.

    Keying on tweet_id alone would drop a tweet that legitimately mentions more than one
    target hashtag: it's collected once per hashtag search, each copy carrying a different
    source_hashtag, and only the first (by raw-file iteration order) would survive. Since
    storage partitions by source_hashtag, that silently erases the tweet from every hashtag
    except the one that happened to sort first. Keying on the pair keeps one copy per
    hashtag the tweet is actually relevant to.

    Uses in-memory sets of seen IDs/hashes for O(1) average-case lookup per record —
    O(n) memory in unique records, not the full dataset held twice.
    """
    seen_ids: set[tuple[str, str]] = set()
    seen_hashes: set[str] = set()
    for record in records:
        dedup_key = (record["tweet_id"], record["source_hashtag"])
        if dedup_key in seen_ids:
            continue
        seen_ids.add(dedup_key)

        chash = content_hash([str(record.get(field, "")) for field in near_duplicate_fields])
        if chash in seen_hashes:
            continue
        seen_hashes.add(chash)

        yield record
