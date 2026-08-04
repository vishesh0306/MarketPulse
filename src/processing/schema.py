"""Typed schema for a processed tweet record, matching ARCHITECTURE.md section 4 exactly."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class TweetRecord(BaseModel):
    tweet_id: str
    username: str
    created_at: datetime
    collected_at: datetime
    text: str
    text_normalized: str
    likes: int
    retweets: int
    replies: int
    mentions: list[str]
    hashtags: list[str]
    source_hashtag: str
    lang_hint: str


def validate(record: dict[str, Any]) -> tuple[bool, str | None]:
    """Validates a raw record dict against TweetRecord. Returns (is_valid, reason_if_invalid)."""
    raise NotImplementedError
