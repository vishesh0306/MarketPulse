"""Typed schema for a processed tweet record, matching ARCHITECTURE.md section 4 exactly."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class TweetRecord(BaseModel):
    tweet_id: str = Field(min_length=1)
    username: str = Field(min_length=1)
    created_at: datetime
    collected_at: datetime
    text: str = Field(min_length=1)
    text_normalized: str
    likes: int = Field(ge=0)
    retweets: int = Field(ge=0)
    replies: int = Field(ge=0)
    mentions: list[str]
    hashtags: list[str]
    source_hashtag: str = Field(min_length=1)
    lang_hint: str


def validate(record: dict[str, Any]) -> tuple[bool, str | None]:
    """Validates a raw record dict against TweetRecord. Returns (is_valid, reason_if_invalid)."""
    try:
        TweetRecord.model_validate(record)
    except ValidationError as exc:
        first_error = exc.errors()[0]
        field = ".".join(str(part) for part in first_error["loc"]) or "record"
        return False, f"{field}: {first_error['msg']}"
    return True, None
