"""Text cleaning and Unicode normalization for scraped tweet text."""

from __future__ import annotations


def clean_text(raw_text: str) -> str:
    """Strips URLs/boilerplate, normalizes whitespace, applies NFC Unicode normalization."""
    raise NotImplementedError


def normalize_for_features(cleaned_text: str) -> str:
    """Lowercases and strips residual punctuation for the text_normalized column."""
    raise NotImplementedError


def detect_lang_hint(text: str) -> str:
    """Heuristic language tag ('en' / 'hi' / 'mixed') for downstream NLP routing."""
    raise NotImplementedError
