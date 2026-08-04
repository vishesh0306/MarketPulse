"""Text cleaning and Unicode normalization for scraped tweet text."""

from __future__ import annotations

import re
import unicodedata

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_ALPHA_RE = re.compile(r"[A-Za-z]")


def clean_text(raw_text: str) -> str:
    """Strips URLs/boilerplate, normalizes whitespace, applies NFC Unicode normalization.

    NFC normalization matters for Devanagari/mixed Hindi-English text: visually identical
    strings can be represented as different byte sequences (composed vs. decomposed
    characters), which would otherwise hash differently in the deduplicator.
    """
    normalized = unicodedata.normalize("NFC", raw_text)
    without_urls = _URL_RE.sub("", normalized)
    return _WHITESPACE_RE.sub(" ", without_urls).strip()


def normalize_for_features(cleaned_text: str) -> str:
    """Lowercases the cleaned, URL-stripped text for the text_normalized column."""
    return cleaned_text.lower()


def detect_lang_hint(text: str) -> str:
    """Heuristic language tag ('en' / 'hi' / 'mixed') for downstream NLP routing.

    Based on the presence of Devanagari-script characters alongside Latin ones — cheap
    and good enough for routing, not a substitute for a real language-detection model.
    """
    devanagari_count = len(_DEVANAGARI_RE.findall(text))
    latin_count = len(_LATIN_ALPHA_RE.findall(text))
    if devanagari_count and latin_count:
        return "mixed"
    if devanagari_count:
        return "hi"
    return "en"
