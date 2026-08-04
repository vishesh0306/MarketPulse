"""TF-IDF vectorization plus engineered features: virality, lexicon sentiment, hashtag momentum."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from src.utils.config_loader import AnalysisConfig


def fit_tfidf(
    texts: list[str],
    max_features: int,
    ngram_range: tuple[int, int],
    min_df: int,
    extra_stopwords: list[str],
) -> tuple[TfidfVectorizer, spmatrix]:
    """Fits a TF-IDF vectorizer on text_normalized and returns (vectorizer, sparse matrix).

    This is the "text-to-numerical-vector" deliverable: a full TF-IDF matrix over the
    corpus. It is fit once per analysis run (batch, not per-bucket) — the composite
    trading signal itself is built from the more directly explainable features below
    (sentiment/virality/momentum) rather than raw TF-IDF weights.
    """
    stopwords = sorted(ENGLISH_STOP_WORDS.union(extra_stopwords))
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words=stopwords,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def virality_score(likes: int, retweets: int, replies: int) -> float:
    """Engagement-weighted virality: log1p(likes + 2*retweets + replies)."""
    return float(np.log1p(likes + 2 * retweets + replies))


_WORD_BOUNDARY_CACHE: dict[str, re.Pattern[str]] = {}


def _term_pattern(term: str) -> re.Pattern[str]:
    pattern = _WORD_BOUNDARY_CACHE.get(term)
    if pattern is None:
        pattern = re.compile(r"\b" + re.escape(term.lower()) + r"\b")
        _WORD_BOUNDARY_CACHE[term] = pattern
    return pattern


def sentiment_score(text_normalized: str, bullish_terms: list[str], bearish_terms: list[str]) -> float:
    """Lexicon-based sentiment in [-1, 1] from bullish/bearish term matches.

    (bullish_count - bearish_count) / (bullish_count + bearish_count); 0.0 when no
    lexicon terms match (neutral/unknown, not a guess in either direction).
    """
    bullish_count = sum(1 for term in bullish_terms if _term_pattern(term).search(text_normalized))
    bearish_count = sum(1 for term in bearish_terms if _term_pattern(term).search(text_normalized))
    total = bullish_count + bearish_count
    if total == 0:
        return 0.0
    return (bullish_count - bearish_count) / total


def hashtag_momentum(bucket_hashtags: list[list[str]]) -> dict[str, float]:
    """Co-occurrence momentum per hashtag within a time bucket: the average number of
    *other* target hashtags each tweet mentioning this hashtag also mentions. Higher means
    more cross-hashtag chatter — a proxy for a broader market conversation, not just one
    hashtag's own volume.
    """
    co_occurrence_totals: Counter[str] = Counter()
    occurrence_counts: Counter[str] = Counter()
    for tags in bucket_hashtags:
        unique_tags = set(tags)
        for tag in unique_tags:
            occurrence_counts[tag] += 1
            co_occurrence_totals[tag] += len(unique_tags) - 1
    return {
        tag: co_occurrence_totals[tag] / occurrence_counts[tag] if occurrence_counts[tag] else 0.0
        for tag in occurrence_counts
    }


def build_feature_frame(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Assembles per-tweet engineered features (virality, normalized virality, sentiment).

    Hashtag momentum is intentionally excluded here — it's a bucket-level quantity (about
    co-occurrence across tweets in a time window), computed separately in signal_generator
    once tweets are bucketed, not a per-row feature.
    """
    out = df.copy()
    out["virality"] = np.log1p(out["likes"] + 2 * out["retweets"] + out["replies"])

    v_min, v_max = out["virality"].min(), out["virality"].max()
    v_range = (v_max - v_min) or 1.0
    out["virality_norm"] = (out["virality"] - v_min) / v_range

    bullish = config.sentiment_lexicon.bullish
    bearish = config.sentiment_lexicon.bearish
    out["sentiment"] = out["text_normalized"].apply(lambda t: sentiment_score(t, bullish, bearish))

    return out


def tfidf_summary(vectorizer: TfidfVectorizer, matrix: spmatrix) -> dict[str, Any]:
    """Compact, loggable summary of a fitted TF-IDF matrix (shape, sparsity, top terms)."""
    vocab_size = len(vectorizer.vocabulary_)
    nnz = matrix.nnz
    total_cells = matrix.shape[0] * matrix.shape[1] if matrix.shape[1] else 1
    mean_weight_per_term = np.asarray(matrix.sum(axis=0)).ravel()
    top_indices = mean_weight_per_term.argsort()[::-1][:10]
    feature_names = vectorizer.get_feature_names_out()
    top_terms = [feature_names[i] for i in top_indices]
    return {
        "documents": matrix.shape[0],
        "vocabulary_size": vocab_size,
        "sparsity": 1 - (nnz / total_cells),
        "top_terms": top_terms,
    }
