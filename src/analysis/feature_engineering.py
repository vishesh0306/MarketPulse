"""TF-IDF vectorization plus engineered features: virality, lexicon sentiment, hashtag momentum."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(texts: list[str], max_features: int, ngram_range: tuple[int, int], min_df: int,
              extra_stopwords: list[str]) -> tuple[TfidfVectorizer, spmatrix]:
    """Fits a TF-IDF vectorizer on text_normalized and returns (vectorizer, sparse matrix)."""
    raise NotImplementedError


def virality_score(likes: int, retweets: int, replies: int) -> float:
    """Engagement-weighted virality: log1p(likes + 2*retweets + replies)."""
    raise NotImplementedError


def sentiment_score(text_normalized: str, bullish_terms: list[str], bearish_terms: list[str]) -> float:
    """Lexicon-based sentiment in [-1, 1] from bullish/bearish term matches."""
    raise NotImplementedError


def hashtag_momentum(bucket_hashtags: list[list[str]]) -> dict[str, float]:
    """Co-occurrence momentum score per hashtag within a time bucket."""
    raise NotImplementedError


def build_feature_frame(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Assembles all engineered features (virality, sentiment, momentum, TF-IDF summary) per row."""
    raise NotImplementedError
