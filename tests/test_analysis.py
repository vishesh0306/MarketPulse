"""Tests for src/analysis: feature_engineering, signal_generator, aggregator."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.analysis.aggregator import rollup
from src.analysis.feature_engineering import (
    fit_tfidf,
    hashtag_momentum,
    sentiment_score,
    virality_score,
)
from src.analysis.signal_generator import (
    _bucket_result,  # tested directly since joblib's subprocess workers hide it from coverage
    bootstrap_confidence_interval,
    bucket_tweets,
    composite_signal,
    generate_signals,
)
from src.utils.config_loader import load_settings

# ---- feature_engineering ---------------------------------------------------


def test_virality_score_increases_with_engagement() -> None:
    assert virality_score(0, 0, 0) == 0.0
    assert virality_score(100, 0, 0) < virality_score(100, 50, 0)
    assert virality_score(0, 10, 0) > virality_score(0, 0, 10)  # retweets weighted 2x


def test_sentiment_score_bullish() -> None:
    assert sentiment_score("nifty breakout above resistance", ["breakout"], ["breakdown"]) == 1.0


def test_sentiment_score_bearish() -> None:
    assert sentiment_score("market breakdown expected", ["breakout"], ["breakdown"]) == -1.0


def test_sentiment_score_neutral_when_no_terms_match() -> None:
    assert sentiment_score("just a regular tweet about weather", ["breakout"], ["breakdown"]) == 0.0


def test_sentiment_score_word_boundary_no_partial_match() -> None:
    # "breakoutish" should not match "breakout" as a substring.
    assert sentiment_score("breakoutish nonsense word", ["breakout"], ["breakdown"]) == 0.0


def test_sentiment_lexicon_config_has_no_cancelling_substring_overlap() -> None:
    """Regression test: config/settings.yaml previously listed bare 'short' as bearish
    while also listing 'short covering' as bullish — a tweet using the bullish phrase
    would also match the bearish substring and cancel out. sentiment_score() is a plain
    lexicon counter with no special-casing for this, so the fix has to be (and is) lexicon
    curation: no bearish term may be a substring of a bullish term, and vice versa."""
    settings = load_settings()
    bullish = settings.analysis.sentiment_lexicon.bullish
    bearish = settings.analysis.sentiment_lexicon.bearish

    for bull_term in bullish:
        for bear_term in bearish:
            assert bear_term not in bull_term, f"{bear_term!r} is a substring of bullish {bull_term!r}"
            assert bull_term not in bear_term, f"{bull_term!r} is a substring of bearish {bear_term!r}"

    score = sentiment_score("short covering rally in banknifty", bullish, bearish)
    assert score == 1.0


def test_hashtag_momentum_higher_for_more_co_occurring_tags() -> None:
    bucket = [["nifty50", "sensex"], ["nifty50", "sensex", "banknifty"], ["intraday"]]
    momentum = hashtag_momentum(bucket)
    assert momentum["intraday"] == 0.0
    assert momentum["nifty50"] > momentum["intraday"]


def test_fit_tfidf_returns_matrix_matching_corpus_size() -> None:
    texts = ["nifty breakout today", "sensex crash today", "banknifty rally continues"]
    vectorizer, matrix = fit_tfidf(texts, max_features=50, ngram_range=(1, 1), min_df=1, extra_stopwords=["rt"])
    assert matrix.shape[0] == len(texts)
    assert matrix.shape[1] <= 50


# ---- signal_generator -------------------------------------------------------


def test_bucket_tweets_floors_to_window() -> None:
    df = pd.DataFrame({"created_at": pd.to_datetime(["2026-08-04T12:07:00Z", "2026-08-04T12:14:00Z"])})
    bucketed = bucket_tweets(df, bucket_minutes=15)
    assert bucketed["bucket_start"].iloc[0] == pd.Timestamp("2026-08-04T12:00:00Z")
    assert bucketed["bucket_start"].iloc[1] == pd.Timestamp("2026-08-04T12:00:00Z")


def test_composite_signal_matches_weighted_sum() -> None:
    features = pd.DataFrame({"sentiment": [1.0, -1.0], "virality_norm": [0.5, 0.5], "momentum_norm": [0.2, 0.2]})
    weights = {"sentiment": 0.5, "virality": 0.3, "hashtag_momentum": 0.2}
    result = composite_signal(features, weights)
    expected = 0.5 * 0.0 + 0.3 * 0.5 + 0.2 * 0.2
    assert result == pytest.approx(expected)


def test_composite_signal_empty_features_returns_zero() -> None:
    empty = pd.DataFrame(columns=["sentiment", "virality_norm", "momentum_norm"])
    assert composite_signal(empty, {"sentiment": 1.0, "virality": 0.0, "hashtag_momentum": 0.0}) == 0.0


def test_bucket_result_shapes_output_dict() -> None:
    """Direct unit test for the joblib worker function — generate_signals() also exercises
    this, but via Parallel(n_jobs=-1), which can run in a subprocess invisible to coverage
    even though the behavior is genuinely tested end to end."""
    group = pd.DataFrame(
        {"sentiment": [1.0, -1.0, 1.0], "virality_norm": [0.5, 0.5, 0.5], "momentum_norm": [0.2, 0.2, 0.2]}
    )
    weights = {"sentiment": 0.5, "virality": 0.3, "hashtag_momentum": 0.2}
    result = _bucket_result(("nifty50", pd.Timestamp("2026-08-04T12:00:00Z")), group, weights, 50, 0.90, fallback_std=0.1)

    assert result["hashtag"] == "nifty50"
    assert result["tweet_count"] == 3
    assert result["ci_lower"] <= result["composite_signal"] <= result["ci_upper"]


def test_bootstrap_ci_narrower_for_larger_sample() -> None:
    rng = np.random.default_rng(0)
    small_sample = rng.normal(0, 1, size=3).tolist()
    large_sample = rng.normal(0, 1, size=200).tolist()

    small_lo, small_hi = bootstrap_confidence_interval(small_sample, n_resamples=200, confidence_level=0.90, seed=1)
    large_lo, large_hi = bootstrap_confidence_interval(large_sample, n_resamples=200, confidence_level=0.90, seed=1)

    assert (small_hi - small_lo) > (large_hi - large_lo)


def test_bootstrap_ci_single_tweet_is_not_falsely_narrow() -> None:
    """Regression test: a single-value bucket must not collapse to zero-width — that would
    claim maximum confidence exactly where there's the least data to support it."""
    lo, hi = bootstrap_confidence_interval([0.5], n_resamples=100, confidence_level=0.90, fallback_std=0.3)
    assert hi > lo


def test_bootstrap_ci_empty_returns_zero_width() -> None:
    lo, hi = bootstrap_confidence_interval([], n_resamples=100, confidence_level=0.90)
    assert lo == hi == 0.0


# ---- aggregator --------------------------------------------------------------


def test_rollup_weights_by_tweet_count() -> None:
    df = pd.DataFrame(
        {
            "hashtag": ["nifty50", "nifty50"],
            "bucket_start": [
                datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc),
                datetime(2026, 8, 4, 12, 15, tzinfo=timezone.utc),
            ],
            "composite_signal": [1.0, 0.0],
            "ci_lower": [0.8, -0.2],
            "ci_upper": [1.2, 0.2],
            "tweet_count": [90, 10],
        }
    )
    result = rollup(df, "1h")
    assert len(result) == 1
    # Weighted mean should lean heavily toward the 90-tweet bucket's value of 1.0.
    assert result.iloc[0]["composite_signal"] == pytest.approx(0.9)
    assert result.iloc[0]["tweet_count"] == 100


def test_generate_signals_empty_processed_dir_returns_empty_frame(tmp_path) -> None:
    empty_df = pd.DataFrame(
        columns=["tweet_id", "username", "created_at", "text", "text_normalized", "likes",
                 "retweets", "replies", "mentions", "hashtags", "source_hashtag", "lang_hint"]
    )
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    empty_df.to_parquet(processed_dir / "part-0.parquet")

    result = generate_signals(processed_dir, load_settings())
    assert result.empty
    assert list(result.columns) == ["bucket_start", "hashtag", "composite_signal", "ci_lower", "ci_upper", "tweet_count"]


def test_rollup_empty_returns_empty() -> None:
    empty = pd.DataFrame(columns=["hashtag", "bucket_start", "composite_signal", "ci_lower", "ci_upper", "tweet_count"])
    assert rollup(empty, "1h").empty


# ---- generate_signals (integration) -----------------------------------------


def test_generate_signals_end_to_end(tmp_path) -> None:
    """Raw-shaped processed data -> full generate_signals() pipeline, asserting on
    output shape/columns and that no exception is raised — the integration-style check
    the roadmap calls for, run against a tiny synthetic dataset rather than the full one."""
    base = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
    rows = [
        {
            "tweet_id": str(i),
            "username": f"user{i % 3}",
            "created_at": base,
            "collected_at": base,
            "text": "Nifty breakout above resistance" if i % 2 == 0 else "Market breakdown expected today",
            "text_normalized": "nifty breakout above resistance" if i % 2 == 0 else "market breakdown expected today",
            "likes": i,
            "retweets": i % 5,
            "replies": i % 3,
            "mentions": [],
            "hashtags": ["nifty50"],
            "source_hashtag": "nifty50",
            "lang_hint": "en",
        }
        for i in range(1, 21)
    ]
    df = pd.DataFrame(rows)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    df.to_parquet(processed_dir / "part-0.parquet")

    settings = load_settings()
    result = generate_signals(processed_dir, settings)

    assert list(result.columns) == ["bucket_start", "hashtag", "composite_signal", "ci_lower", "ci_upper", "tweet_count"]
    assert len(result) == 1
    assert result.iloc[0]["hashtag"] == "nifty50"
    assert result.iloc[0]["tweet_count"] == 20
    assert result.iloc[0]["ci_lower"] <= result.iloc[0]["composite_signal"] <= result.iloc[0]["ci_upper"]
