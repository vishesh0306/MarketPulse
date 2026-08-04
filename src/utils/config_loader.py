"""Loads config/settings.yaml into a typed, validated object — the single source of truth
for hashtags, rate limits, storage paths, TF-IDF params, and signal weights."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


class ScrollConfig(BaseModel):
    min_pause_seconds: float
    max_pause_seconds: float
    max_scrolls_per_session: int
    min_scroll_pixels: int
    max_scroll_pixels: int


class RateLimiterConfig(BaseModel):
    bucket_capacity: int
    refill_rate_per_second: float
    backoff_base_seconds: float
    backoff_max_seconds: float
    backoff_multiplier: float


class AntiDetectionConfig(BaseModel):
    rotate_user_agent_every_n_actions: int
    soft_block_indicators: list[str]


class ScraperConfig(BaseModel):
    hashtags: list[str]
    search_url_template: str
    hours_lookback: int
    min_tweets_target: int
    headless: bool
    worker_pool_size: int
    scroll: ScrollConfig
    rate_limiter: RateLimiterConfig
    anti_detection: AntiDetectionConfig


class StorageConfig(BaseModel):
    raw_dir: str
    processed_dir: str
    rejects_dir: str
    output_dir: str
    signals_dir: str
    plots_dir: str
    chunk_size_rows: int
    parquet_compression: str
    target_part_file_mb: int


class ProcessingConfig(BaseModel):
    near_duplicate_hash_fields: list[str]


class TfidfConfig(BaseModel):
    max_features: int
    ngram_range: tuple[int, int]
    min_df: int
    stopwords_extra: list[str]


class BootstrapConfig(BaseModel):
    n_resamples: int
    confidence_level: float


class SentimentLexicon(BaseModel):
    bullish: list[str]
    bearish: list[str]


class AnalysisConfig(BaseModel):
    tfidf: TfidfConfig
    bucket_minutes: int
    bootstrap: BootstrapConfig
    signal_weights: dict[str, float]
    sentiment_lexicon: SentimentLexicon


class AggregationConfig(BaseModel):
    rollup_windows: list[str]


class VisualizationConfig(BaseModel):
    reservoir_sample_size: int
    dpi: int


class LoggingConfig(BaseModel):
    level: str
    dir: str


class Settings(BaseModel):
    scraper: ScraperConfig
    storage: StorageConfig
    processing: ProcessingConfig
    analysis: AnalysisConfig
    aggregation: AggregationConfig
    visualization: VisualizationConfig
    logging: LoggingConfig = Field(alias="logging")

    model_config = {"populate_by_name": True}


@lru_cache(maxsize=8)
def load_settings(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Loads and validates config/settings.yaml. Cached per path so repeated calls are free."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)
