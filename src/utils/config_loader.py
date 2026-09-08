"""Loads config/settings.yaml into a typed, validated object — the single source of truth
for hashtags, rate limits, storage paths, TF-IDF params, and signal weights."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

ParquetCompression = Literal["snappy", "gzip", "brotli", "lz4", "zstd"]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


class PaginationConfig(BaseModel):
    min_pause_seconds: float
    max_pause_seconds: float
    max_pages_per_session: int
    page_render_timeout_seconds: int


class RateLimiterConfig(BaseModel):
    bucket_capacity: int
    refill_rate_per_second: float
    backoff_base_seconds: float
    backoff_max_seconds: float
    backoff_multiplier: float


class AntiDetectionConfig(BaseModel):
    soft_block_indicators: list[str]


class ScraperConfig(BaseModel):
    # The four hashtags named in the assignment — always collected.
    hashtags: list[str]
    # "Similar hashtags" (the assignment's phrasing) collected alongside the core four to
    # widen coverage of Indian market chatter toward the 2,000-tweet target. Empty is fine.
    related_hashtags: list[str] = []
    search_path_template: str
    nitter_hosts: list[str]
    hours_lookback: int
    min_tweets_target: int
    headless: bool
    worker_pool_size: int
    pagination: PaginationConfig
    rate_limiter: RateLimiterConfig
    anti_detection: AntiDetectionConfig

    @property
    def all_hashtags(self) -> list[str]:
        """Core + related, de-duplicated, order preserved (core first)."""
        seen: dict[str, None] = {}
        for tag in [*self.hashtags, *self.related_hashtags]:
            seen.setdefault(tag, None)
        return list(seen)


class StorageConfig(BaseModel):
    raw_dir: str
    processed_dir: str
    rejects_dir: str
    output_dir: str
    signals_dir: str
    plots_dir: str
    chunk_size_rows: int
    parquet_compression: ParquetCompression


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
    random_seed: int


class SentimentLexicon(BaseModel):
    bullish: list[str]
    bearish: list[str]


class AnalysisConfig(BaseModel):
    tfidf: TfidfConfig
    bucket_minutes: int
    min_bucket_tweets: int = 1
    bootstrap: BootstrapConfig
    signal_weights: dict[str, float]
    sentiment_lexicon: SentimentLexicon
    filter_market_hours: bool


class AggregationConfig(BaseModel):
    rollup_windows: list[str]


class VisualizationConfig(BaseModel):
    reservoir_sample_size: int
    dpi: int


class LoggingConfig(BaseModel):
    level: str
    dir: str


class RealtimeRssConfig(BaseModel):
    path_template: str
    poll_interval_seconds: float
    request_timeout_seconds: float
    max_items_per_poll: int


class RealtimeDedupConfig(BaseModel):
    id_prefix: str
    content_prefix: str
    ttl_seconds: int


class RealtimeApiConfig(BaseModel):
    host: str
    port: int


class RealtimeConfig(BaseModel):
    redis_url: str
    backfill_hours: int
    bucket_seal_grace_seconds: int
    queue_maxsize: int
    queue_full_policy: Literal["block", "drop_oldest"]
    rss: RealtimeRssConfig
    dedup: RealtimeDedupConfig
    api: RealtimeApiConfig


class Settings(BaseModel):
    scraper: ScraperConfig
    storage: StorageConfig
    processing: ProcessingConfig
    analysis: AnalysisConfig
    aggregation: AggregationConfig
    visualization: VisualizationConfig
    logging: LoggingConfig = Field(alias="logging")
    realtime: RealtimeConfig

    model_config = {"populate_by_name": True}


@lru_cache(maxsize=8)
def load_settings(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Loads and validates config/settings.yaml. Cached per path so repeated calls are free."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)
