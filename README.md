# MarketPulse

Real-time market intelligence system that scrapes Indian stock-market discussion from Twitter/X, cleans and stores it efficiently, and converts the text into quantitative trading signals.

Built for the Qode technical assignment (data collection, processing, analysis) and structured the way the underlying engineering role expects a production system to be built: phased delivery, validation gates between phases, readable self-explaining code, and explicit handling of accuracy, performance, and scale.

## What this repo does

1. **Collects** tweets mentioning Indian market hashtags (`#nifty50`, `#sensex`, `#intraday`, `#banknifty`, and related tags) using Selenium — no paid/official Twitter API.
2. **Cleans and stores** the tweets as deduplicated, Unicode-safe, schema-validated Parquet files partitioned by date.
3. **Converts text to signals** using TF-IDF / embeddings + engineered features (sentiment, engagement-weighted virality, hashtag momentum), aggregated into a composite trading signal with a confidence interval.
4. **Visualizes** the signal and underlying volume/sentiment trends using memory-efficient, sampled/streaming plots suitable for large datasets.
5. **Scales**: every stage is designed so a 10x increase in daily tweet volume requires configuration changes, not a redesign (see `ARCHITECTURE.md`).

## How this repo is organized as a delivery

This project is built in **phases** (see `DEVELOPMENT_ROADMAP.md`). Each phase has an explicit exit checklist (see `VALIDATION_CHECKLIST.md`) that must pass before the next phase starts. `PROMPTS.md` contains the exact prompts used to drive an AI coding agent through each phase and its validation checkpoint, so the build is reproducible and auditable end to end. `PROJECT_STRUCTURE.md` has the full folder/file layout to create locally.

## Setup

### Prerequisites
- Python 3.11+
- Google Chrome + matching ChromeDriver (managed automatically via `webdriver-manager`)
- (Optional, for local Postgres/TimescaleDB/Redis) Docker + Docker Compose

### Install

```bash
git clone <repo-url> market-pulse
cd market-pulse
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # fill in any local overrides
```

### Run the pipeline

```bash
# 1. Collect tweets (Selenium, headless by default)
python -m src.scraper.twitter_scraper --hashtags nifty50,sensex,intraday,banknifty --hours 24 --min-tweets 2000

# 2. Clean, dedupe, store as Parquet
python -m src.processing.storage --input data/raw --output data/processed

# 3. Generate signals
python -m src.analysis.signal_generator --input data/processed --output data/output

# 4. Plot
python -m src.visualization.streaming_plots --input data/output

# Or run everything:
bash scripts/run_pipeline.sh
```

### Run tests

```bash
pytest tests/ -v --cov=src
```

### Validate a phase

```bash
python scripts/validate_phase.py --phase 2
```

## Key design decisions (full detail in `ARCHITECTURE.md`)

- **No paid APIs**: collection uses Selenium against the public Twitter/X search UI, with randomized wait intervals, scroll-based pagination, randomized user-agent/viewport, and exponential backoff on rate-limit signals.
- **Storage**: Parquet (columnar, compressed, fast for analytical reads), partitioned by `date=YYYY-MM-DD`, with a dedup layer keyed on tweet ID + content hash.
- **Signals**: TF-IDF baseline (fast, no GPU, interpretable) with an optional embedding-based path behind a feature flag, combined into a composite score with a bootstrapped confidence interval.
- **Memory-efficient plotting**: reservoir sampling / time-bucket aggregation before plotting, so chart generation cost is O(sample size), not O(dataset size).
- **Scalability**: stateless scraper workers behind a queue, chunked/streaming Parquet writes, and a storage layer that maps cleanly onto PostgreSQL/TimescaleDB for time series and Redis for hot-path caching — matching the target tooling stack.

## Sample output

`docs/sample_output/` holds a sample of raw tweets, the cleaned dataset schema, a sample signal file, and a sample plot, so the pipeline can be verified even if a live scraping session gets rate-limited during review.

## Constraints and honesty notes

- Twitter/X's terms of service restrict automated scraping. This project is built strictly for assignment evaluation purposes, uses only publicly viewable data, respects rate limits, and does not attempt to bypass authentication or paywalls.
- If Twitter/X blocks the scraping session during grading/demo, the sample dataset lets the downstream pipeline (processing → analysis → visualization) still be run and verified end to end.
