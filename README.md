# MarketPulse

Real-time market intelligence system that scrapes Indian stock-market discussion from Twitter/X, cleans and stores it efficiently, and converts the text into quantitative trading signals.

Built for the Qode technical assignment (data collection, processing, analysis) and structured the way the underlying engineering role expects a production system to be built: phased delivery, validation gates between phases, readable self-explaining code, and explicit handling of accuracy, performance, and scale.

## What this repo does

1. **Collects** tweets mentioning Indian market hashtags (`#nifty50`, `#sensex`, `#intraday`, `#banknifty`) using Selenium against Nitter — an open-source, login-free HTML front-end for X.com's public content (x.com's own search UI now redirects unauthenticated requests to a login wall; see `docs/approach.md`). No paid/official Twitter API, no `tweepy`. Automatic failover across multiple configured Nitter hosts, and a `ProcessPoolExecutor`-based worker pool so hashtags scrape concurrently.
2. **Cleans and stores** the tweets as deduplicated, Unicode-safe, schema-validated Parquet files partitioned by date and hashtag.
3. **Converts text to signals** using TF-IDF vectorization + engineered features (lexicon-based sentiment tuned for Indian market slang, engagement-weighted virality, hashtag co-occurrence momentum), aggregated into a composite trading signal with a bootstrapped confidence interval per time bucket.
4. **Visualizes** the signal and underlying volume/engagement trends using memory-efficient, sampled/streaming plots suitable for large datasets.
5. **Scales**: every stage is designed so a 10x increase in daily tweet volume requires configuration changes, not a redesign (see `ARCHITECTURE.md` §6).

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
```

`.env.example` is only needed if you extend storage to the optional local
Postgres/TimescaleDB/Redis stack in `docker-compose.yml` — the core pipeline below reads
all its configuration from `config/settings.yaml` and needs no `.env` file.

### Run the pipeline

```bash
# 1. Collect tweets (Selenium, headless by default; --workers scrapes hashtags concurrently)
python -m src.scraper.twitter_scraper --hashtags nifty50,sensex,intraday,banknifty --hours 24 --min-tweets 2000 --workers 4

# 2. Clean, dedupe, store as Parquet
python -m src.processing.storage --input data/raw --output data/processed

# 3. Generate signals
python -m src.analysis.signal_generator --input data/processed --output data/output

# 4. Plot (--processed points at the same dir used in step 2, for the engagement-distribution sample)
python -m src.visualization.streaming_plots --input data/output --processed data/processed

# Or run everything:
bash scripts/run_pipeline.sh
```

Each step writes a machine-readable run summary to `logs/run_<phase>_<timestamp>.json`
and appends a structured JSON log line per event to `logs/<module>.log`.

### Run tests

```bash
pytest tests/ -v --cov=src
```

### Validate a phase

Reads the latest run-summary JSON for a phase and checks it against the mechanically
verifiable items in `VALIDATION_CHECKLIST.md` (record-count reconciliation, non-zero
output, etc.) — judgment calls like "does the sentiment look sane" are out of scope here
and covered by the manual review process in `PROMPTS.md` instead.

```bash
python scripts/validate_phase.py --phase scraper      # or: processing, signals
```

## Key design decisions (full detail in `ARCHITECTURE.md` and `docs/approach.md`)

- **No paid APIs**: x.com's own search UI redirects unauthenticated requests to a login
  wall (confirmed by testing it directly), so collection targets Nitter — an open-source,
  login-free HTML front-end for the same public X.com content — via Selenium, with
  automatic failover across multiple configured hosts, randomized wait intervals,
  cursor-based pagination, randomized user-agent/viewport, and exponential backoff on
  rate-limit/stall signals.
- **Storage**: Parquet (columnar, compressed, fast for analytical reads), partitioned by
  `date=YYYY-MM-DD/hashtag=X`, with a dedup layer keyed on tweet ID + content hash, and
  idempotent re-runs (re-running processing never duplicates already-written output).
- **Signals**: TF-IDF as the required text-to-vector deliverable, combined with
  lexicon-based sentiment (tuned for Indian market slang), engagement-weighted virality,
  and hashtag co-occurrence momentum into a composite score with a bootstrapped
  confidence interval that widens for low-volume buckets rather than overstating
  confidence on thin data.
- **Memory-efficient plotting**: reservoir sampling / time-bucket aggregation before
  plotting, so chart generation cost is bounded by sample size, not dataset size —
  verified with `tracemalloc`, not just assumed.
- **Concurrency**: a `ProcessPoolExecutor` worker pool scrapes hashtags concurrently, safe
  by construction since each hashtag already owns its own browser session, rate limiter,
  and output file.
- **Scalability**: chunked/streaming Parquet writes today, with a storage layer designed
  to map cleanly onto PostgreSQL/TimescaleDB for time series and Redis for hot-path
  caching at 10x+ scale — see `ARCHITECTURE.md` §6 for the full scaling table.

## Sample output

`docs/sample_output/` holds a real sample of raw tweets, the cleaned/processed dataset
schema, a sample signal output file, and a rendered plot, so the pipeline's downstream
stages can be verified even if a live scraping session is unavailable during review.

## Constraints and honesty notes

- X.com's terms of service restrict automated scraping. This project is built strictly
  for assignment evaluation purposes, uses only publicly viewable data via a third-party
  open-source mirror, respects rate limits, and does not attempt to bypass authentication
  or paywalls.
- Nitter instances are unofficial, third-party, and their availability/index completeness
  fluctuates outside this project's control — the scraper fails over across multiple
  configured hosts, but if all are unavailable during review, `docs/sample_output/` lets
  the downstream pipeline (processing → analysis → visualization) still be run and
  verified end to end.
- `docs/approach.md` documents every known limitation and bug found during development in
  detail, including a collection run that landed short of the assignment's 2,000-tweet
  target because of a genuine, verified ceiling on currently-available live data — not
  from a bug or an early stop.
