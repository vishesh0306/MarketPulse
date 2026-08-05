# MarketPulse

MarketPulse collects Indian stock-market discussion from Twitter/X, cleans and stores it, and converts the text into quantitative trading signals with confidence intervals.

## What it does

1. Scrapes tweets for `#nifty50`, `#sensex`, `#intraday`, and `#banknifty` with Selenium against [Nitter](https://github.com/zedeus/nitter), a login-free mirror of X.com's public content — no paid or official Twitter API.
2. Cleans, deduplicates, and stores the tweets as partitioned Parquet files.
3. Converts tweet text into a composite trading signal per hashtag (TF-IDF, sentiment, engagement, hashtag momentum), filtered to NSE trading hours, with a bootstrapped confidence interval per time bucket.
4. Plots volume, signal, and engagement trends.

Collection targets 2,000 tweets in 24 hours; actual counts depend on live availability of the (unofficial, rate-limited) Nitter mirrors at run time.

## Setup

```bash
git clone <repo-url>
cd MarketPulse
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.11+ and Google Chrome (ChromeDriver is managed automatically).

## Run

```bash
# full pipeline
bash scripts/run_pipeline.sh

# or step by step
python -m src.scraper.twitter_scraper --hashtags nifty50,sensex,intraday,banknifty --hours 24 --min-tweets 2000
python -m src.processing.storage --input data/raw --output data/processed
python -m src.analysis.signal_generator --input data/processed --output data/output
python -m src.visualization.streaming_plots --input data/output --processed data/processed
```

## Run with Docker

```bash
docker build -t marketpulse .
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/logs:/app/logs" marketpulse
```

## Tests

```bash
pytest tests/ -v --cov=src
```

## Output

- `data/raw/` — scraped tweets (JSON Lines)
- `data/processed/` — cleaned, deduplicated Parquet files, partitioned by date and hashtag
- `data/output/signals/` — composite trading signal per hashtag per time bucket, with confidence intervals
- `data/output/plots/` — volume, signal, and engagement charts
- `logs/` — a structured log and run summary per pipeline stage
- `docs/sample_output/` — a sample of each stage's output, for reference without running the scraper

## Project structure

```
src/
├── scraper/          tweet collection (Selenium)
├── processing/        cleaning, deduplication, Parquet storage
├── analysis/          TF-IDF, signal generation
├── visualization/     plotting
└── utils/             config and logging
```
