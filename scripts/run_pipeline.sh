#!/usr/bin/env bash
# Runs the full MarketPulse pipeline end to end: collect -> process -> analyze -> plot.
set -euo pipefail

HASHTAGS="${HASHTAGS:-nifty50,sensex,intraday,banknifty}"
HOURS="${HOURS:-24}"
MIN_TWEETS="${MIN_TWEETS:-2000}"
WORKERS="${WORKERS:-4}"

echo "[1/4] Collecting tweets..."
python -m src.scraper.twitter_scraper --hashtags "$HASHTAGS" --hours "$HOURS" --min-tweets "$MIN_TWEETS" --workers "$WORKERS"

echo "[2/4] Processing & storing..."
python -m src.processing.storage --input data/raw --output data/processed

echo "[3/4] Generating signals..."
python -m src.analysis.signal_generator --input data/processed --output data/output

echo "[4/4] Plotting..."
python -m src.visualization.streaming_plots --input data/output --processed data/processed

echo "Pipeline complete."
