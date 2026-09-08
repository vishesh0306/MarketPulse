#!/usr/bin/env bash
# Runs the full MarketPulse pipeline end to end: collect -> process -> analyze -> plot.
set -euo pipefail

# Resolve an interpreter directly rather than relying on inherited shell state — a venv
# activated in a parent PowerShell session doesn't propagate its PATH into a bash
# subprocess. Prefer the project's own venv when one exists (local development); fall back
# to whatever python/python3 is already on PATH otherwise (e.g. inside Docker, where
# dependencies are installed into the container's system Python and no venv exists).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]; then
    PYTHON="$REPO_ROOT/.venv/Scripts/python.exe"        # Windows venv layout
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"                # POSIX venv layout
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"                                     # no venv — e.g. inside Docker
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "No Python interpreter found (no .venv, no python3/python on PATH). Run 'python -m venv .venv && pip install -r requirements.txt' first." >&2
    exit 1
fi
cd "$REPO_ROOT"

HASHTAGS="${HASHTAGS:-nifty50,sensex,intraday,banknifty}"
HOURS="${HOURS:-24}"
MIN_TWEETS="${MIN_TWEETS:-2000}"
WORKERS="${WORKERS:-4}"
# Set VALIDATE=0 to skip the per-stage exit-criteria checks (exploratory runs).
VALIDATE="${VALIDATE:-1}"

# Runs scripts/validate_phase.py against a stage's run-summary JSON and aborts the
# pipeline if a mechanical check fails — a shortfall, a broken record-count invariant,
# an empty output. set -e already stops on the exit code; this makes the reason visible.
validate() {
    [ "$VALIDATE" = "1" ] || return 0
    echo "    validating $1 output..."
    "$PYTHON" scripts/validate_phase.py --phase "$1"
}

echo "[1/4] Collecting tweets..."
"$PYTHON" -m src.scraper.twitter_scraper --hashtags "$HASHTAGS" --hours "$HOURS" --min-tweets "$MIN_TWEETS" --workers "$WORKERS"
validate scraper

echo "[2/4] Processing & storing..."
"$PYTHON" -m src.processing.storage --input data/raw --output data/processed
validate processing

echo "[3/4] Generating signals..."
"$PYTHON" -m src.analysis.signal_generator --input data/processed --output data/output
validate signals

echo "[4/4] Plotting..."
"$PYTHON" -m src.visualization.streaming_plots --input data/output --processed data/processed

echo "Pipeline complete."
