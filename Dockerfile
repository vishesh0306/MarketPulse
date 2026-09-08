FROM python:3.11-slim

# Chrome + deps for headless Selenium scraping (chromedriver itself is managed at
# runtime by webdriver-manager, so it isn't pinned here).
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        gnupg \
        ca-certificates \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# sync + a short pause before exit: Docker Desktop's bind-mount write-back cache can still
# have pending writes when the container's main process exits, and --rm tears the
# container down immediately after — without this, output written late in the run
# (parquet files, plots, the run-summary logs) can be lost from the host side.
# Capture the pipeline's exit code first and re-raise it after the flush, so a failed
# run still flushes its diagnostic output but the container exits non-zero.
CMD ["bash", "-c", "bash scripts/run_pipeline.sh; rc=$?; sync; sleep 8; exit $rc"]
