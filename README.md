# MarketPulse

MarketPulse collects Indian stock-market discussion from Twitter/X, cleans and stores it, and converts the text into quantitative trading signals with confidence intervals.

## What it does

1. Collects tweets for `#nifty50`, `#sensex`, `#intraday`, `#banknifty` — plus similar Indian-market hashtags (`#nifty`, `#niftybank`, `#giftnifty`, `#stockmarket`, `#sharemarket`, `#nse`) — straight from x.com's own web endpoints, authenticated as an ordinary logged-in browser session. No paid API, no developer API, no API key.
2. Cleans, deduplicates, and stores the tweets as partitioned Parquet files.
3. Converts tweet text into numerical vectors (TF-IDF over the corpus) and, separately, into a composite trading signal per hashtag built from engineered features — lexicon sentiment (English + Romanised Hindi), engagement virality, hashtag momentum — filtered to NSE trading hours. Each 15-minute bucket carries an interval (a bootstrap resample widened to a normal-approximation floor, whichever is wider); buckets with fewer than `min_bucket_tweets` tweets are reported as volume only, signal suppressed, and each carries a `sentiment_coverage` score.
4. Plots volume, signal, and engagement trends.

A **real-time service** (`python -m src.realtime`) tails the same hashtags, maintains the signal incrementally, and serves it over HTTP/WebSocket — see [Real-time service](#real-time-service).

The core four hashtags and the similar-hashtag list are both in `config/settings.yaml` (`scraper.hashtags` / `scraper.related_hashtags`).

---

## Quick start

Five steps from a clone to charts. Needs Python 3.11 or 3.12 and one throwaway X account.

### 1. Install

```bash
git clone <repo-url>
cd MarketPulse
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get your X cookies

Collection reads x.com as a logged-in user, so it needs one account's session cookies. **Use a throwaway account, not your main one.**

1. Log in to <https://x.com> in your browser
2. Press **F12** to open DevTools
3. Go to **Application** → **Cookies** → `https://x.com`
4. Find and copy the values of **`auth_token`** and **`ct0`**

> These are not an API key and cost nothing — they're the ordinary session cookies your browser already has. Treat them like a password.

### 3. Save them

```bash
cp .env.example .env
```

Open `.env` and paste your two values:

```
X_AUTH_TOKEN=paste_auth_token_here
X_CT0=paste_ct0_here
```

`.env` is gitignored, so it will never be committed.

### 4. Collect tweets

```bash
python -m src.scraper.x_collector --hours 24 --min-tweets 2000
```

Expect this to take **30–45 minutes**. x.com allows ~1,000 tweets per 15 minutes per account, so the run pauses and resumes automatically. Leave it running. Tweets land in `data/raw/` as they arrive.

### 5. Analyse and plot

```bash
python -m src.processing.storage       --input data/raw       --output data/processed
python -m src.analysis.signal_generator --input data/processed --output data/output
python -m src.visualization.streaming_plots --input data/output --processed data/processed
```

> `data/raw/` keeps every collection run, which is what makes reprocessing idempotent — but
> tweets that were inside the window when collected drift outside it as time passes. The
> processing stage therefore re-applies the lookback (`--lookback-hours`, default: the
> collector's `hours_lookback`), so the analysed dataset is always exactly the last 24
> hours no matter how many runs have accumulated. Pass `--lookback-hours 0` to keep
> everything.

**Your results:**

| Where | What |
|---|---|
| `data/output/plots/` | three PNG charts — volume, signal with confidence intervals, engagement |
| `data/output/signals/` | the trading signal per hashtag per 15-minute bucket (Parquet) |
| `data/processed/` | the cleaned, deduplicated tweets (Parquet) |
| `logs/` | a run summary per stage — counts, timings, peak memory |

### Faster, or a different time window

- **Halve the wait** — add a second throwaway account's cookies; twscrape rotates between them automatically.
- **Aim at market hours** — see [Targeting the trading session](#targeting-the-trading-session) below.
- **Live signal over WebSocket** — see [Real-time service](#real-time-service) below.

---

## Collection: how, and why it changed

The original collector drove Selenium against [Nitter](https://github.com/zedeus/nitter), a login-free mirror of X's public content. That stopped working. X removed anonymous access in 2024, so Nitter instances now need real X account tokens, X bans those tokens quickly, and the public mirrors answer `403`/`410`/dead DNS. A run against the seven configured hosts collects **zero** tweets.

The working path is `src/scraper/x_collector.py`, built on [twscrape](https://github.com/vladkens/twscrape). It authenticates with the `auth_token`/`ct0` cookies of a logged-in browser session and calls the same JSON endpoints x.com's own front-end calls. Nothing is purchased and no developer API is involved — it is a logged-in session reading search results. Practical benefits over HTML scraping: no parsing of a virtualised React timeline, and engagement counts arrive as real numbers.

The Selenium/Nitter path is kept in `src/scraper/twitter_scraper.py` as a documented fallback should a usable mirror reappear. Both write identical JSONL, so everything downstream is unchanged.

### Getting credentials

Use a throwaway X account. Log in at x.com, then **DevTools (F12) → Application → Cookies → `https://x.com`** and copy `auth_token` and `ct0` into a `.env` (gitignored — see `.env.example`):

```bash
X_AUTH_TOKEN=...
X_CT0=...
```

Treat them like a password. They last weeks; logging out invalidates them.

### Rate limits

x.com allows ~50 search requests per 15 minutes per account — roughly **1,000 tweets per window**. twscrape tracks this per account and waits for the reset automatically, so a 2,000-tweet run completes unattended in ~30–45 minutes on one account. Registering more accounts multiplies throughput and twscrape rotates between them:

```python
await api.pool.add_account_cookies("acct2", "auth_token=...; ct0=...")
```

## Setup

```bash
git clone <repo-url>
cd MarketPulse
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in X_AUTH_TOKEN / X_CT0
```

Requires Python 3.11–3.12 (the pinned dependencies don't yet build on 3.13+). The Selenium fallback additionally needs Google Chrome (ChromeDriver is managed automatically); the real-time service needs Redis — `docker compose up realtime` brings one up.

## Run

```bash
# 1. collect (x.com, last 24h, stops once the run-wide target is met)
python -m src.scraper.x_collector --hours 24 --min-tweets 2000

# 2-4. process, analyse, plot
python -m src.processing.storage --input data/raw --output data/processed
python -m src.analysis.signal_generator --input data/processed --output data/output
python -m src.visualization.streaming_plots --input data/output --processed data/processed
```

A run that finishes below `--min-tweets` exits non-zero, so the pipeline stops rather than carrying a thin corpus into the analysis. `--allow-shortfall` opts out.

### Targeting the trading session

x.com's Latest search pages backwards from *now*. Collecting in the evening therefore spends the rate-limit budget on post-close chatter and may never reach the trading session — which is exactly the data the market-hours-filtered signal needs. `--offset-hours` ends the window earlier so a run can aim at the NSE session directly:

```bash
# the 6.3h session, when it ended ~8.4h ago
python -m src.scraper.x_collector --hours 6.3 --offset-hours 8.4 --min-tweets 1500
```

On a real run this took `#nifty` from 15 in-session tweets to 983.

### Nitter fallback

```bash
bash scripts/run_pipeline.sh      # Selenium/Nitter path — currently collects nothing
```

## Real-time service

The batch collector is a one-shot 24h run. The real-time service picks up from there: it warm-starts from `data/processed`, then polls each hashtag's Nitter **RSS** feed every ~20s, dedupes against **Redis**, and keeps a per-bucket signal updated incrementally (Welford, `O(1)` per tweet). Buckets seal to a JSONL history and to WebSocket subscribers once their window closes.

The tail loop currently reads RSS, so it is subject to the same Nitter outage described above; the incremental-signal, Redis-dedup, queue and API layers are source-agnostic and work against any collector that yields the standard record shape.

```bash
docker compose up realtime          # starts redis + the service on :8000

# or locally, with a Redis reachable at $REDIS_URL (default redis://localhost:6379/0)
python -m src.realtime --hashtags nifty50,sensex --port 8000
```

| Endpoint | |
|---|---|
| `GET /signals[?live=true]` | sealed bucket history, optionally plus live un-sealed buckets |
| `WS /ws/signals` | history on connect, then each bucket as it seals |
| `GET /metrics` | ingest lag p50/p95, tweets/min per hashtag, dedup hit rate, per-host poll success, per-hashtag staleness |
| `GET /health` | liveness |

Config lives under `realtime:` in `config/settings.yaml` (poll interval, queue size and full-queue policy, seal grace period, dedup TTL).

## Run with Docker

```bash
docker compose run --rm marketpulse    # batch pipeline (Nitter path)
docker compose up realtime             # real-time service + redis
```

## Tests

```bash
pytest tests/ -v --cov=src
```

## Output

- `data/raw/` — collected tweets (JSON Lines)
- `data/processed/` — cleaned, deduplicated Parquet, partitioned by date and hashtag
- `data/output/signals/` — composite signal per hashtag per bucket with confidence intervals; hourly/daily rollups alongside
- `data/output/plots/` — volume, signal, and engagement charts
- `data/output/realtime/` — the real-time service's sealed-bucket history (JSON Lines)
- `logs/` — a structured log and run summary per stage
- `docs/sample_output/` — a sample of each stage's output from a real run, for reference without collecting

## Project structure

```
src/
├── scraper/           x_collector (x.com, primary) + twitter_scraper (Nitter, fallback)
├── processing/        cleaning, deduplication, Parquet storage
├── analysis/          TF-IDF, feature engineering, signal generation, rollups
├── visualization/     memory-bounded plotting
├── realtime/          tail loop, incremental signal, Redis dedup, HTTP/WS API
└── utils/             config and logging
```
