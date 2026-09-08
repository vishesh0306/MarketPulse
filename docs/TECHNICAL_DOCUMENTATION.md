# Technical Documentation

## Approach

MarketPulse has two modes over one shared analysis core: a **batch pipeline** (collect → process → analyze → visualize) and a **real-time service** that tails new tweets and keeps an incremental signal live over HTTP/WebSocket.

### Collection

The original design scraped Nitter — an open-source, login-free front-end for X's public content — with Selenium, which was the natural constraint-driven choice given the assignment rules out paid and official APIs. That approach no longer works. X removed anonymous access in 2024, so Nitter instances now require real X account tokens; X bans those tokens quickly, and the public mirrors answer `403`, `410`, or dead DNS. A run against the seven configured hosts collects zero tweets.

**The working collector reads x.com directly.** `x_collector.py` authenticates with the `auth_token`/`ct0` cookies of an ordinary logged-in browser session and calls the same JSON endpoints x.com's own front-end calls. Nothing is purchased and the developer API is not involved — this is a logged-in session reading its own search results, which is what "scrape Twitter/X, no API, consider Selenium" points at once the site is login-walled. Two practical wins over HTML scraping: no parsing of a virtualised React timeline, and engagement counts arrive as integers rather than being absent.

The Selenium/Nitter path is retained in `twitter_scraper.py` as a documented fallback. Both collectors emit identical JSONL, so every downstream stage is agnostic to which produced the data.

Collection covers the assignment's four hashtags — `#nifty50`, `#sensex`, `#intraday`, `#banknifty` — plus similar Indian-market tags (`#nifty`, `#niftybank`, `#giftnifty`, `#stockmarket`, `#sharemarket`, `#nse`). Hashtags share a single run-wide counter, so a dense hashtag keeps collecting to cover one that runs dry rather than each stopping at a fixed `min_tweets / N` slice. A run finishing below `--min-tweets` exits non-zero (unless `--allow-shortfall`), so `run_pipeline.sh`, `validate_phase.py` and the container all treat a shortfall as failure.

**Rate limiting.** x.com permits ~50 search requests per 15 minutes per account, about 1,000 tweets per window. twscrape tracks this per account and per endpoint, waits for the reset, and rotates to any other registered account — so a 2,000-tweet run completes unattended across two or three windows, and adding accounts multiplies throughput linearly. The Nitter path has its own defences for a hostile mirror: token-bucket pacing with jitter, exponential backoff, soft-block detection on anti-bot challenge pages, multi-host failover, and a per-host user-agent re-roll.

**Two behaviours of x.com search that the collector has to correct for.** Its `since_time`/`until_time` operators are a hint, not a guarantee — a trial run returned a 2023 tweet — so the window is enforced again client-side and out-of-window drops are counted in the run summary. And Latest-tab results page backwards from *now*, so an evening run spends its entire rate-limit budget on post-close chatter and never reaches the trading session; `--offset-hours` ends the window earlier so a run can aim at the NSE session directly. On a real run that took `#nifty` from 15 in-session tweets to 983.

**Concurrency.** Hashtags are collected sequentially here, deliberately: the rate limit is per *account* and shared across all hashtags, so issuing them in parallel would exhaust the same budget faster with no gain. Concurrency is applied where it does pay — one browser process per hashtag in the Selenium path, `joblib` across buckets in the bootstrap, and an async collector/updater split in the real-time service.

Each tweet is recorded with username, timestamp, text, engagement counts, mentions, and hashtags, streamed to disk as JSON Lines.

### Processing & Storage

Raw tweets are cleaned (Unicode NFC normalization, URL stripping), validated against a typed schema, and deduplicated on both exact `(tweet_id, source_hashtag)` and a content hash of normalized text + author + source hashtag. Keying on the hashtag as well as the ID matters here: a tweet mentioning more than one target hashtag is legitimately collected once per hashtag search, and storage partitions by hashtag, so each of those copies belongs in its own partition rather than being deduplicated away. Invalid records are quarantined with a reason rather than dropped, and one truncated JSONL line quarantines that line instead of aborting the run.

Processed data is written as Parquet, partitioned by date and hashtag. Writes are chunked so the *write* path never holds the whole dataset; the dedup structures are in-memory sets, so processing memory is `O(unique records)` for a run, not constant. Output is written to a temp directory and swapped in atomically, so a re-run reproduces the same row count and a mid-run crash leaves the previous output intact.

### Analysis & Signal Generation

Tweet text is vectorized with TF-IDF (the "text to numerical vectors" deliverable); the trading signal itself is built from interpretable engineered features rather than raw TF-IDF weights. Those features are: lexicon-based sentiment (English plus Romanised-Hindi terms such as "tezi", "mandi", "girega", with a negation check so "not a buy at these levels" doesn't score bullish), engagement-weighted virality, and co-occurrence momentum restricted to the tracked hashtags (so a tweet stuffed with unrelated tags doesn't read as elevated momentum).

Sentiment carries the sign of the composite; virality and momentum are unsigned `[0,1]` multipliers on top of it, not additive terms — a neutral tweet stays at 0 no matter how viral, and a bullish and bearish tweet with identical engagement produce equal-magnitude, opposite-sign signals.

Signals are bucketed in 15-minute windows, filtered to the NSE session (09:15–15:30 IST, Mon–Fri) — off-hours chatter stays in `data/processed`, it's just excluded from the *trading* signal. Each bucket gets an interval: a bootstrap resample of the bucket's per-tweet contributions, widened to a normal-approximation floor (`z·σ/√n`), whichever is wider — the floor uses the bucket's own variance when it has one, and the dataset-wide variance only when it has fewer than two points. Buckets below `min_bucket_tweets` (default 5) are **suppressed**: their volume is still reported but the signal and interval are null, because an interval built from two or three tweets is dominated by the fallback variance and all but guaranteed to span zero. Every bucket also carries `sentiment_coverage` — the fraction of its tweets the lexicon matched — so a near-zero signal from a genuinely balanced bucket is distinguishable from one the (mostly English) lexicon couldn't read. Hourly and daily rollups are written alongside; they roll volume over every fine bucket but weight the signal over only the scored ones.

### Real-time service

`python -m src.realtime` (or `docker compose up realtime`) runs a FastAPI app on `:8000`. On startup it warm-starts from the last 24h of `data/processed` — whatever the batch collector left — then tails each hashtag's Nitter **RSS** feed (`/search/rss`: one GET, no browser, no JS challenge) every ~20s, guarded by a per-hashtag high-water tweet-id mark and a **Redis** dedup store (each id and content hash a TTL'd key, so state is bounded and survives a restart). New tweets cross a bounded `asyncio.Queue` — with an explicit block-vs-drop-oldest policy — into the incremental signal engine, which keeps an `O(1)`-per-tweet Welford accumulator per `(hashtag, 15-min bucket)` plus a bounded reservoir for a seal-time bootstrap. When a bucket's window plus a grace period has passed it is sealed once, appended to a JSONL history, and pushed to WebSocket subscribers.

- `GET /signals[?live=true]` — sealed history, optionally plus the live un-sealed buckets
- `WS /ws/signals` — the history on connect, then each bucket as it seals
- `GET /metrics` — ingest lag p50/p95, tweets/minute per hashtag, dedup hit rate, per-host poll success, per-hashtag signal staleness

The RSS feed carries no engagement counts, so streamed tweets have virality 0 — for a tweet seconds old that is also the true value, and the batch collector stays the tool for the deep backfill where paging is genuinely needed.

The tail loop still reads RSS and is therefore subject to the same Nitter outage described under Collection. The incremental-signal, Redis-dedup, queue and API layers are source-agnostic and run against any collector emitting the standard record shape, so repointing the tail at `x_collector` is a contained change rather than a redesign.

### Visualization

Plots are built from pre-aggregated bucket data and a bounded reservoir sample of raw tweets, so plotting memory is a function of the sample size, not the dataset. Suppressed buckets render as gaps rather than lines drawn through one or two tweets.

## Key Design Decisions

- **Parquet, partitioned by date and hashtag** — compressed columnar storage suited to analytical reads and incremental processing.
- **TF-IDF plus engineered features, not embeddings** — a fast, interpretable signal that runs on CPU with no external model.
- **Multiplicative, not additive, signal composition** — sentiment sets the sign, virality and momentum scale its magnitude.
- **Suppress rather than publish thin buckets** — a suppressed bucket is more useful to a consumer than an interval that means nothing.
- **Shared collection target** — the run chases 2,000 tweets total, not a fixed slice per hashtag.
- **Cookie-authenticated x.com over a public mirror** — forced by Nitter's collapse, but strictly better: no HTML parsing, real engagement counts, and a rate limit that is documented and waited out rather than guessed at.
- **Enforce the time window client-side** — x.com's own `since_time` operators leak older tweets, so the collector re-checks every record rather than trusting the query.
- **RSS for the tail, browser paging for the backfill** — an order of magnitude cheaper per tweet for a frequent poll; heavier paging only where depth is needed.
- **Redis for real-time dedup** — bounded by TTL, survives restarts; the in-process set stays fine for a batch run that exits.
- **Fail loudly on a shortfall** — the 2,000-tweet minimum is enforced at the scraper, the pipeline script and the container.

## Scalability

The stage design holds at higher volume; the backend under each stage would change. Parquet partitioning supports incremental reads with a distributed engine (Polars/Spark) instead of a full re-read. TF-IDF fitting can move to a mini-batch `HashingVectorizer`. The scraper's worker pool can move to a queue-backed system (Celery/RQ) spanning machines. The real-time dedup store, incremental signal and WebSocket API are already in place; the next steps are sharding the tail loop by hashtag across processes and moving the sealed-bucket history from JSONL to the partitioned Parquet store.

## Tech Stack

Python, twscrape, Selenium, pandas, PyArrow, scikit-learn, matplotlib, pydantic, joblib, httpx, Redis, FastAPI, uvicorn.
