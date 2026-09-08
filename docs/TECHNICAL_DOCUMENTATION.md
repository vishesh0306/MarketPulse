# Technical Documentation

## Approach

MarketPulse has two modes over one shared analysis core: a **batch pipeline** (collect → process → analyze → visualize) and a **real-time service** that tails new tweets and keeps an incremental signal live over HTTP/WebSocket.

### Collection

Tweets are collected with Selenium against Nitter, an open-source front-end for X.com that serves the same public content over plain HTML without requiring a login — the constraint-driven choice given the assignment rules out paid/official APIs. A list of Nitter hosts is configured with automatic failover, since they're unofficial mirrors and availability varies; a hashtag keeps trying configured hosts in order until it reaches its target or runs out of hosts, not just until one host stops being rate-limited. The user agent is re-rolled per host so one worker doesn't present the same fingerprint to the whole list.

Collection covers the assignment's four hashtags — `#nifty50`, `#sensex`, `#intraday`, `#banknifty` — plus a short list of similar Indian-market tags (`#nifty`, `#niftybank`, `#giftnifty`, `#stockmarket`, `#sharemarket`, `#nse`) to widen coverage toward the 2,000-tweet target given how thin the public mirrors are. Hashtags are scraped concurrently through a process pool — one browser session and rate limiter per hashtag — and share a single run-wide collected counter, so a dense hashtag keeps collecting to cover one that runs dry rather than both stopping at a fixed `min_tweets / N` slice. A token-bucket rate limiter with exponential backoff handles rate limiting, and a soft-block detector recognizes anti-bot challenge pages and retries instead of failing outright. A run that finishes below `--min-tweets` exits non-zero (unless `--allow-shortfall` is passed), so `run_pipeline.sh`, `validate_phase.py` and the container all treat a shortfall as a failure.

Each tweet is recorded with username, timestamp, text, engagement counts, mentions, and hashtags, streamed to disk as JSON Lines.

### Processing & Storage

Raw tweets are cleaned (Unicode NFC normalization, URL stripping), validated against a typed schema, and deduplicated on both exact `(tweet_id, source_hashtag)` and a content hash of normalized text + author + source hashtag. Keying on the hashtag as well as the ID matters here: a tweet mentioning more than one target hashtag is legitimately collected once per hashtag search, and storage partitions by hashtag, so each of those copies belongs in its own partition rather than being deduplicated away. Invalid records are quarantined with a reason rather than dropped, and one truncated JSONL line quarantines that line instead of aborting the run.

Processed data is written as Parquet, partitioned by date and hashtag. Writes are chunked so the *write* path never holds the whole dataset; the dedup structures are in-memory sets, so processing memory is `O(unique records)` for a run, not constant. Output is written to a temp directory and swapped in atomically, so a re-run reproduces the same row count and a mid-run crash leaves the previous output intact.

### Analysis & Signal Generation

Tweet text is vectorized with TF-IDF (the "text to numerical vectors" deliverable); the trading signal itself is built from interpretable engineered features rather than raw TF-IDF weights. Those features are: lexicon-based sentiment (English plus Romanised-Hindi terms such as "tezi", "mandi", "girega", with a negation check so "not a buy at these levels" doesn't score bullish), engagement-weighted virality, and co-occurrence momentum restricted to the tracked hashtags (so a tweet stuffed with unrelated tags doesn't read as elevated momentum).

Sentiment carries the sign of the composite; virality and momentum are unsigned `[0,1]` multipliers on top of it, not additive terms — a neutral tweet stays at 0 no matter how viral, and a bullish and bearish tweet with identical engagement produce equal-magnitude, opposite-sign signals.

Signals are bucketed in 15-minute windows, filtered to the NSE session (09:15–15:30 IST, Mon–Fri) — off-hours chatter stays in `data/processed`, it's just excluded from the *trading* signal. Each bucket gets an interval: a bootstrap resample of the bucket's per-tweet contributions, widened to a normal-approximation floor (`z·σ/√n`), whichever is wider — the floor uses the bucket's own variance when it has one, and the dataset-wide variance only when it has fewer than two points. Buckets below `min_bucket_tweets` (default 5) are **suppressed**: their volume is still reported but the signal and interval are null, because an interval built from two or three tweets is dominated by the fallback variance and all but guaranteed to span zero. Every bucket also carries `sentiment_coverage` — the fraction of its tweets the lexicon matched — so a near-zero signal from a genuinely balanced bucket is distinguishable from one the (mostly English) lexicon couldn't read. Hourly and daily rollups are written alongside; they roll volume over every fine bucket but weight the signal over only the scored ones.

### Real-time service

`python -m src.realtime` (or `docker compose up realtime`) runs a FastAPI app on `:8000`. On startup it warm-starts from the last 24h of `data/processed` — whatever the Selenium backfill left — then tails each hashtag's Nitter **RSS** feed (`/search/rss`: one GET, no browser, no JS challenge) every ~20s, guarded by a per-hashtag high-water tweet-id mark and a **Redis** dedup store (each id and content hash a TTL'd key, so state is bounded and survives a restart). New tweets cross a bounded `asyncio.Queue` — with an explicit block-vs-drop-oldest policy — into the incremental signal engine, which keeps an `O(1)`-per-tweet Welford accumulator per `(hashtag, 15-min bucket)` plus a bounded reservoir for a seal-time bootstrap. When a bucket's window plus a grace period has passed it is sealed once, appended to a JSONL history, and pushed to WebSocket subscribers.

- `GET /signals[?live=true]` — sealed history, optionally plus the live un-sealed buckets
- `WS /ws/signals` — the history on connect, then each bucket as it seals
- `GET /metrics` — ingest lag p50/p95, tweets/minute per hashtag, dedup hit rate, per-host poll success, per-hashtag signal staleness

The RSS feed carries no engagement counts, so streamed tweets have virality 0 — for a tweet seconds old that is also the true value, and Selenium stays the tool for the deep backfill where cursor paging is genuinely needed.

### Visualization

Plots are built from pre-aggregated bucket data and a bounded reservoir sample of raw tweets, so plotting memory is a function of the sample size, not the dataset. Suppressed buckets render as gaps rather than lines drawn through one or two tweets.

## Key Design Decisions

- **Parquet, partitioned by date and hashtag** — compressed columnar storage suited to analytical reads and incremental processing.
- **TF-IDF plus engineered features, not embeddings** — a fast, interpretable signal that runs on CPU with no external model.
- **Multiplicative, not additive, signal composition** — sentiment sets the sign, virality and momentum scale its magnitude.
- **Suppress rather than publish thin buckets** — a suppressed bucket is more useful to a consumer than an interval that means nothing.
- **Shared collection target** — the run chases 2,000 tweets total, not a fixed slice per hashtag.
- **RSS for the tail, Selenium for the backfill** — an order of magnitude cheaper per tweet for a frequent poll; Selenium only where paging is needed.
- **Redis for real-time dedup** — bounded by TTL, survives restarts; the in-process set stays fine for a batch run that exits.
- **Fail loudly on a shortfall** — the 2,000-tweet minimum is enforced at the scraper, the pipeline script and the container.

## Scalability

The stage design holds at higher volume; the backend under each stage would change. Parquet partitioning supports incremental reads with a distributed engine (Polars/Spark) instead of a full re-read. TF-IDF fitting can move to a mini-batch `HashingVectorizer`. The scraper's worker pool can move to a queue-backed system (Celery/RQ) spanning machines. The real-time dedup store, incremental signal and WebSocket API are already in place; the next steps are sharding the tail loop by hashtag across processes and moving the sealed-bucket history from JSONL to the partitioned Parquet store.

## Tech Stack

Python, Selenium, pandas, PyArrow, scikit-learn, matplotlib, pydantic, joblib, httpx, Redis, FastAPI, uvicorn.
