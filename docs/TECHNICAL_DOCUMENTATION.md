# Technical Documentation

## Approach

MarketPulse is a four-stage pipeline: collect, process, analyze, visualize.

### Collection

Tweets are collected with Selenium against Nitter, an open-source front-end for X.com that serves the same public content over plain HTML without requiring a login. Multiple Nitter hosts are configured with automatic failover, since they're unofficial mirrors and availability varies.

Collection targets `#nifty50`, `#sensex`, `#intraday`, and `#banknifty`, scraped concurrently through a process pool — one browser session and rate limiter per hashtag. A token-bucket rate limiter with exponential backoff handles rate limiting, and a soft-block detector recognizes anti-bot challenge pages and retries instead of failing outright.

Each tweet is recorded with username, timestamp, text, engagement counts, mentions, and hashtags, streamed to disk as JSON Lines.

### Processing & Storage

Raw tweets are cleaned (Unicode normalization, URL stripping), validated against a typed schema, and deduplicated on both exact tweet ID and a content hash of normalized text + author. Invalid records are quarantined with a reason rather than dropped.

Processed data is written as Parquet, partitioned by date and hashtag, using chunked writes to keep memory usage independent of dataset size.

### Analysis & Signal Generation

Tweet text is vectorized with TF-IDF. The trading signal itself is built from three interpretable features rather than raw TF-IDF weights: lexicon-based sentiment (tuned for Indian market slang such as "short buildup" and "long buildup"), engagement-weighted virality, and hashtag co-occurrence momentum. These combine into a weighted composite score per 15-minute bucket, with a bootstrapped confidence interval that widens for buckets with fewer tweets.

### Visualization

Plots are built from pre-aggregated bucket data and a bounded reservoir sample of raw tweets, so memory usage stays constant regardless of dataset size.

## Key Design Decisions

- **Parquet, partitioned by date and hashtag** — compressed, columnar storage suited to analytical reads and incremental processing.
- **TF-IDF plus engineered features, not embeddings** — a fast, interpretable signal that runs on CPU with no external model.
- **One browser session per hashtag, run concurrently** — avoids shared state between scraper workers.
- **Idempotent processing** — re-running the pipeline against the same input never duplicates output.

## Scalability

The four-stage design holds at higher volume; only the backend under each stage would change. Parquet partitioning supports incremental reads with a distributed engine (Polars/Spark) instead of a full re-read each run. TF-IDF fitting can move to a mini-batch `HashingVectorizer`. Dedup lookups can move from an in-process set to Redis. The scraper's worker pool can move to a queue-backed system (Celery/RQ) spanning multiple machines. Signals can be served over a REST/WebSocket API (FastAPI) instead of flat files.

## Tech Stack

Python, Selenium, pandas, PyArrow, scikit-learn, matplotlib, pydantic, joblib.
