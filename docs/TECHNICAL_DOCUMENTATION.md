# Technical Documentation

## Approach

MarketPulse is a four-stage pipeline: collect, process, analyze, visualize.

### Collection

Tweets are collected with Selenium against Nitter, an open-source front-end for X.com that serves the same public content over plain HTML without requiring a login — the constraint-driven choice given the assignment rules out paid/official APIs. Multiple Nitter hosts are configured with automatic failover, since they're unofficial mirrors and availability varies; a hashtag keeps trying configured hosts in order until it reaches its target or runs out of hosts, not just until one host stops being rate-limited.

Collection targets `#nifty50`, `#sensex`, `#intraday`, and `#banknifty`, scraped concurrently through a process pool — one browser session and rate limiter per hashtag. A token-bucket rate limiter with exponential backoff handles rate limiting, and a soft-block detector recognizes anti-bot challenge pages and retries instead of failing outright.

Each tweet is recorded with username, timestamp, text, engagement counts, mentions, and hashtags, streamed to disk as JSON Lines.

### Processing & Storage

Raw tweets are cleaned (Unicode normalization, URL stripping), validated against a typed schema, and deduplicated on both exact `(tweet_id, source_hashtag)` and a content hash of normalized text + author + source hashtag. Keying on the hashtag as well as the ID matters here: a tweet mentioning more than one target hashtag is legitimately collected once per hashtag search, and storage partitions by hashtag, so each of those copies belongs in its own partition rather than being deduplicated away. Invalid records are quarantined with a reason rather than dropped.

Processed data is written as Parquet, partitioned by date and hashtag, using chunked writes to keep memory usage independent of dataset size.

### Analysis & Signal Generation

Tweet text is vectorized with TF-IDF. The trading signal itself is built from three interpretable features rather than raw TF-IDF weights: lexicon-based sentiment (tuned for Indian market slang such as "short buildup" and "long buildup", with a negation check so "not a buy at these levels" doesn't score as bullish), engagement-weighted virality, and co-occurrence momentum restricted to the four target hashtags (so a tweet stuffed with unrelated tags doesn't read as elevated momentum).

Sentiment carries the sign of the composite; virality and momentum are unsigned `[0,1]` confidence multipliers on top of it, not additive terms — a neutral tweet stays at a signal of 0 no matter how viral it is, and a bullish and bearish tweet with identical engagement produce equal-magnitude, opposite-sign signals.

Signals are bucketed in 15-minute windows, filtered to the NSE trading session (09:15–15:30 IST, Monday–Friday) by default — chatter outside market hours is real data and stays in `data/processed`, it's just excluded from the *trading* signal, where thin off-hours buckets would otherwise dominate the confidence-interval spread. Each bucket gets a bootstrapped confidence interval, combined with a standard-error estimate that only floors at the dataset-wide variance when a bucket has fewer than two tweets to estimate its own — a well-sampled bucket keeps its real, tighter local variance. Hourly and daily rollups are written alongside the 15-minute signal.

### Visualization

Plots are built from pre-aggregated bucket data and a bounded reservoir sample of raw tweets, so memory usage stays constant regardless of dataset size.

## Key Design Decisions

- **Parquet, partitioned by date and hashtag** — compressed, columnar storage suited to analytical reads and incremental processing.
- **TF-IDF plus engineered features, not embeddings** — a fast, interpretable signal that runs on CPU with no external model.
- **Multiplicative, not additive, signal composition** — sentiment sets the sign, virality and momentum scale its magnitude, so the signal can't be pushed bullish purely by engagement on a sentiment-neutral tweet.
- **Market-hours filtering** — signal buckets are generated only from tweets inside the NSE trading session, since bucketing overnight chatter into a trading signal mostly measures conversation volume, not market sentiment, and dilutes the confidence-interval math.
- **One browser session per hashtag, run concurrently** — avoids shared state between scraper workers.
- **Idempotent processing** — re-running the pipeline against the same input never duplicates output.

## Scalability

The four-stage design holds at higher volume; only the backend under each stage would change. Parquet partitioning supports incremental reads with a distributed engine (Polars/Spark) instead of a full re-read each run. TF-IDF fitting can move to a mini-batch `HashingVectorizer`. Dedup lookups can move from an in-process set to Redis. The scraper's worker pool can move to a queue-backed system (Celery/RQ) spanning multiple machines. Signals can be served over a REST/WebSocket API (FastAPI) instead of flat files.

## Tech Stack

Python, Selenium, pandas, PyArrow, scikit-learn, matplotlib, pydantic, joblib.
