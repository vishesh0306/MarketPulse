"""Real-time collection and signalling: a Nitter-RSS tail loop feeding an incremental
per-bucket signal, served over HTTP/WebSocket. Sits alongside the batch pipeline in
src/ rather than replacing it — the batch scraper still does the initial 24h backfill.
"""
