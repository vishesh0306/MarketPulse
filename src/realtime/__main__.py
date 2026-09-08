"""Entrypoint for the real-time service: `python -m src.realtime`.

Wires a Redis connection, the dedup store, the incremental signal engine and the tail
supervisor into the FastAPI app, then serves it with uvicorn. The tail loop itself is
started and stopped by the app's lifespan.
"""

from __future__ import annotations

import argparse
import os

import uvicorn
from redis.asyncio import Redis

from src.realtime.api import create_app
from src.realtime.dedup_store import RedisDedupStore
from src.realtime.supervisor import TailSupervisor, build_engine
from src.utils.config_loader import load_settings
from src.utils.logger import get_logger, set_level

logger = get_logger("realtime_main")


def main() -> None:
    settings = load_settings()
    set_level(logger, settings.logging.level)
    rt = settings.realtime

    parser = argparse.ArgumentParser(description="MarketPulse real-time RSS tail + signal service.")
    parser.add_argument("--hashtags", type=str, default=",".join(settings.scraper.all_hashtags))
    parser.add_argument("--host", type=str, default=rt.api.host)
    parser.add_argument("--port", type=int, default=rt.api.port)
    parser.add_argument("--redis-url", type=str, default=os.environ.get("REDIS_URL", rt.redis_url))
    args = parser.parse_args()

    hashtags = [t.strip().lstrip("#") for t in args.hashtags.split(",") if t.strip()]

    redis: Redis = Redis.from_url(args.redis_url, decode_responses=True)
    dedup = RedisDedupStore(
        redis, id_prefix=rt.dedup.id_prefix, content_prefix=rt.dedup.content_prefix, ttl_seconds=rt.dedup.ttl_seconds
    )
    supervisor = TailSupervisor(settings, dedup, build_engine(settings))
    app = create_app(supervisor, hashtags)

    logger.info(
        "starting realtime service",
        extra={"extra_fields": {"hashtags": hashtags, "redis": args.redis_url, "port": args.port}},
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=settings.logging.level.lower())


if __name__ == "__main__":
    main()
