"""FastAPI surface: /signals, /metrics, /health and the WebSocket feed."""

from __future__ import annotations

from pathlib import Path

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

from src.realtime.api import create_app
from src.realtime.dedup_store import RedisDedupStore
from src.realtime.supervisor import TailSupervisor, build_engine
from src.utils.config_loader import load_settings


@pytest.fixture
def supervisor(tmp_path: Path) -> TailSupervisor:
    settings = load_settings().model_copy(deep=True)
    sup = TailSupervisor(
        settings,
        RedisDedupStore(fakeredis.aioredis.FakeRedis(decode_responses=True)),
        build_engine(settings),
        output_dir=tmp_path,
    )
    sup._sealed_rows = [
        {"hashtag": "nifty50", "bucket_start": "2026-09-08T06:15:00+00:00", "composite_signal": 0.12,
         "ci_lower": 0.01, "ci_upper": 0.23, "tweet_count": 40, "sentiment_coverage": 0.3, "suppressed": False,
         "sealed_at": "2026-09-08T06:32:00+00:00"},
    ]
    return sup


@pytest.fixture
def client(supervisor: TailSupervisor, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Keep the lifespan-started tail loop from making real network calls.
    async def _no_feed(*_a: object, **_k: object) -> tuple[list, None]:
        return [], None

    monkeypatch.setattr("src.realtime.supervisor.fetch_hashtag_feed", _no_feed)
    app = create_app(supervisor, ["nifty50"])
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ws_clients"] == 0


def test_signals_returns_sealed_history(client: TestClient) -> None:
    body = client.get("/signals").json()
    assert len(body["sealed"]) == 1
    assert body["sealed"][0]["hashtag"] == "nifty50"
    assert "live" not in body


def test_signals_live_flag_adds_live_snapshot(client: TestClient) -> None:
    body = client.get("/signals?live=true").json()
    assert "live" in body and isinstance(body["live"], list)


def test_metrics_shape(client: TestClient) -> None:
    body = client.get("/metrics").json()
    for key in ("ingest_lag_seconds", "tweets_per_minute", "dedup_hit_rate", "sealed_buckets_total"):
        assert key in body


def test_ws_sends_history_then_live_seal(client: TestClient, supervisor: TailSupervisor) -> None:
    with client.websocket_connect("/ws/signals") as ws:
        history = ws.receive_json()
        assert history["type"] == "history"
        assert len(history["rows"]) == 1

        # A seal batch reaches the socket via the hub registered as a supervisor sink.
        new_rows = [{"hashtag": "sensex", "bucket_start": "2026-09-08T06:30:00+00:00", "tweet_count": 12,
                     "composite_signal": -0.05, "ci_lower": -0.2, "ci_upper": 0.1, "sentiment_coverage": 0.25,
                     "suppressed": False, "sealed_at": "2026-09-08T06:47:00+00:00"}]
        client.portal.call(supervisor._seal_sinks[-1], new_rows)  # hub.broadcast

        pushed = ws.receive_json()
        assert pushed["type"] == "sealed"
        assert pushed["rows"][0]["hashtag"] == "sensex"
