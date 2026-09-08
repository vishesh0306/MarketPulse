"""HTTP + WebSocket surface for the real-time signal.

`GET /signals` is the history a new client backfills from; `WS /ws/signals` pushes each
bucket as it seals so a browser can draw the signal live instead of re-fetching a PNG on a
timer. `GET /metrics` exposes the latency/throughput gauges.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.realtime.supervisor import TailSupervisor
from src.utils.logger import get_logger

logger = get_logger("realtime_api")


class WebSocketHub:
    """Fan-out of sealed-bucket batches to every connected WebSocket. Registered as a
    seal sink on the supervisor."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def broadcast(self, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        payload = {"type": "sealed", "rows": rows}
        async with self._lock:
            targets = list(self._clients)
        for ws in targets:
            try:
                if ws.application_state == WebSocketState.CONNECTED:
                    await ws.send_json(payload)
            except (WebSocketDisconnect, RuntimeError):
                await self.disconnect(ws)


def create_app(supervisor: TailSupervisor, hashtags: list[str]) -> FastAPI:
    hub = WebSocketHub()
    supervisor._seal_sinks.append(hub.broadcast)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(supervisor.run(hashtags))
        logger.info("realtime service up", extra={"extra_fields": {"hashtags": hashtags}})
        try:
            yield
        finally:
            supervisor.stop()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(task, timeout=10)

    app = FastAPI(title="MarketPulse real-time signal", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "live_buckets": supervisor.live_snapshot().__len__(), "ws_clients": hub.client_count}

    @app.get("/signals")
    async def signals(live: bool = False) -> dict[str, object]:
        sealed = supervisor.sealed_history()
        body: dict[str, object] = {"sealed": sealed}
        if live:
            body["live"] = supervisor.live_snapshot()
        return body

    @app.get("/metrics")
    async def metrics() -> dict[str, object]:
        return supervisor.metrics.snapshot()

    @app.websocket("/ws/signals")
    async def ws_signals(ws: WebSocket) -> None:
        await hub.connect(ws)
        try:
            await ws.send_json({"type": "history", "rows": supervisor.sealed_history()})
            while True:
                # No client->server messages are expected; this just keeps the socket
                # open and notices a disconnect.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await hub.disconnect(ws)

    return app
