"""
WebSocket endpoint — broadcasts position updates to all connected clients.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts position data."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connected (%d total)", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket disconnected (%d total)", len(self.active_connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send JSON message to all connected clients."""
        if not self.active_connections:
            return

        data = json.dumps(message)
        disconnected = []
        for conn in self.active_connections:
            try:
                await conn.send_text(data)
            except Exception:
                disconnected.append(conn)

        for conn in disconnected:
            self.disconnect(conn)

    @property
    def count(self) -> int:
        return len(self.active_connections)


# Singleton connection manager
manager = ConnectionManager()


def get_manager() -> ConnectionManager:
    return manager


@router.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time position updates.

    Clients receive messages on each simulation tick:
    {
        "type": "positions",
        "timestep": 42,
        "time_iso": "2026-02-07T04:12:47...",
        "objects": [
            {"id": 0, "name": "CALSPHERE 1", "lat": 6.05, "lon": 58.49, "alt_km": 995.3, "threat_tier": "MINIMAL"},
            ...
        ]
    }
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; handle any client messages
            data = await websocket.receive_text()
            # Clients can send commands like {"type": "ping"}
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except (json.JSONDecodeError, KeyError):
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def broadcast_positions(timestep: int) -> None:
    """
    Callback for SimulationClock — broadcasts positions to all WebSocket clients.
    Called once per simulation tick.
    """
    from src.api.main import app_state

    catalog = app_state.get("catalog")
    if catalog is None or not catalog.is_loaded:
        return

    threat_tiers = app_state.get("threat_tiers", {})

    positions = catalog.get_all_positions_at_timestep(timestep)
    objects = [
        {
            "id": p["id"],
            "name": p["name"],
            "lat": round(p["lat"], 4),
            "lon": round(p["lon"], 4),
            "alt_km": round(p["alt_km"], 2),
            "threat_tier": threat_tiers.get(p["id"], "MINIMAL"),
        }
        for p in positions
    ]

    message = {
        "type": "positions",
        "timestep": timestep,
        "time_iso": catalog.time_isos[timestep] if timestep < len(catalog.time_isos) else "",
        "objects": objects,
    }

    await manager.broadcast(message)
