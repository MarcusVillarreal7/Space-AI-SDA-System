"""
System metrics endpoint — live counters for the dashboard.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from src.api.models import MetricsResponse

router = APIRouter(prefix="/api", tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get live system metrics."""
    from src.api.main import app_state
    from src.api.routes.websocket import get_manager

    catalog = app_state.get("catalog")
    metrics = app_state.get("metrics", {})
    manager = get_manager()

    api_requests = metrics.get("api_requests", 0)
    latency_sum = metrics.get("api_latency_sum", 0.0)
    avg_latency = (latency_sum / api_requests * 1000) if api_requests > 0 else 0.0

    return MetricsResponse(
        objects_tracked=catalog.n_objects if catalog else 0,
        websocket_connections=manager.count,
        api_requests=api_requests,
        avg_api_latency_ms=round(avg_latency, 2),
        assessments_completed=metrics.get("assessments_completed", 0),
        uptime_seconds=round(time.time() - metrics.get("start_time", time.time()), 1),
    )
