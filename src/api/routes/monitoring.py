"""
Monitoring endpoints — prediction logs, aggregate stats, ingestion history.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from src.api.database import get_ingestion_logs, get_prediction_logs, get_prediction_stats
from src.api.models import (
    IngestionLogResponse,
    PredictionLogResponse,
    PredictionStatsResponse,
)

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])
logger = logging.getLogger(__name__)


@router.get("/predictions", response_model=list[PredictionLogResponse])
async def prediction_history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Get paginated prediction log history."""
    logs = get_prediction_logs(limit=limit, offset=offset)
    return [PredictionLogResponse(**log) for log in logs]


@router.get("/stats", response_model=PredictionStatsResponse)
async def prediction_stats():
    """Get aggregate prediction statistics (tier distribution, avg latency)."""
    stats = get_prediction_stats()
    return PredictionStatsResponse(**stats)


@router.get("/ingestions", response_model=list[IngestionLogResponse])
async def ingestion_history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Get paginated ingestion log history."""
    logs = get_ingestion_logs(limit=limit, offset=offset)
    return [IngestionLogResponse(**log) for log in logs]
