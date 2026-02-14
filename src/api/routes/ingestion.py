"""
TLE ingestion endpoints — POST new objects, check ingestion status.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.api.models import (
    IngestionStatusResponse,
    TLEIngestRequest,
    TLEIngestResponse,
)

router = APIRouter(prefix="/api/ingest", tags=["ingestion"])
logger = logging.getLogger(__name__)


def _get_ingestion_service():
    from src.api.main import app_state
    return app_state["ingestion_service"]


@router.post("/tle", response_model=TLEIngestResponse)
async def ingest_tle(request: TLEIngestRequest):
    """Ingest a new object from TLE data. Propagates via SGP4 and adds to catalog."""
    service = _get_ingestion_service()
    try:
        result = service.ingest_tle(request.name, request.tle_line1, request.tle_line2)
    except Exception as e:
        logger.exception("TLE ingestion failed for '%s'", request.name)
        raise HTTPException(status_code=400, detail=str(e))
    return TLEIngestResponse(**result)


@router.get("/status", response_model=IngestionStatusResponse)
async def ingestion_status():
    """Return count of ingested objects and total catalog size."""
    service = _get_ingestion_service()
    return IngestionStatusResponse(**service.get_status())
