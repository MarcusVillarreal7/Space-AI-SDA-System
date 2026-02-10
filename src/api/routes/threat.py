"""
Threat assessment endpoints — summary, per-object assessment, alerts.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from src.api.models import AlertResponse, ThreatAssessmentResponse, ThreatSummaryResponse

router = APIRouter(prefix="/api/threat", tags=["threat"])
logger = logging.getLogger(__name__)


def _get_threat_service():
    from src.api.main import app_state
    return app_state["threat_service"]


def _get_catalog():
    from src.api.main import app_state
    return app_state["catalog"]


@router.get("/summary", response_model=ThreatSummaryResponse)
async def threat_summary():
    """Get threat tier distribution across all tracked objects."""
    service = _get_threat_service()
    catalog = _get_catalog()
    by_tier = service.get_tier_summary()
    return ThreatSummaryResponse(
        total=catalog.n_objects,
        by_tier=by_tier,
    )


@router.get("/object/{object_id}", response_model=ThreatAssessmentResponse)
async def assess_object(object_id: int):
    """Run on-demand threat assessment for a specific object."""
    catalog = _get_catalog()
    service = _get_threat_service()

    data = catalog.get_positions_and_velocities(object_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Object {object_id} not found")

    positions, velocities, timestamps = data

    # Use last 20 observations for assessment
    window = min(20, len(timestamps))

    logger.info("Assessment requested for object %d (window=%d obs)", object_id, window)

    result = service.assess_object(
        object_id,
        positions[-window:],
        velocities[-window:],
        timestamps[-window:],
    )

    # Add object name
    summary = catalog.get_object_summary(object_id)
    result["object_name"] = summary["name"] if summary else f"Object-{object_id}"

    logger.info(
        "Assessment complete: object %d → %s (score=%.1f)",
        object_id, result["threat_tier"], result["threat_score"],
    )

    return ThreatAssessmentResponse(**result)


@router.get("/alerts", response_model=list[AlertResponse])
async def get_alerts(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Get recent threat alerts."""
    from src.api.database import get_alerts
    alerts = get_alerts(limit=limit, offset=offset)
    return [AlertResponse(**a) for a in alerts]
