"""
Threat assessment endpoints — summary, per-object assessment, alerts,
trajectory prediction, conjunction analysis, and full assess-all.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from src.api.models import (
    AlertResponse,
    AssessAllStatus,
    ConjunctionResponse,
    ThreatAssessmentResponse,
    ThreatSummaryResponse,
    TrajectoryPredictionResponse,
)

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

    # Look up object type for type-aware assessment routing
    summary = catalog.get_object_summary(object_id)
    object_type = summary.get("object_type", "PAYLOAD") if summary else "PAYLOAD"

    logger.info("Assessment requested for object %d (window=%d obs, type=%s)", object_id, window, object_type)

    result = service.assess_object(
        object_id,
        positions[-window:],
        velocities[-window:],
        timestamps[-window:],
        full_positions=positions,
        full_velocities=velocities,
        full_timestamps=timestamps,
        object_type=object_type,
    )

    # Add object name and type
    result["object_name"] = summary["name"] if summary else f"Object-{object_id}"
    result["object_type"] = object_type

    logger.info(
        "Assessment complete: object %d → %s (score=%.1f)",
        object_id, result["threat_tier"], result["threat_score"],
    )

    return ThreatAssessmentResponse(**result)


@router.get("/object/{object_id}/prediction", response_model=TrajectoryPredictionResponse)
async def predict_trajectory(object_id: int):
    """Predict 30-step future trajectory for a specific object using the TrajectoryTransformer."""
    catalog = _get_catalog()
    service = _get_threat_service()

    result = service.predict_trajectory(object_id, catalog)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Prediction unavailable for object {object_id}")

    return TrajectoryPredictionResponse(**result)


@router.post("/assess-all", response_model=AssessAllStatus)
async def assess_all():
    """Start full assessment of all objects in background."""
    from src.api.main import app_state
    if app_state.get("read_only"):
        raise HTTPException(status_code=403, detail="Read-only deployment — use demo video for full ML pipeline run")

    catalog = _get_catalog()
    service = _get_threat_service()
    clock = app_state.get("clock")
    timestep = clock.timestep if clock else 0

    status = service.get_assess_all_status()
    if status["running"]:
        return AssessAllStatus(**status)

    asyncio.create_task(service.assess_all(catalog, timestep))
    # Return immediate status
    return AssessAllStatus(running=True, completed=0, total=catalog.n_objects)


@router.get("/assess-all/status", response_model=AssessAllStatus)
async def assess_all_status():
    """Get progress of the running assess-all operation."""
    service = _get_threat_service()
    return AssessAllStatus(**service.get_assess_all_status())


@router.get("/conjunctions", response_model=ConjunctionResponse)
async def get_conjunctions():
    """Get current top conjunction (collision) risks."""
    from src.api.main import app_state
    conj_service = app_state.get("conjunction_service")
    if conj_service is None:
        return ConjunctionResponse(pairs=[], analyzed_pairs=0, timestamp="")
    return ConjunctionResponse(**conj_service.get_results())


@router.get("/alerts", response_model=list[AlertResponse])
async def get_alerts(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Get recent threat alerts."""
    from src.api.database import get_alerts
    alerts = get_alerts(limit=limit, offset=offset)
    return [AlertResponse(**a) for a in alerts]


@router.delete("/alerts")
async def delete_alerts():
    """Clear all alerts from the database."""
    from src.api.main import app_state
    if app_state.get("read_only"):
        raise HTTPException(status_code=403, detail="Read-only deployment")
    from src.api.database import clear_alerts
    count = clear_alerts()
    return {"deleted": count}
