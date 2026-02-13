"""
Object endpoints — list all tracked objects and get details.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.api.models import ObjectDetail, ObjectSummary, ObjectTypeEnum, TrajectoryPoint, ThreatTierEnum

router = APIRouter(prefix="/api/objects", tags=["objects"])


def _get_catalog():
    """Get the SpaceCatalog from app state (injected at startup)."""
    from src.api.main import app_state
    return app_state["catalog"]


def _get_threat_tiers():
    """Get pre-computed threat tiers."""
    from src.api.main import app_state
    return app_state.get("threat_tiers", {})


@router.get("", response_model=list[ObjectSummary])
async def list_objects(
    regime: str | None = Query(None, description="Filter by regime: LEO, MEO, GEO, HEO"),
    object_type: str | None = Query(None, description="Filter by type: PAYLOAD, DEBRIS, ROCKET_BODY"),
    limit: int = Query(1000, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List all tracked space objects with summary info."""
    catalog = _get_catalog()
    summaries = catalog.get_all_summaries()
    tiers = _get_threat_tiers()

    if regime:
        summaries = [s for s in summaries if s["regime"] == regime.upper()]
    if object_type:
        summaries = [s for s in summaries if s.get("object_type", "PAYLOAD") == object_type.upper()]

    page = summaries[offset:offset + limit]

    return [
        ObjectSummary(
            **s,
            threat_tier=ThreatTierEnum(tiers.get(s["id"], "MINIMAL")),
            threat_score=0.0,
        )
        for s in page
    ]


@router.get("/{object_id}", response_model=ObjectDetail)
async def get_object(
    object_id: int,
    trajectory_start: int = Query(0, ge=0),
    trajectory_end: int | None = Query(None),
):
    """Get detailed info for a single object, including trajectory."""
    catalog = _get_catalog()
    summary = catalog.get_object_summary(object_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Object {object_id} not found")

    tiers = _get_threat_tiers()

    trajectory_data = catalog.get_object_trajectory(
        object_id, start=trajectory_start, end=trajectory_end
    )

    return ObjectDetail(
        **summary,
        threat_tier=ThreatTierEnum(tiers.get(object_id, "MINIMAL")),
        trajectory=[TrajectoryPoint(**t) for t in trajectory_data],
    )
