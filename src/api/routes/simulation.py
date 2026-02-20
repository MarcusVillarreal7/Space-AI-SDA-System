"""
Simulation control endpoints — play/pause/speed/seek.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.api.models import SimulationStatus

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


def _get_clock():
    from src.api.main import app_state
    return app_state["clock"]


def _get_catalog():
    from src.api.main import app_state
    return app_state["catalog"]


@router.get("/status", response_model=SimulationStatus)
async def simulation_status():
    """Get current simulation clock status."""
    clock = _get_clock()
    catalog = _get_catalog()
    return SimulationStatus(
        is_playing=clock.is_playing,
        speed=clock.speed,
        timestep=clock.timestep,
        max_timestep=catalog.n_timesteps - 1,
        time_iso=catalog.time_isos[clock.timestep] if catalog.time_isos else "",
    )


@router.post("/play")
async def play():
    """Start simulation playback."""
    clock = _get_clock()
    clock.play()
    return {"status": "playing"}


@router.post("/pause")
async def pause():
    """Pause simulation playback."""
    clock = _get_clock()
    clock.pause()
    return {"status": "paused"}


@router.post("/speed")
async def set_speed(speed: float = Query(..., ge=0.1, le=3600)):
    """Set simulation speed multiplier."""
    clock = _get_clock()
    clock.set_speed(speed)
    return {"speed": clock.speed}


@router.post("/seek")
async def seek(timestep: int = Query(..., ge=0)):
    """Seek to a specific timestep."""
    clock = _get_clock()
    catalog = _get_catalog()
    max_ts = catalog.n_timesteps - 1
    if timestep > max_ts:
        raise HTTPException(
            status_code=400,
            detail=f"Timestep {timestep} exceeds max {max_ts}",
        )
    clock.seek(timestep)
    return {"timestep": clock.timestep}


@router.post("/reset")
async def reset_simulation():
    """Reset simulation to initial state: clock to 0, clear all assessments and alerts."""
    from src.api.main import app_state
    if app_state.get("read_only"):
        raise HTTPException(status_code=403, detail="Read-only deployment — reset disabled")
    from src.api.database import clear_alerts, clear_assessment_cache

    # Reset clock to beginning and resume playback
    clock = _get_clock()
    clock.seek(0)
    clock.play()

    # Clear threat tiers
    app_state["threat_tiers"] = {}

    # Clear threat service state
    service = app_state.get("threat_service")
    if service:
        service._assessments.clear()
        service._tier_counts = {
            "MINIMAL": 0, "LOW": 0, "MODERATE": 0,
            "ELEVATED": 0, "CRITICAL": 0,
        }
        service._assess_all_running = False
        service._assess_all_completed = 0
        service._assess_all_total = 0

    # Clear database
    clear_alerts()
    clear_assessment_cache()

    return {"status": "reset", "timestep": 0}
