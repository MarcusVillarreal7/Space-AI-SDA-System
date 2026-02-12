"""
FastAPI Application — Space Domain Awareness Dashboard Backend.

Serves REST API + WebSocket for the operational dashboard.
In production mode, also serves the built React static files.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.data_manager import SpaceCatalog
from src.api.database import init_db
from src.api.simulation_clock import SimulationClock
from src.api.threat_service import ThreatService

logger = logging.getLogger(__name__)

# Global app state (accessed by route modules)
app_state: dict = {}

DATA_PATH = Path("data/processed/ml_train_1k/ground_truth.parquet")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and start services on startup, clean up on shutdown."""
    t0 = time.perf_counter()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load space catalog
    catalog = SpaceCatalog()
    catalog.load(DATA_PATH)

    # Inject threat scenarios (modifies objects 990-996)
    from src.api.scenario_injector import ScenarioInjector
    injector = ScenarioInjector()
    injected = injector.inject(catalog)
    logger.info("Injected %d threat scenarios", len(injected))

    app_state["catalog"] = catalog

    # Initialize simulation clock
    clock = SimulationClock(max_timestep=catalog.n_timesteps - 1)
    app_state["clock"] = clock

    # Initialize threat tiers (default all MINIMAL, populated by threat_service)
    app_state["threat_tiers"] = {}

    # Initialize threat service
    threat_service = ThreatService()
    app_state["threat_service"] = threat_service

    # Initialize conjunction service
    from src.api.conjunction_service import ConjunctionService
    conjunction_service = ConjunctionService(run_interval=10)
    app_state["conjunction_service"] = conjunction_service

    # Initialize database
    init_db()

    # Track metrics
    app_state["metrics"] = {
        "api_requests": 0,
        "api_latency_sum": 0.0,
        "assessments_completed": 0,
        "start_time": time.time(),
    }

    # Register WebSocket broadcast callback
    from src.api.routes.websocket import broadcast_positions
    clock.on_tick(broadcast_positions)

    # Register conjunction analysis callback
    clock.on_tick(conjunction_service.on_tick)

    # Start clock
    await clock.start()

    elapsed = time.perf_counter() - t0
    logger.info("Backend ready in %.2fs — %d objects, %d timesteps",
                elapsed, catalog.n_objects, catalog.n_timesteps)

    yield

    # Shutdown
    await clock.stop()
    logger.info("Backend shut down")


app = FastAPI(
    title="Space Domain Awareness API",
    description="Operational dashboard backend for satellite tracking and threat assessment",
    version="4.0.0",
    lifespan=lifespan,
)

# CORS — allow dev server on port 5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
from src.api.routes.objects import router as objects_router
from src.api.routes.simulation import router as simulation_router
from src.api.routes.threat import router as threat_router
from src.api.routes.websocket import router as ws_router
from src.api.routes.metrics import router as metrics_router

app.include_router(objects_router)
app.include_router(simulation_router)
app.include_router(threat_router)
app.include_router(ws_router)
app.include_router(metrics_router)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    catalog = app_state.get("catalog")
    return {
        "status": "ok",
        "objects_loaded": catalog.n_objects if catalog else 0,
        "timesteps": catalog.n_timesteps if catalog else 0,
    }


# Mount static files for production (built React app)
dist_path = Path("src/dashboard/dist")
if dist_path.exists():
    app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="static")
