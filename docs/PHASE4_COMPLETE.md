# Phase 4: Operational Dashboard — Complete

## Overview

Phase 4 delivers a real-time operational dashboard for the Space Domain Awareness system. It visualizes 1,000 satellite tracks on a 3D CesiumJS globe with threat tier coloring, provides on-demand ML-powered threat assessment, and showcases all Phase 3 model metrics.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  React Frontend (TypeScript + Vite)                 │
│  ├── CesiumJS Globe (1000 satellites, color-coded)  │
│  ├── Playback Controls (play/pause/speed/seek)      │
│  ├── Object Detail (threat + trajectory + charts)   │
│  ├── Threat Summary (tier distribution bars)        │
│  ├── Alert Feed (ELEVATED/CRITICAL alerts)          │
│  └── Bottom Tabs (Tracking, ML, System, About)      │
├─────────────────────────────────────────────────────┤
│  WebSocket /api/ws ──── position updates @ 1Hz      │
│  REST API /api/* ────── objects, threat, simulation  │
├─────────────────────────────────────────────────────┤
│  FastAPI Backend                                    │
│  ├── SpaceCatalog (parquet → numpy, geodetic cache) │
│  ├── SimulationClock (async timer, configurable)    │
│  ├── ThreatService (wraps ML pipeline, caching)     │
│  ├── SQLite (alerts, assessment cache)              │
│  └── ConnectionManager (WebSocket broadcast)        │
└─────────────────────────────────────────────────────┘
```

## Running the Dashboard

### Production Mode
```bash
python scripts/run_dashboard.py
# Dashboard at http://localhost:8000
```

### Development Mode
```bash
# Terminal 1: Backend with hot reload
python scripts/run_dashboard.py --dev

# Terminal 2: Frontend with Vite HMR
cd src/dashboard && npm run dev
# Frontend at http://localhost:5173 (proxies API to :8000)
```

### Custom Port
```bash
python scripts/run_dashboard.py --port 8080
```

## API Reference

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check, object/timestep counts |
| GET | `/api/objects` | List all 1000 objects (filterable by regime) |
| GET | `/api/objects/{id}` | Object detail with full trajectory |
| GET | `/api/simulation/status` | Playback state (playing, speed, timestep) |
| POST | `/api/simulation/play` | Start playback |
| POST | `/api/simulation/pause` | Pause playback |
| POST | `/api/simulation/speed?speed=60` | Set speed multiplier |
| POST | `/api/simulation/seek?timestep=500` | Jump to timestep |
| GET | `/api/threat/summary` | Threat tier distribution |
| GET | `/api/threat/object/{id}` | On-demand threat assessment |
| GET | `/api/threat/alerts` | Recent alerts |
| GET | `/api/metrics` | System metrics |
| WS | `/api/ws` | WebSocket position stream |

### WebSocket Message Format
```json
{
  "type": "positions",
  "timestep": 42,
  "time_iso": "2026-02-07T04:12:47.850116+00:00",
  "objects": [
    {
      "id": 0,
      "name": "CALSPHERE 1",
      "lat": 6.0499,
      "lon": 58.4860,
      "alt_km": 995.31,
      "threat_tier": "MINIMAL"
    }
  ]
}
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI 0.109 + uvicorn 0.27 |
| Database | SQLite + SQLAlchemy 2.0 |
| Frontend | React 18 + TypeScript 5.6 |
| Build | Vite 5.4 |
| 3D Globe | CesiumJS + Resium |
| Charts | Recharts |
| State | Zustand |
| Styling | TailwindCSS 3.4 |
| ML Pipeline | PyTorch (ThreatAssessmentPipeline) |

## File Structure

### Backend (Python) — `src/api/`
```
src/api/
├── __init__.py
├── main.py              # FastAPI app, lifespan, routes
├── data_manager.py      # SpaceCatalog: parquet loading, geodetic conversion
├── database.py          # SQLite: alerts, assessment cache
├── models.py            # Pydantic response schemas
├── simulation_clock.py  # Async simulation timer
├── threat_service.py    # ThreatAssessmentPipeline wrapper
└── routes/
    ├── __init__.py
    ├── objects.py       # GET /api/objects, /api/objects/{id}
    ├── simulation.py    # Simulation control endpoints
    ├── threat.py        # Threat assessment endpoints
    ├── websocket.py     # WebSocket broadcast
    └── metrics.py       # System metrics
```

### Frontend (TypeScript) — `src/dashboard/`
```
src/dashboard/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
├── index.html
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── types/index.ts
    ├── store/useSimStore.ts
    ├── services/
    │   ├── api.ts
    │   └── websocket.ts
    ├── styles/globals.css
    └── components/
        ├── Globe.tsx
        ├── Header.tsx
        ├── PlaybackControls.tsx
        ├── ThreatSummary.tsx
        ├── AlertFeed.tsx
        ├── ObjectDetail.tsx
        ├── BottomTabs.tsx
        ├── TierBadge.tsx
        ├── ScoreGauge.tsx
        └── tabs/
            ├── TrackingTab.tsx
            ├── MLPerformanceTab.tsx
            ├── SystemMetricsTab.tsx
            └── AboutTab.tsx
```

### Tests
```
tests/
├── unit/
│   ├── test_data_manager.py      # 37 tests
│   ├── test_simulation_clock.py  # 12 tests
│   └── test_threat_service.py    # 8 tests
└── integration/
    └── test_api_endpoints.py     # 20 tests
```

## Performance

| Metric | Result | Target |
|--------|--------|--------|
| Data loading | 0.68s | - |
| Geodetic pre-computation | 0.26s (vectorized) | - |
| API response time | <10ms | <100ms |
| Threat assessment (on-demand) | ~14ms | <50ms |
| WebSocket broadcast | 1000 objects/tick | 1Hz |
| Backend startup | <1s total | - |
| Frontend build | 2.3s | - |
| Memory (backend) | ~35 MB positions | <1GB |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| + / = | Increase speed |
| - | Decrease speed |
| Escape | Deselect object |

## Testing

```bash
# Run Phase 4 tests
PYTHONPATH=. pytest tests/unit/test_data_manager.py \
  tests/unit/test_simulation_clock.py \
  tests/unit/test_threat_service.py \
  tests/integration/test_api_endpoints.py -v

# Run ALL tests (Phases 1-4)
PYTHONPATH=. pytest tests/ -v
# 410 tests passing
```

## Key Design Decisions

1. **Pre-computed geodetic coordinates**: Vectorized ECI→ECEF→geodetic conversion for all 1000×1440 positions at startup (0.26s) avoids per-frame computation.

2. **SimulationClock**: Async background task with configurable speed (1x-3600x). Default 60x = 1 simulated minute per real second.

3. **Lazy ML pipeline loading**: ThreatService lazy-loads ThreatAssessmentPipeline on first assessment request to avoid slowing startup.

4. **WebSocket broadcast**: Single ConnectionManager broadcasts position JSON to all connected clients on each clock tick.

5. **SQLite caching**: Assessment results cached by (object_id, timestep) to avoid redundant ML inference.
