# System Architecture

## Design Principles

1. **Type-Aware Processing** — Different object types (payload, debris, rocket body) take different pipeline paths
2. **Uncertainty-Aware** — All ML outputs include confidence bounds
3. **Explainability** — Every threat score has a human-readable explanation with reasoning
4. **Real-Time** — 1Hz WebSocket updates, sub-10ms assessment latency
5. **Validation-First** — 443 tests, 54-check validation suite

## Data Flow

```
                    CelesTrak TLEs
                         │
                    ┌────▼────┐
                    │  Mixed  │  build_mixed_catalog.py
                    │ Catalog │  600 PAYLOAD + 350 DEBRIS + 50 RB
                    └────┬────┘
                         │
                    ┌────▼─────┐
                    │ SGP4     │  orbital_mechanics.py
                    │ Propagate│  sensor_models.py
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ Dataset  │  data_generator.py
                    │ Parquet  │  ground_truth.parquet (pos/vel/time + object_type)
                    └────┬─────┘
                         │
                 ┌───────▼───────┐
                 │  SpaceCatalog │  data_manager.py
                 │  (in-memory)  │  Loads parquet → numpy arrays + geodetic cache
                 └───────┬───────┘
                         │
              ┌──────────▼──────────┐
              │  ThreatService      │  threat_service.py
              │  assess_by_type()   │  Routes by object_type
              └──┬──────────────┬───┘
                 │              │
        PAYLOAD  │              │  DEBRIS / ROCKET_BODY
                 ▼              ▼
    ┌────────────────┐   ┌──────────────┐
    │ Full Pipeline  │   │ Collision    │
    │ (6 steps)      │   │ Only (3 steps)│
    └───────┬────────┘   └──────┬───────┘
            │                   │
            ▼                   ▼
    ┌────────────────────────────────┐
    │  ThreatScore (0-100)           │
    │  ThreatTier (5 levels)         │
    │  Explanation (contextual)      │
    └───────────────┬────────────────┘
                    │
         ┌──────────▼──────────┐
         │  Dashboard          │
         │  WebSocket + REST   │
         │  CesiumJS Globe     │
         └─────────────────────┘
```

## Layer 1: Simulation

Generates realistic orbital data from real CelesTrak TLE files.

**Key Files:**
- `src/simulation/orbital_mechanics.py` — SGP4 propagation via Skyfield
- `src/simulation/sensor_models.py` — Radar/optical sensor simulation
- `src/simulation/data_generator.py` — Dataset creation pipeline
- `src/simulation/noise_models.py` — Measurement uncertainty injection
- `src/simulation/tle_loader.py` — TLE parsing and object catalog loading

**Outputs:** Parquet files with position (x,y,z), velocity (vx,vy,vz), timestamps, object_type

## Layer 2: Tracking

State estimation and data association for multi-object tracking.

**Key Files:**
- `src/tracking/kalman_filters.py` — Extended & Unscented Kalman Filters
- `src/tracking/data_association.py` — Global Nearest Neighbor + Hungarian algorithm
- `src/tracking/track_manager.py` — Track lifecycle (initiation, maintenance, deletion)
- `src/tracking/maneuver_detection.py` — Chi-squared innovation tests
- `src/tracking/multi_object_tracker.py` — Orchestrates all tracking components

## Layer 3: ML Ensemble (7 Models)

### Neural Models (4)

**TrajectoryTransformer + ParallelHead** (371K params)
- `src/ml/models/trajectory_transformer.py`
- Encoder-only transformer with parallel position+velocity prediction heads
- Input: 20-step feature sequence (24D) → Output: 30-step future trajectory
- Trained: `scripts/train_transformer_scaled.py`
- Checkpoint: `checkpoints/phase3_parallel/best_model.pt`

**ManeuverClassifier CNN-LSTM-Attention** (719K params)
- `src/ml/models/maneuver_classifier.py`
- 1D CNN feature extraction → bidirectional LSTM → attention → 6-class output
- Classes: Normal, Drift/Decay, Station-keeping, Minor Maneuver, Major Maneuver, Deorbit
- Checkpoint: `checkpoints/phase3_day4/maneuver_classifier.pt`

**BehaviorAutoencoder** (~2.5K params)
- `src/ml/anomaly/autoencoder.py`
- Reconstruction-based anomaly detection on behavioral feature vectors
- TPR 100%, FPR 5%
- Checkpoint: `checkpoints/phase3_anomaly/`

**CollisionRiskPredictor** (~90K params)
- `src/ml/models/collision_predictor.py`
- Predicts collision risk from relative trajectory encoding
- Checkpoint: `checkpoints/collision_predictor/`

### Rule-Based Models (3)

**IntentClassifier** — `src/ml/intent/intent_classifier.py`
- Maps maneuver class (0-5) → base intent + threat level
- Applies 7 escalation rules using ProximityContext:
  1. Maneuver + approaching asset → RENDEZVOUS
  2. Station-keeping + co-orbital + closing → SURVEILLANCE
  3. PHASING pattern → RENDEZVOUS/SURVEILLANCE
  4. SHADOWING pattern → SURVEILLANCE/HIGH
  5. EVASION pattern → EVASIVE
  6. Very close + closing fast → COLLISION_AVOIDANCE
  7. CNN-LSTM station-keeping + sustained co-orbital → SHADOWING
- Co-orbital check (< 1 km/s relative velocity) prevents high-speed flyby false positives

**ThreatScorer** — `src/ml/threat/threat_scorer.py`
- Weighted fusion: intent (35%) + anomaly (25%) + proximity (25%) + pattern (15%)
- Proximity sub-score: 50% distance + 30% closing rate + 20% TCA
- Tiers: 0-19 MINIMAL, 20-39 LOW, 40-59 MODERATE, 60-79 ELEVATED, 80-100 CRITICAL

**ThreatEscalator** — `src/ml/intent/threat_escalation.py`
- Detects multi-step behavioral patterns across maneuver history
- PHASING: 3+ maneuvers in 24h window
- SHADOWING: Station-keeping near asset for 12h+ (co-orbital only)
- EVASION: Sudden maneuver after 5+ normal observations

### Pipeline Orchestration

**ThreatAssessmentPipeline** — `src/ml/threat_assessment.py`
- `assess()` — Full 6-step pipeline for payloads
- `assess_by_type()` — Routes DEBRIS/RB to `_assess_passive_object()`
- `_find_closest_approach()` — Rodrigues rotation propagation for LEO asset tracking
- `_derive_maneuvers()` — Adaptive gravity subtraction (min of Euler/trapezoidal)

## Layer 4: Dashboard

### Backend (FastAPI)

**Key Files:**
- `src/api/main.py` — App startup, static file serving, CORS
- `src/api/data_manager.py` — SpaceCatalog (parquet → numpy, geodetic cache)
- `src/api/threat_service.py` — Wraps pipeline with caching, alerts, batch assessment
- `src/api/scenario_injector.py` — 7 threat scenarios on objects 990-996
- `src/api/conjunction_service.py` — Periodic pairwise proximity analysis
- `src/api/simulation_clock.py` — Async timer, configurable speed
- `src/api/database.py` — SQLite for alerts and assessment cache

**Routes:**
- `routes/objects.py` — GET /api/objects, /api/objects/{id} (type filter)
- `routes/threat.py` — POST /api/threat/assess/{id}, assess-all, predictions
- `routes/websocket.py` — WS /api/ws (1Hz position + threat_tier broadcast)
- `routes/simulation.py` — Play/pause/speed/seek/reset
- `routes/metrics.py` — System metrics

### Frontend (React + CesiumJS)

**Key Files:**
- `src/dashboard/src/components/Globe.tsx` — CesiumJS 3D globe, size by type, color by tier
- `src/dashboard/src/components/ObjectDetail.tsx` — Threat assessment panel, CNN-LSTM chart, altitude profile
- `src/dashboard/src/components/AlertFeed.tsx` — ELEVATED/CRITICAL alert stream
- `src/dashboard/src/components/ThreatSummary.tsx` — Tier distribution bars
- `src/dashboard/src/components/tabs/TrackingTab.tsx` — Object table with type filter pills
- `src/dashboard/src/components/tabs/AboutTab.tsx` — Data source attribution, type breakdown
- `src/dashboard/src/store/useSimStore.ts` — Zustand state management
- `src/dashboard/src/types/index.ts` — TypeScript types, tier colors, object type constants

## Layer 5: Validation

- 443 unit tests across `tests/unit/`
- 54-check validation suite: `scripts/validate_pipeline.py`
- Integration tests with real parquet data
- GPU sandbox testing for CUDA operations

## Protected Assets (6)

| Asset | Regime | Altitude |
|-------|--------|----------|
| ISS | LEO | 420 km |
| DSP-23 | GEO | 35,786 km |
| SBIRS-GEO-1 | GEO | 35,786 km |
| WGS-10 | GEO | 35,786 km |
| GPS-IIR-7 | MEO | 20,200 km |
| TDRS-13 | GEO | 35,786 km |

---

**Last Updated**: 2026-02-12
