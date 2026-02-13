# Space Domain Awareness AI System

> Multi-Object Tracking with ML-Enhanced Threat Assessment for Space Defense Applications

## Overview

A production-grade space tracking and threat assessment system that processes 1,000 space objects in real time. The system ingests TLE orbital data, runs it through a 7-model ML ensemble (4 neural + 3 rule-based), and produces explainable threat assessments on a 3D operational dashboard.

Built as a defense portfolio project demonstrating the full pipeline from raw orbital data to operator-facing threat intelligence.

## System Architecture

```
TLE Data ─→ Simulation ─→ Tracking ─→ ML Ensemble ─→ Dashboard
  (CelesTrak)   (SGP4)      (EKF/UKF)   (7 models)    (CesiumJS)

Mixed Catalog (1,000 objects):
  600 PAYLOAD  ─→ Full 6-step pipeline (CNN-LSTM + intent + anomaly + scoring)
  350 DEBRIS   ─→ Collision-only path (proximity scoring)
   50 ROCKET BODY ─→ Collision-only + breakup risk
```

## Quick Start

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd src/dashboard && npm install && npm run build && cd ../..

# Download TLEs and build mixed catalog
python scripts/download_tle_data.py
python scripts/build_mixed_catalog.py --seed 42 --generate

# Launch dashboard
python scripts/run_dashboard.py
# Open http://localhost:8000
```

## ML Models (7 Total)

| Model | Type | Params | Performance |
|-------|------|--------|-------------|
| TrajectoryTransformer + ParallelHead | PyTorch | 371K | Pos RMSE 7.57 km, Vel RMSE 1.00 km/s |
| ManeuverClassifier (CNN-LSTM-Attention) | PyTorch | 719K | 84.5% accuracy, 6 classes |
| BehaviorAutoencoder | PyTorch | ~2.5K | TPR 100%, FPR 5% |
| CollisionRiskPredictor | PyTorch | ~90K | Relative trajectory encoding |
| IntentClassifier | Rule-based | -- | 10 intent categories, 7 escalation rules |
| ThreatScorer | Rule-based | -- | Weighted fusion (intent 35%, anomaly 25%, proximity 25%, pattern 15%) |
| ThreatEscalator | Rule-based | -- | Phasing, shadowing, evasion pattern detection |

## Threat Assessment Pipeline

Every object is assessed through a type-aware pipeline:

**PAYLOAD objects (full pipeline):**
1. Derive maneuver history from velocity changes (gravity-subtracted)
2. Classify current maneuver via CNN-LSTM (6 classes) or heuristic fallback
3. Scan trajectory for closest approach to 6 protected assets (ISS, DSP-23, SBIRS, WGS-10, GPS-IIR-7, TDRS-13)
4. Classify intent via proximity context + 7 escalation rules + co-orbital detection
5. Detect behavioral anomalies via autoencoder reconstruction error
6. Fuse into 0-100 threat score with 5 tiers (MINIMAL / LOW / MODERATE / ELEVATED / CRITICAL)

**DEBRIS / ROCKET BODY (collision-only):**
1. Scan for closest approach to protected assets
2. Compute proximity score (distance + closing rate + TCA)
3. Add breakup risk bonus for rocket bodies (+5)

## 7 Injected Threat Scenarios

| ID | Object | Scenario | Expected |
|----|--------|----------|----------|
| 990 | COSMOS-2558 | Rendezvous approach toward ISS | CRITICAL |
| 991 | LUCH/OLYMP | GEO shadowing of DSP-23, 15km standoff | ELEVATED |
| 992 | COSMOS-2542 | Sudden 0.5 km/s evasive maneuver | ELEVATED |
| 993 | DEBRIS-KZ-1A | Collision course debris toward ISS | CRITICAL |
| 994 | SJ-17 | 3 phasing burns toward SBIRS-GEO-1 | ELEVATED |
| 995 | SHIJIAN-21 | GEO drift + station-keep near WGS-10 | ELEVATED |
| 996 | OBJECT-2024-999A | Approach then evasive maneuver near TDRS-13 | ELEVATED |

## Dashboard

- **Frontend**: React 18 + TypeScript + CesiumJS + Recharts + Zustand
- **Backend**: FastAPI + SQLite + WebSocket (1Hz position updates)
- **Globe**: 1,000 objects color-coded by threat tier, size by object type, pulsing for CRITICAL/ELEVATED
- **Object Detail**: Threat assessment, CNN-LSTM classification chart, altitude profile, contextual explanations
- **Tracking Tab**: Sortable table with type filter pills (All / Satellite / Debris / Rocket Body)
- **Alert Feed**: ELEVATED and CRITICAL assessments only

## Performance

| Metric | Value |
|--------|-------|
| Assessment throughput | 226 objects/sec |
| Mean latency | 7.0 ms/object (full pipeline) |
| Tests | 443 passing |
| Validation | 54 PASS / 3 WARN / 0 FAIL |
| Code coverage | 75% |

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Neural Networks | PyTorch (torch.nn) |
| Tracking | NumPy, SciPy (EKF/UKF) |
| Features | NumPy, scikit-learn (StandardScaler) |
| Backend | FastAPI, SQLite, WebSocket, Uvicorn |
| Frontend | React 18, TypeScript, CesiumJS, Recharts, Zustand, Vite |
| Testing | pytest, 443 tests |
| Data | CelesTrak TLEs, Parquet (PyArrow) |

## Project Structure

```
space-ai/
├── src/
│   ├── simulation/          # SGP4 propagation, sensor models, data generation
│   ├── tracking/            # EKF/UKF, data association, track management
│   ├── ml/
│   │   ├── models/          # TrajectoryTransformer, ManeuverClassifier, CollisionPredictor
│   │   ├── features/        # Feature extraction, sequence building, augmentation
│   │   ├── training/        # Trainer, loss functions
│   │   ├── uncertainty/     # MC Dropout, ensembles, conformal prediction
│   │   ├── intent/          # IntentClassifier, ProximityContext, ThreatEscalator
│   │   ├── anomaly/         # BehaviorAutoencoder, anomaly detection
│   │   ├── threat/          # ThreatScorer, ThreatExplainer
│   │   ├── inference.py     # ManeuverPredictor, TrajectoryPredictor wrappers
│   │   └── threat_assessment.py  # End-to-end pipeline (assess + assess_by_type)
│   ├── api/                 # FastAPI backend, routes, services
│   └── dashboard/           # React + CesiumJS frontend
├── scripts/                 # Training, evaluation, catalog building
├── tests/                   # 443 unit tests
├── docs/
│   ├── timeline/            # Chronological phase completion reports
│   ├── technical/           # ML module deep-dives
│   └── archive/             # Historical progress logs
├── checkpoints/             # Trained model weights (gitignored)
└── data/                    # TLE + parquet datasets (gitignored)
```

## Documentation

- [Phase 0: Setup & Infrastructure](docs/timeline/PHASE0_COMPLETE.md)
- [Phase 1: Simulation Layer](docs/timeline/PHASE1_COMPLETE.md)
- [Phase 2: Tracking Engine](docs/timeline/PHASE2_COMPLETE.md)
- [Phase 3: ML Models & Recovery](docs/timeline/PHASE3_COMPLETE.md)
- [Phase 4: Operational Dashboard](docs/timeline/PHASE4_COMPLETE.md)
- [Phase 5: Validation & Threat Scenarios](docs/timeline/PHASE5_COMPLETE.md)
- [Phase 5.5: Mixed Catalog & Co-Orbital Fix](docs/timeline/PHASE5_MIXED_CATALOG.md)

### Technical References
- [Anomaly Detection](docs/technical/ANOMALY_DETECTION.md)
- [Intent Classification](docs/technical/INTENT_CLASSIFICATION.md)
- [Threat Scoring](docs/technical/THREAT_SCORING.md)
- [E2E Integration](docs/technical/E2E_INTEGRATION.md)
- [Parallel Head Architecture](docs/technical/PARALLEL_HEAD.md)

## Author

**Marcus** — Building defense-relevant AI with production-grade engineering.

## License

MIT License. See [LICENSE](LICENSE).

---

**Status**: All phases complete | **Last Updated**: 2026-02-12
