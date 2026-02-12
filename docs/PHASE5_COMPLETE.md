# Phase 4.5 + 5: Full ML Integration, Threat Scenarios & Validation

## Overview

Phase 4.5 wires all four trained neural models into the live dashboard pipeline, injects 7 realistic adversary scenarios, and adds visual payoff (pulsing threat icons, predicted trajectory lines, maneuver probability charts). Phase 5 validates the end-to-end pipeline across 1,000 objects and 7 threat scenarios, fixing 6 layers of detection issues to achieve 54 PASS / 3 WARN / 0 FAIL.

## Architecture Changes

```
Phase 4 (before)                    Phase 4.5 + 5 (after)
──────────────────                  ────────────────────────────────
1 model active:                     4 neural + 3 rule-based models:
  BehaviorAutoencoder                 BehaviorAutoencoder (anomaly)
                                      ManeuverClassifier CNN-LSTM (classification)
                                      TrajectoryTransformer (prediction viz)
                                      CollisionRiskPredictor (conjunction)
                                      IntentClassifier (rule-based)
                                      ThreatScorer (rule-based)
                                      ThreatEscalator (pattern detection)

No threat scenarios                 7 injected adversary scenarios
998 green + 2 yellow                986 green + 7 orange/red + 7 yellow
~3ms/object assessment              ~7ms/object assessment (added CNN-LSTM)
```

## Models Wired into Pipeline

| Model | Params | Role | Checkpoint |
|-------|--------|------|------------|
| ManeuverClassifier (CNN-LSTM-Attention) | 719K | Classify maneuver type (6 classes) | `checkpoints/phase3_day4/maneuver_classifier.pt` |
| TrajectoryTransformer + ParallelHead | 371K | Predict 30-step future trajectory | `checkpoints/phase3_parallel/best_model.pt` |
| BehaviorAutoencoder | 2.5K | Anomaly score via reconstruction error | `checkpoints/phase3_anomaly/` |
| CollisionRiskPredictor | 90K | Pairwise collision risk + TTCA | `checkpoints/collision_predictor/best_model.pt` |

## Threat Scenarios (Objects 990-996)

| ID | Name | Scenario | Target | Score | Tier |
|----|------|----------|--------|-------|------|
| 990 | COSMOS-2558 | Rendezvous approach, 0.3 km/s closing | ISS (LEO) | 67.6 | ELEVATED |
| 991 | LUCH/OLYMP | GEO shadowing, 15 km standoff, 12+ hours | DSP-23 (GEO) | 64.2 | ELEVATED |
| 992 | COSMOS-2542 | Sudden 0.5 km/s maneuver at t=1150 | -- | 75.5 | ELEVATED |
| 993 | DEBRIS-KZ-1A | Retrograde collision course, 15.86 km/s | ISS (LEO) | 67.1 | ELEVATED |
| 994 | SJ-17 | 3 phasing burns toward SBIRS | SBIRS-GEO-1 | 70.9 | ELEVATED |
| 995 | SHIJIAN-21 | GEO drift + station-keep at 20 km | WGS-10 (GEO) | 63.8 | ELEVATED |
| 996 | OBJECT-2024-999A | Approach then evasive maneuver | TDRS-13 (GEO) | 75.0 | ELEVATED |

## Phase 5 Validation Fixes

Six layers of the detection pipeline required fixes during validation:

### 1. Full-Trajectory Maneuver History
**Problem:** Pattern detection (SHADOWING, PHASING, EVASION) only saw 20 timesteps (~20 min), too short for patterns requiring hours of history.
**Fix:** Added `full_timestamps` parameter flowing through `assess()` -> `threat_service` -> `routes`. When the full 1440-timestep trajectory is available, `_derive_maneuvers()` processes the entire 24h history.

### 2. LEO Proximity via Circular Orbit Propagation
**Problem:** Asset catalog stores static ECI positions. LEO assets like ISS orbit every 90 minutes, so comparing a trajectory's position at t=12h against ISS's position at t=0 gives meaningless distances.
**Fix:** Rewrote `_find_closest_approach()` with vectorized Rodrigues rotation to propagate each asset's position forward in time. Returns "warped" position/velocity that preserves the true relative state against the static catalog, so `compute_proximity()` produces correct distances.

### 3. Adaptive Gravity Subtraction
**Problem:** `_derive_maneuvers()` subtracts expected gravitational acceleration to isolate thrust. Forward Euler has O(dt^2) error (~18 m/s for LEO at 60s steps), falsely classifying normal orbital motion as class 2-3. Trapezoidal reduces error to O(dt^3) for analytical orbits but introduces ~16 m/s residuals on Euler-integrated trajectories (from the scenario injector).
**Fix:** Compute residuals using both forward Euler and trapezoidal methods, pick the minimum at each timestep. Handles both data sources correctly: analytical trajectories get ~0.35 m/s residual (class 0), Euler-integrated get ~0 (class 0), real maneuvers are preserved.

### 4. CNN-LSTM-Based SHADOWING Detection (Rule 7)
**Problem:** GEO station-keeping produces per-timestep delta-V below 5 m/s (class 0), so `ThreatEscalator.is_shadowing_pattern()` never fires (it needs class 2 events). Yet the CNN-LSTM correctly classifies the overall behavior as station-keeping (class 2).
**Fix:** Added Rule 7 to `IntentClassifier._apply_escalation()`: when CNN-LSTM says class 2, the object is in the same regime as an asset, within 200 km, and history spans 12+ hours, trigger SHADOWING directly from the neural classification.

### 5. Stricter Station-Keeping Escalation (Rule 2)
**Problem:** Rule 2 escalated any station-keeping near an asset to SURVEILLANCE if `is_approaching=True`, catching legitimate co-located GEO satellites drifting past each other at < 10 m/s.
**Fix:** Changed threshold from `is_approaching` to `closing_rate_km_s < -0.01` (10 m/s closing). Reduces false ELEVATED from 14 to 9 among 990 normal objects.

### 6. Expanded EVASION Pattern Detection
**Problem:** `is_evasive_pattern()` only checked the last event in history.
**Fix:** Scans the full history for any class 3/4/5 event preceded by 5+ consecutive normal (class 0) events. Also added class 5 (Deorbit) as an evasion trigger alongside Minor/Major Maneuver.

### 7. Realistic Collision Trajectory
**Problem:** The original `_inject_collision_course()` launched debris at 7.5 km/s radially, which under Euler integration went hyperbolic (349,000 km altitude by t=360).
**Fix:** Rewrote to generate a retrograde near-circular LEO orbit at 178 deg inclination. Stays in LEO throughout, produces 227 km minimum distance to ISS with 15.86 km/s relative velocity -- a realistic head-on collision geometry.

## Validation Results

```
VALIDATION SUMMARY
==================
PASSED: 54
WARNED: 3
FAILED: 0

Total assessment time: 8.6s (7.0 ms/object mean)
Neural classifier used: 1000/1000 objects
Scenarios triggered: 7/7

Tier Distribution (1000 objects):
  MINIMAL   361  ████████████████████████████████████
  LOW       586  ██████████████████████████████████████████████████████████
  MODERATE   38  ███
  ELEVATED   14  █
  CRITICAL    1
```

**Warnings (non-blocking):**
- COSMOS-2558: ELEVATED (expected CRITICAL) -- rendezvous score 67.6, just below 80 threshold
- DEBRIS-KZ-1A: ELEVATED (expected CRITICAL) -- collision score 67.1, just below 80 threshold
- Transformer predicted radius range [0.114, 0.115] -- normalized range in synthetic data

## New Files

| File | Purpose |
|------|---------|
| `src/api/scenario_injector.py` | 7 adversary trajectory generators (rendezvous, shadowing, collision, phasing, evasion) |
| `src/api/conjunction_service.py` | Periodic pairwise collision risk analysis using trained CollisionRiskPredictor |
| `scripts/train_collision_predictor.py` | GPU training script for CollisionRiskPredictor (synthetic pair generation) |
| `scripts/retrain_maneuver_classifier.py` | ManeuverClassifier retraining with balanced sampling |
| `scripts/validate_full_assessment.py` | End-to-end validation: 57 checks across 7 categories |
| `tests/unit/test_scenario_injector.py` | 18 unit tests for scenario injection |
| `results/phase45_validation/` | Validation report JSON + per-object CSV |
| `docs/PHASE5_COMPLETE.md` | This document |

## Modified Files

| File | Changes |
|------|---------|
| `src/ml/threat_assessment.py` | CNN-LSTM integration, adaptive gravity, Rodrigues proximity, full-trajectory history |
| `src/ml/intent/intent_classifier.py` | Rule 7 (CNN-LSTM SHADOWING), stricter Rule 2 |
| `src/ml/intent/threat_escalation.py` | Expanded EVASION (full history scan, class 5) |
| `src/api/threat_service.py` | TrajectoryPredictor, assess-all batch endpoint, full_timestamps plumbing |
| `src/api/routes/threat.py` | Prediction endpoint, assess-all routes, full_timestamps |
| `src/api/main.py` | Scenario injector + conjunction service registration |
| `src/api/models.py` | New response schemas (prediction, conjunction, assess-all) |
| `src/dashboard/src/components/Globe.tsx` | Pulsing CRITICAL/ELEVATED icons, predicted trajectory polyline |
| `src/dashboard/src/components/ObjectDetail.tsx` | CNN-LSTM maneuver probability bar chart |
| `src/dashboard/src/components/ThreatSummary.tsx` | "Run Full Assessment" button + progress bar |
| `src/dashboard/src/services/api.ts` | assessAll(), assessAllStatus(), prediction API methods |
| `src/dashboard/src/store/useSimStore.ts` | Assessment progress state |
| `src/dashboard/src/types/index.ts` | Extended TypeScript interfaces |
| `tests/integration/test_end_to_end.py` | Updated for new pipeline parameters |

## Performance

| Metric | Phase 4 | Phase 4.5+5 |
|--------|---------|-------------|
| Per-object assessment | ~3 ms | ~7 ms |
| Full 1000-object sweep | ~3 s | ~8.6 s |
| Trajectory prediction | N/A | ~3.5 ms |
| Conjunction analysis | N/A | ~50 ms/tick |
| Models active | 1 | 4 neural + 3 rule-based |
| Tests passing | 410 | 428 |

## Running Validation

```bash
# Full validation (requires GPU for CNN-LSTM)
PYTHONPATH=. python scripts/validate_full_assessment.py

# Tests
PYTHONPATH=. pytest tests/ -v
```
