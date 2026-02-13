# Phase 5.5: Mixed Object Catalog, Co-Orbital Fix, Enhanced Explanations

**Date**: 2026-02-11 — 2026-02-12
**Result**: Realistic mixed catalog, false-positive elimination, contextual threat explanations

## Problem

The original catalog contained 1,000 active satellites from CelesTrak's `active.tle`. Every object was treated as a maneuverable payload and run through the full 6-step threat pipeline. This is unrealistic — real SDA catalogs are roughly 37% active satellites and 63% debris + rocket bodies. Running CNN-LSTM classification and intent analysis on debris produces nonsensical results.

Additionally, high-speed LEO flybys (e.g., a university CubeSat passing the ISS at 4.5 km/s) were being falsely classified as SURVEILLANCE because the escalation rules only checked distance and regime, not relative velocity.

## Changes

### 1. Mixed Object Catalog

**Script**: `scripts/build_mixed_catalog.py`

Built a realistic 1,000-object catalog from multiple CelesTrak TLE sources:
- 600 PAYLOAD (active satellites)
- 350 DEBRIS (from Cosmos-2251, Fengyun-1C, Iridium-33, Indian ASAT debris clouds)
- 50 ROCKET_BODY (identified by "R/B" in TLE name)

`object_type` is stored as a parquet column and flows through the entire stack: SpaceCatalog → API routes → WebSocket broadcast → frontend.

### 2. Type-Aware Threat Pipeline

**File**: `src/ml/threat_assessment.py`

Added `assess_by_type()` dispatcher:
- **PAYLOAD** → full 6-step pipeline (maneuver classification, intent, anomaly, scoring)
- **DEBRIS / ROCKET_BODY** → collision-only path (proximity scoring only, skips CNN-LSTM/intent/anomaly)
- Rocket bodies get +5 breakup risk bonus (pressurized fuel tanks)

This eliminated nonsensical maneuver classifications for non-maneuverable objects and improved assess-all throughput by ~40%.

### 3. Co-Orbital False Positive Fix

**Problem**: QB50P1 (a university CubeSat) was flagged as CRITICAL/SURVEILLANCE when it flew past the ISS at 4.53 km/s relative velocity. That's a brief flyby, not surveillance.

**Root cause**: Escalation rules (Rules 2, 7, and `is_shadowing_pattern`) only checked distance and regime match, not whether objects were actually co-moving.

**Fix**:
- Added `relative_speed_km_s` and `is_coorbital` (threshold: 1.0 km/s) to `ProximityContext`
- Gated SURVEILLANCE/SHADOWING rules on `is_coorbital = True`
- High-speed flybys now classified as COLLISION_AVOIDANCE (if close) or NOMINAL
- True co-orbital threats (e.g., LUCH/OLYMP at GEO with ~0.01 km/s relative velocity) still detected correctly

**Files modified**:
- `src/ml/intent/proximity_context.py` — added fields + computation
- `src/ml/intent/intent_classifier.py` — gated Rules 2, 7
- `src/ml/intent/threat_escalation.py` — gated `is_shadowing_pattern()`

### 4. Contextual Threat Explanations

**Problem**: All objects showed the same generic explanation text. Operators couldn't understand WHY something was or wasn't a threat.

**Fix**: Rewrote `ThreatExplainer` to generate contextual explanations:
- **Behavioral context**: "This satellite is currently performing routine station-keeping burns..."
- **Threat reasoning** (ELEVATED/CRITICAL): "This object is maintaining a persistent standoff position near a protected asset..."
- **Benign justification** (MINIMAL/LOW): "Assessment basis: no proximity to protected assets; no hostile intent indicators..."
- **Passive objects**: Distance-based messaging with action recommendations

### 5. Frontend Updates

- Globe: dot size varies by type (PAYLOAD=4, DEBRIS=2, ROCKET_BODY=3)
- TrackingTab: type filter pills (All / Satellite / Debris / Rocket Body), type column
- ObjectDetail: type badge, conditional CNN-LSTM chart (PAYLOAD only), "collision-only" note for passive objects, multi-line Analysis panel
- AboutTab: object type composition breakdown

## Issues Encountered

### Browser cache masking backend changes
After deploying the mixed catalog, the tracking tab showed 999 satellites and 1 debris despite the backend correctly returning 600/350/50. Root cause: browser was caching old API responses. Fixed with hard refresh (Ctrl+Shift+R).

### Gravity subtraction residuals
The scenario injector uses forward Euler integration for trajectory propagation. Applying trapezoidal gravity subtraction to Euler-integrated data produced ~16 m/s residuals (classified as station-keeping), masking real maneuver events. Fixed with adaptive method: compute both Euler and trapezoidal residuals, pick the smaller one per timestep.

## Validation

- 443 tests passing
- 54 PASS / 3 WARN / 0 FAIL validation
- 7/7 threat scenarios still trigger at ELEVATED or CRITICAL
- High-speed flyby false positives eliminated
- Co-orbital shadowing correctly detected
