# Intent Classification Module

**Date:** 2026-02-07
**Status:** Complete

## Overview

The intent classification module maps the 6 maneuver classes from the CNN-LSTM classifier to 10 operational intent categories, incorporating proximity context and behavioral pattern detection to produce a threat level.

## Architecture

```
Maneuver Class (0-5) + Confidence
        │
        ▼
┌─────────────────┐     ┌──────────────────┐
│  Base Mapping    │     │  Asset Catalog    │
│  class → intent  │     │  6 HV assets     │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│ Threat Escalator │     │ Proximity Context │
│ phasing/shadow/  │◄────│ distance, closing │
│ evasion patterns │     │ rate, TCA, regime │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
              ┌─────────────┐
              │ IntentResult │
              │ intent, threat│
              │ explanation   │
              └──────────────┘
```

## Maneuver → Intent Mapping

| Class | Maneuver | Base Intent | Base Threat |
|-------|----------|-------------|-------------|
| 0 | Normal | NOMINAL | NONE |
| 1 | Drift/Decay | NOMINAL | LOW |
| 2 | Station-keeping | STATION_KEEPING | NONE |
| 3 | Minor Maneuver | ORBIT_MAINTENANCE | LOW |
| 4 | Major Maneuver | ORBIT_RAISING | MODERATE |
| 5 | Deorbit | DEORBIT | LOW |

## Intent Categories (10)

| Intent | Description |
|--------|-------------|
| NOMINAL | Normal operations, no maneuver activity |
| STATION_KEEPING | Routine orbit maintenance |
| ORBIT_MAINTENANCE | Minor adjustment, non-threatening |
| COLLISION_AVOIDANCE | Close approach with fast closing rate |
| DEORBIT | End-of-life disposal burn |
| ORBIT_RAISING | Significant altitude change |
| RENDEZVOUS | Approaching a high-value asset |
| SURVEILLANCE | Shadowing or loitering near an asset |
| EVASIVE | Maneuver suggesting observation avoidance |
| UNKNOWN | Unrecognized maneuver class |

## Threat Levels (5)

| Level | Value | Meaning |
|-------|-------|---------|
| NONE | 0 | No threat |
| LOW | 1 | Minimal concern |
| MODERATE | 2 | Worth monitoring |
| ELEVATED | 3 | Active monitoring required |
| HIGH | 4 | Immediate attention needed |

## Escalation Rules

1. **Minor/Major maneuver + approaching asset** (within warning radius) → RENDEZVOUS, ELEVATED
2. **Very close approach** (within approach threshold) → HIGH
3. **Station-keeping near asset** (same regime) → SURVEILLANCE, ELEVATED
4. **Phasing pattern** (3+ maneuvers in 24h) → SURVEILLANCE/RENDEZVOUS, ELEVATED
5. **Shadowing pattern** (sustained station-keeping near asset) → SURVEILLANCE, HIGH
6. **Evasion pattern** (maneuver after long normal stretch) → EVASIVE, ELEVATED
7. **Low confidence** (<50%) → threat reduced by 1 level

## Asset Catalog

6 simulated high-value assets across LEO/MEO/GEO:
- ISS (LEO, priority 1)
- GPS IIF-01 (MEO, priority 2)
- DSP-23 (GEO, priority 1)
- SBIRS GEO-1 (GEO, priority 1)
- TDRS-13 (GEO, priority 2)
- WGS-10 (GEO, priority 2)

## Files

| File | LOC | Description |
|------|-----|-------------|
| `src/ml/intent/__init__.py` | 25 | Module exports |
| `src/ml/intent/intent_classifier.py` | 220 | Core classifier with escalation |
| `src/ml/intent/proximity_context.py` | 105 | Distance/closing rate/TCA computation |
| `src/ml/intent/threat_escalation.py` | 140 | Pattern detection (phasing/shadowing/evasion) |
| `src/ml/intent/asset_catalog.py` | 130 | High-value asset catalog |
| `tests/unit/test_intent_classifier.py` | 270 | 33 unit tests, >91% coverage |

## Usage

```python
from src.ml.intent import IntentClassifier

classifier = IntentClassifier()
result = classifier.classify(
    maneuver_class=3,             # Minor Maneuver
    confidence=0.85,
    position_km=(6900.0, 0.0, 0.0),
    velocity_km_s=(-1.0, 0.0, 0.0),
)
print(result.intent)        # IntentCategory.RENDEZVOUS
print(result.threat_level)  # ThreatLevel.ELEVATED
print(result.explanation)   # Human-readable explanation
```

## Test Results

```
33 passed in 2.66s
Coverage: intent_classifier 91%, proximity_context 96%, threat_escalation 95%, asset_catalog 98%
```
