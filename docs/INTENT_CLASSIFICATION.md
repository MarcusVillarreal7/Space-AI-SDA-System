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

**33 tests passed in 2.66s** | Run date: 2026-02-07

### Coverage

| Module | Statements | Missed | Coverage |
|--------|-----------|--------|----------|
| `intent_classifier.py` | 90 | 8 | 91% |
| `proximity_context.py` | 52 | 2 | 96% |
| `threat_escalation.py` | 57 | 3 | 95% |
| `asset_catalog.py` | 44 | 1 | 98% |

### Asset Catalog (6 tests)

| Test | Validates | Result |
|------|-----------|--------|
| `test_default_catalog_non_empty` | Default catalog has >=4 assets | PASS |
| `test_by_regime` | GEO filter returns >=2, all GEO | PASS |
| `test_by_priority` | Priority filter respects max_priority | PASS |
| `test_nearest` | Position near ISS orbit returns ISS | PASS |
| `test_within_range` | 0 assets at origin, all at 100K km | PASS |
| `test_empty_catalog` | Empty catalog returns None safely | PASS |

### Proximity Context (8 tests)

| Test | Validates | Result |
|------|-----------|--------|
| `test_classify_regime_leo` | 6771 km altitude -> LEO | PASS |
| `test_classify_regime_meo` | 26560 km altitude -> MEO | PASS |
| `test_classify_regime_geo` | 42164 km altitude -> GEO | PASS |
| `test_approaching_object` | Negative closing rate, correct distance (~129 km) | PASS |
| `test_receding_object` | Positive closing rate when moving away | PASS |
| `test_tca_finite_when_approaching` | TCA > 0 and < inf for approaching object | PASS |
| `test_tca_infinite_when_receding` | TCA = inf for receding object | PASS |
| `test_no_catalog` | Empty catalog returns inf distance, None asset | PASS |

### Threat Escalation Patterns (8 tests)

| Test | Validates | Result |
|------|-----------|--------|
| `test_phasing_detected` | 4 minor maneuvers in window -> phasing | PASS |
| `test_phasing_not_detected_too_few` | 2 maneuvers -> no phasing | PASS |
| `test_phasing_ignores_normal` | All-normal history -> no phasing | PASS |
| `test_shadowing_detected` | SK near asset for 14h -> shadowing | PASS |
| `test_shadowing_not_detected_far` | SK at 5000 km -> no shadowing | PASS |
| `test_evasive_detected` | 5 normal + 1 major -> evasion | PASS |
| `test_evasive_not_detected_short` | Too-short history -> no evasion | PASS |
| `test_detect_patterns_combined` | Multiple patterns detected together | PASS |

### Intent Classifier (11 tests)

| Test | Validates | Result |
|------|-----------|--------|
| `test_normal_class_no_threat` | Class 0 -> NOMINAL, NONE | PASS |
| `test_station_keeping_base` | Class 2 far from assets -> SK, NONE | PASS |
| `test_deorbit` | Class 5 -> DEORBIT, LOW | PASS |
| `test_minor_maneuver_approaching_asset_escalates` | Class 3 approaching ISS -> RENDEZVOUS, >=ELEVATED | PASS |
| `test_major_maneuver_very_close_is_high_threat` | Class 4 at ~29 km from ISS -> HIGH | PASS |
| `test_low_confidence_reduces_threat` | 30% confidence reduces threat by 1 level | PASS |
| `test_phasing_escalation` | Phasing history -> PHASING pattern, >=ELEVATED | PASS |
| `test_explanation_contains_class_name` | Output includes "Major Maneuver" and "90%" | PASS |
| `test_explanation_contains_asset` | Output includes "International Space Station" | PASS |
| `test_unknown_maneuver_class` | Class 99 -> UNKNOWN intent | PASS |
| `test_all_base_mappings` | All classes 0-5 produce valid IntentResult | PASS |

### Full Suite Regression

113 total tests passing (80 pre-existing + 33 new intent tests). No regressions.
