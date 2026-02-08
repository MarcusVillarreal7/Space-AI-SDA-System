# Threat Scoring System

**Date:** 2026-02-08
**Status:** Complete

## Overview

The threat scoring system fuses intent classification, anomaly detection, and proximity context into a single 0-100 composite threat score with operational tier assignment and human-readable explanations.

## Architecture

```
┌──────────────────┐   ┌──────────────────┐
│  IntentResult     │   │  AnomalyResult    │
│  intent, threat,  │   │  score, percentile│
│  proximity,       │   │  is_anomaly,      │
│  patterns         │   │  top_features     │
└────────┬─────────┘   └────────┬─────────┘
         │                      │
         ▼                      ▼
┌──────────────────────────────────────────┐
│            ThreatScorer                   │
│                                           │
│  ┌─────────┐ ┌─────────┐ ┌──────────┐   │
│  │ Intent  │ │ Anomaly │ │Proximity │   │
│  │ Sub-    │ │ Sub-    │ │ Sub-     │   │
│  │ score   │ │ score   │ │ score    │   │
│  │ (0-100) │ │ (0-100) │ │ (0-100)  │   │
│  └────┬────┘ └────┬────┘ └────┬─────┘   │
│       │           │           │          │
│  ┌────┴───────────┴───────────┴────┐     │
│  │  Weighted Sum + Pattern Score   │     │
│  │  w_i×I + w_a×A + w_p×P + w_t×T │     │
│  └─────────────┬───────────────────┘     │
│                │                         │
│           clamp [0, 100]                 │
└────────────────┬─────────────────────────┘
                 │
                 ▼
         ┌─────────────┐
         │ ThreatScore  │
         │ score: 0-100 │
         │ tier: 5-level│
         │ explanation  │
         └─────────────┘
```

## Sub-Score Components

### 1. Intent Score (weight: 0.35)

Maps `IntentResult.threat_level` (0-4) to a base score, plus an intent category bonus:

| Threat Level | Base Score |
|-------------|-----------|
| NONE | 0 |
| LOW | 15 |
| MODERATE | 40 |
| ELEVATED | 70 |
| HIGH | 100 |

| Intent Category | Bonus |
|----------------|-------|
| NOMINAL, STATION_KEEPING | +0 |
| ORBIT_MAINTENANCE, DEORBIT | +5 |
| COLLISION_AVOIDANCE, ORBIT_RAISING | +10 |
| UNKNOWN | +15 |
| RENDEZVOUS | +20 |
| SURVEILLANCE | +25 |
| EVASIVE | +30 |

### 2. Anomaly Score (weight: 0.25)

Uses `AnomalyResult.percentile` directly as the 0-100 score. Objects above the anomaly threshold (typically P95) contribute significantly.

### 3. Proximity Score (weight: 0.25)

Combines three proximity components:

| Component | Weight | Formula | Range |
|-----------|--------|---------|-------|
| Distance | 0.50 | `100 × exp(-3 × d / R)` | 100 at d=0, ~5 at d=R, 0 beyond |
| Closing rate | 0.30 | `min(100, |rate| × 10)` | 100 at 10 km/s approach |
| TCA | 0.20 | Log-scale: 100 at ≤10min, 0 at ≥24h | Finite only when approaching |

### 4. Pattern Score (weight: 0.15)

Based on escalation patterns detected in maneuver history:

| Pattern | Score |
|---------|-------|
| PHASING | 40 |
| EVASION | 50 |
| SHADOWING | 60 |

When multiple patterns are detected, the maximum score is used.

## Threat Tiers

| Tier | Score Range | Operational Response |
|------|------------|---------------------|
| MINIMAL | 0-19 | No action required |
| LOW | 20-39 | Log for periodic review |
| MODERATE | 40-59 | Active monitoring recommended |
| ELEVATED | 60-79 | Alert operations team for assessment |
| CRITICAL | 80-100 | Immediate response required |

## Configurable Weights

Weights can be customized per operational context:

```python
from src.ml.threat import ThreatScorer, ScoringWeights

# Default: balanced
scorer = ThreatScorer()

# Intent-heavy (trust classification more)
scorer = ThreatScorer(weights=ScoringWeights(
    intent=0.5, anomaly=0.2, proximity=0.2, pattern=0.1
))

# Proximity-heavy (focus on asset protection)
scorer = ThreatScorer(weights=ScoringWeights(
    intent=0.2, anomaly=0.2, proximity=0.5, pattern=0.1
))
```

## Files

| File | LOC | Description |
|------|-----|-------------|
| `src/ml/threat/__init__.py` | 18 | Module exports |
| `src/ml/threat/threat_scorer.py` | 260 | Core scorer with 4 sub-scores |
| `src/ml/threat/threat_explainer.py` | 70 | Human-readable explanations |
| `tests/unit/test_threat_scorer.py` | 320 | 38 unit tests, 100% coverage |

## Usage

```python
from src.ml.threat import ThreatScorer
from src.ml.intent import IntentClassifier
from src.ml.anomaly import AnomalyDetector

# Setup
intent_classifier = IntentClassifier()
anomaly_detector = AnomalyDetector.load("checkpoints/phase3_anomaly")
threat_scorer = ThreatScorer()

# Classify intent
intent_result = intent_classifier.classify(
    maneuver_class=4, confidence=0.9,
    position_km=(6800.0, 0.0, 0.0),
    velocity_km_s=(-1.0, 0.0, 0.0),
)

# Detect anomaly (requires BehaviorProfile)
anomaly_result = anomaly_detector.detect(profile)

# Compute threat score
threat = threat_scorer.score(
    object_id="SAT-42",
    intent_result=intent_result,
    anomaly_result=anomaly_result,
)

print(threat.score)         # 72.3
print(threat.tier)          # ThreatTier.ELEVATED
print(threat.explanation)   # Multi-line operational explanation
```

## Test Results

**38 tests passed in 1.70s** | Run date: 2026-02-08

### Coverage

| Module | Statements | Missed | Coverage |
|--------|-----------|--------|----------|
| `threat_scorer.py` | 125 | 0 | 100% |
| `threat_explainer.py` | 26 | 0 | 100% |

### Threat Tier Assignment (5 tests)

| Test | Validates | Result |
|------|-----------|--------|
| `test_minimal` | 0-19.9 → MINIMAL | PASS |
| `test_low` | 20-39.9 → LOW | PASS |
| `test_moderate` | 40-59.9 → MODERATE | PASS |
| `test_elevated` | 60-79.9 → ELEVATED | PASS |
| `test_critical` | 80-100 → CRITICAL | PASS |

### Proximity Sub-score Functions (8 tests)

| Test | Validates | Result |
|------|-----------|--------|
| `test_distance_score_at_zero` | d=0 → score=100 | PASS |
| `test_distance_score_at_warning_radius` | d=R → score<10 | PASS |
| `test_distance_score_beyond_warning` | d>R → score=0 | PASS |
| `test_distance_score_monotonic` | Closer = higher score | PASS |
| `test_closing_score_receding` | Moving away → score=0 | PASS |
| `test_closing_score_approaching` | -5 km/s → score=50 | PASS |
| `test_closing_score_fast_approach` | -10 km/s → score=100 | PASS |
| `test_closing_score_clamped` | -20 km/s → clamped at 100 | PASS |
| `test_tca_very_short` | 5 min → score=100 | PASS |
| `test_tca_long` | 24h → score=0 | PASS |
| `test_tca_infinite` | inf → score=0 | PASS |
| `test_tca_moderate` | 1h → 0 < score < 100 | PASS |

### Scoring Weights (2 tests)

| Test | Validates | Result |
|------|-----------|--------|
| `test_default_weights_sum` | Default weights sum to 1.0 | PASS |
| `test_custom_weights` | Custom weights sum correctly | PASS |

### ThreatScorer Integration (11 tests)

| Test | Validates | Result |
|------|-----------|--------|
| `test_no_inputs_returns_zero` | No inputs → score=0, MINIMAL | PASS |
| `test_nominal_intent_low_score` | NOMINAL/NONE → score<20 | PASS |
| `test_high_threat_intent` | RENDEZVOUS/HIGH + close proximity → MODERATE+ | PASS |
| `test_anomaly_only` | Anomaly P98 alone → score>20 | PASS |
| `test_combined_intent_and_anomaly` | SURVEILLANCE + SHADOWING + anomaly → ELEVATED+ | PASS |
| `test_pattern_scores` | PHASING+EVASION → pattern_score≥40 | PASS |
| `test_score_clamped_to_100` | Extreme weights → clamped ≤100 | PASS |
| `test_score_never_negative` | Empty inputs → score≥0 | PASS |
| `test_custom_weights_change_score` | Intent-heavy vs anomaly-heavy produces different scores | PASS |
| `test_proximity_contributes_when_close` | Close proximity → higher score than far | PASS |
| `test_result_has_all_fields` | All ThreatScore fields populated | PASS |

### Explainer (3 tests)

| Test | Validates | Result |
|------|-----------|--------|
| `test_minimal_threat` | Contains object_id, score, tier, "No action" | PASS |
| `test_critical_threat_has_factors` | Contains "CRITICAL", "Immediate response", factors | PASS |
| `test_explanation_shows_sub_scores` | Contains intent/anomaly/proximity sub-scores | PASS |

### Scenario Tests (5 tests)

| Test | Scenario | Expected | Result |
|------|----------|----------|--------|
| `test_normal_satellite` | NOMINAL, no anomaly | MINIMAL, <20 | PASS |
| `test_station_keeping_near_asset` | SK near ISS, no anomaly | <40 | PASS |
| `test_surveillance_with_anomaly` | SURVEILLANCE + SHADOWING + P99 anomaly | ELEVATED+ | PASS |
| `test_fast_approach_unknown_intent` | COLLISION_AVOIDANCE, 20km, -8 km/s | MODERATE+ | PASS |
| `test_deorbiting_satellite` | DEORBIT, far from assets | MINIMAL/LOW | PASS |

### Full Suite Regression

188 total tests passing (150 pre-existing + 38 new threat tests). No regressions.
