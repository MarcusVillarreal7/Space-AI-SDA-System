# Anomaly Detection Module

**Date:** 2026-02-07
**Status:** Complete

## Overview

The anomaly detection module uses a reconstruction-based autoencoder to identify unusual satellite behavior. It extracts a 19D behavioral feature vector from maneuver history and orbital state, trains an autoencoder on normal behavior, and flags objects whose reconstruction error exceeds a learned threshold.

## Architecture

```
Maneuver History + Orbital State
        │
        ▼
┌──────────────────────┐
│ BehaviorFeatureExtractor │
│  19D feature vector   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Z-score Normalize   │
│ (training mean/std)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  BehaviorAutoencoder  │
│  Encoder: 19→32→16→6 │
│  Decoder: 6→16→32→19 │
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌──────────┐
│ Latent │  │ Recon.   │
│ (6D)   │  │ Error    │
└────────┘  └────┬─────┘
                 │
                 ▼
         ┌─────────────┐
         │ Threshold    │
         │ (95th %ile)  │
         └──────┬──────┘
                │
                ▼
         ┌─────────────┐
         │AnomalyResult│
         │ score, flag, │
         │ explanation  │
         └─────────────┘
```

## Feature Vector (19D)

| Group | Features | Dim | Description |
|-------|----------|-----|-------------|
| Maneuver Statistics | count, rate, mean_dv, max_dv, dv_variance | 5 | How often and how aggressively the object maneuvers |
| Timing Patterns | mean_interval, regularity | 2 | Temporal spacing of maneuvers (regular = station-keeping) |
| Orbital Characteristics | altitude, eccentricity_proxy, inclination_proxy, speed, accel | 5 | Current orbital state |
| Classification History | dominant_frac, entropy, unique_classes | 3 | Diversity of maneuver types |
| Regime Encoding | LEO, MEO, GEO, HEO one-hot | 4 | Orbital regime |

## Autoencoder Architecture

| Layer | Shape | Activation | Notes |
|-------|-------|------------|-------|
| **Encoder** | | | |
| Linear | 19 → 32 | LeakyReLU | + LayerNorm + Dropout(0.1) |
| Linear | 32 → 16 | LeakyReLU | + LayerNorm + Dropout(0.1) |
| Linear | 16 → 6 | (none) | Latent bottleneck |
| **Decoder** | | | |
| Linear | 6 → 16 | LeakyReLU | + LayerNorm + Dropout(0.1) |
| Linear | 16 → 32 | LeakyReLU | + LayerNorm + Dropout(0.1) |
| Linear | 32 → 19 | (none) | Reconstruction |

**Parameter count:** < 5,000

## Anomaly Detection Pipeline

1. **Feature Extraction**: `BehaviorFeatureExtractor.extract()` → 19D vector
2. **Normalization**: Z-score using training mean/std
3. **Reconstruction**: Autoencoder forward pass
4. **Scoring**: Per-sample MSE between input and reconstruction
5. **Thresholding**: Anomaly if score > 95th percentile of training errors
6. **Explanation**: Top-3 contributing features, human-readable text

## Explainability

The `AnomalyExplainer` identifies which features contributed most to the reconstruction error and maps them to human-readable descriptions:

| Feature | Readable Name |
|---------|--------------|
| maneuver_count | maneuver frequency |
| maneuver_rate | maneuver rate |
| mean_delta_v | average thrust magnitude |
| altitude_km | orbital altitude |
| interval_regularity | timing regularity |
| class_entropy | behavior diversity |

## Files

| File | LOC | Description |
|------|-----|-------------|
| `src/ml/anomaly/__init__.py` | 28 | Module exports |
| `src/ml/anomaly/behavior_features.py` | 230 | 19D feature extraction from tracks |
| `src/ml/anomaly/autoencoder.py` | 133 | Encoder-decoder network |
| `src/ml/anomaly/anomaly_detector.py` | 222 | fit/score/detect/save/load API |
| `src/ml/anomaly/explainer.py` | 95 | Human-readable explanations |
| `tests/unit/test_anomaly_detector.py` | 280 | 37 unit tests, >96% coverage |

## Usage

```python
from src.ml.anomaly import (
    BehaviorFeatureExtractor,
    AnomalyDetector,
    ManeuverRecord,
)

# Extract features
extractor = BehaviorFeatureExtractor()
profile = extractor.extract(
    object_id="SAT-42",
    maneuvers=[
        ManeuverRecord(timestamp=0.0, maneuver_class=0, delta_v_magnitude=0.0),
        ManeuverRecord(timestamp=3600.0, maneuver_class=3, delta_v_magnitude=0.05),
    ],
    position_km=(6800.0, 0.0, 0.0),
    velocity_km_s=(0.0, 7.5, 0.0),
)

# Fit on normal data (100+ profiles recommended)
detector = AnomalyDetector()
metrics = detector.fit(normal_profiles, epochs=50)

# Detect anomalies
result = detector.detect(profile)
print(result.is_anomaly)      # True/False
print(result.anomaly_score)   # reconstruction MSE
print(result.top_features)    # ['maneuver_rate', 'max_delta_v', ...]
print(result.explanation)     # human-readable text
```

## Test Results

**37 tests passed in 5.33s** | Run date: 2026-02-07

### Coverage

| Module | Statements | Missed | Coverage |
|--------|-----------|--------|----------|
| `behavior_features.py` | 108 | 4 | 96% |
| `autoencoder.py` | 65 | 1 | 98% |
| `anomaly_detector.py` | 126 | 2 | 98% |
| `explainer.py` | 21 | 0 | 100% |

### Behavior Features (14 tests)

| Test | Validates | Result |
|------|-----------|--------|
| `test_feature_dim_is_19` | FEATURE_DIM == 19, FEATURE_NAMES length | PASS |
| `test_extract_returns_correct_shape` | Output shape (19,), dtype float32 | PASS |
| `test_extract_metadata` | object_id, num_observations, observation_window | PASS |
| `test_empty_maneuvers` | Empty list → zero features, no crash | PASS |
| `test_maneuver_count_correct` | 2 active maneuvers → count=2 | PASS |
| `test_maneuver_rate_scales_with_frequency` | Faster maneuvers → higher rate | PASS |
| `test_regime_encoding_leo` | 6800 km → LEO one-hot [1,0,0,0] | PASS |
| `test_regime_encoding_geo` | 42164 km → GEO one-hot [0,0,1,0] | PASS |
| `test_altitude_feature` | GEO altitude > LEO altitude | PASS |
| `test_speed_feature` | Speed ≈ 7.5 km/s for LEO | PASS |
| `test_classification_entropy_all_same` | All class-0 → entropy ≈ 0 | PASS |
| `test_classification_entropy_diverse` | All 6 classes → entropy > 0.9 | PASS |
| `test_timing_regularity_perfect` | Equal intervals → regularity ≈ 1.0 | PASS |
| `test_dominant_class_fraction` | 4/5 same class → fraction = 0.8 | PASS |

### Autoencoder (8 tests)

| Test | Validates | Result |
|------|-----------|--------|
| `test_default_config` | input_dim=19, latent_dim=6 | PASS |
| `test_forward_shape` | (8,19) → recon (8,19), latent (8,6) | PASS |
| `test_encode_shape` | (4,19) → latent (4,6) | PASS |
| `test_reconstruction_error_shape` | (4,) non-negative errors | PASS |
| `test_param_count` | Count > 0 and < 10,000 | PASS |
| `test_config_round_trip` | to_dict → from_dict preserves values | PASS |
| `test_from_config` | Build from config dict, forward works | PASS |
| `test_custom_architecture` | hidden=(64,32), latent=8 works | PASS |

### Anomaly Detector (8 tests)

| Test | Validates | Result |
|------|-----------|--------|
| `test_unfitted_raises` | RuntimeError before fit() | PASS |
| `test_fit_returns_metrics` | Returns loss, threshold, param_count, n_training | PASS |
| `test_threshold_is_positive` | threshold > 0 after fit | PASS |
| `test_normal_score_below_threshold` | ≥90% of normal profiles below threshold | PASS |
| `test_anomaly_score_higher` | Anomalous profile scores higher than normal | PASS |
| `test_detect_returns_anomaly_result` | Full AnomalyResult with all fields | PASS |
| `test_detect_normal_is_not_anomaly` | Normal profile not flagged | PASS |
| `test_detect_anomaly_flagged` | Anomalous profile flagged, percentile >90 | PASS |
| `test_score_batch_shape` | Batch scoring returns correct shape | PASS |

### Explainer (4 tests)

| Test | Validates | Result |
|------|-----------|--------|
| `test_top_features_returns_correct_count` | Returns exactly 3 feature names | PASS |
| `test_top_features_order` | Highest-error feature first | PASS |
| `test_explain_anomaly` | Contains "ANOMALOUS", object_id, observation count | PASS |
| `test_explain_normal` | Contains "NORMAL", "expected parameters" | PASS |

### Persistence (2 tests)

| Test | Validates | Result |
|------|-----------|--------|
| `test_save_load_round_trip` | Saved/loaded model produces same scores | PASS |
| `test_save_creates_files` | Creates autoencoder.pt, anomaly_meta.json, training_errors.npy | PASS |

### Full Suite Regression

150 total tests passing (113 pre-existing + 37 new anomaly tests). No regressions.
