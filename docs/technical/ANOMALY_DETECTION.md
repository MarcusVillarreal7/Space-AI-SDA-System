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

---

## Training Results (Day 4)

**Date:** 2026-02-08 | **Device:** NVIDIA RTX 4080 Laptop GPU (12.9 GB VRAM)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Training data | 1000 satellite profiles from ground truth simulation |
| Data loading | 10 chunks of 100 objects (93 MB parquet) |
| Epochs | 100 |
| Batch size | 128 |
| Learning rate | 1e-3 (Adam) |
| Threshold percentile | 95th |
| Training time | 4.5 seconds on GPU |

### Profile Extraction

Behavior profiles were extracted from `data/processed/ml_train_1k/ground_truth.parquet` (1000 objects, 1440 timesteps each, 60-second intervals over 24 hours). For each object:

1. Compute per-timestep delta-V from velocity differences
2. Classify each timestep into maneuver class 0-5 using heuristic thresholds
3. Build ManeuverRecord list
4. Extract 19D behavior profile

Delta-V classification thresholds:

| Range (km/s) | Maneuver Class |
|--------------|---------------|
| 0 - 0.005 | 0 (Normal) |
| 0.005 - 0.01 | 1 (Drift/Decay) |
| 0.01 - 0.02 | 2 (Station-keeping) |
| 0.02 - 0.1 | 3 (Minor Maneuver) |
| 0.1 - 1.0 | 4 (Major Maneuver) |
| > 1.0 | 5 (Deorbit) |

### Training Convergence

```
Epoch  10/100  loss=0.695085
Epoch  20/100  loss=0.523131
Epoch  30/100  loss=0.461917
Epoch  40/100  loss=0.384468
Epoch  50/100  loss=0.335095
Epoch  60/100  loss=0.255285
Epoch  70/100  loss=0.217566
Epoch  80/100  loss=0.180684
Epoch  90/100  loss=0.168925
Epoch 100/100  loss=0.153166
```

### Training Error Distribution

| Percentile | Reconstruction Error |
|-----------|---------------------|
| P50 | 0.013088 |
| P75 | 0.048624 |
| P90 | 0.091523 |
| P95 (threshold) | 0.129128 |
| P99 | 3.885675 |
| Max | 6.472743 |

### Synthetic Anomaly Validation

50 synthetic anomalous profiles were generated across 4 types:

| Type | Description | Count |
|------|-------------|-------|
| Burst maneuvers | Extreme maneuver count/rate (count≥15, rate≥20/day) | 13 |
| High delta-V | Abnormally large thrust (mean=2.0, max=5.0 km/s) | 13 |
| Regime mismatch | GEO position with LEO-like maneuver activity | 12 |
| Erratic timing | Low regularity + high behavioral entropy | 12 |

### Final Detection Results

| Metric | Value |
|--------|-------|
| True Positive Rate (TPR) | **100% (50/50)** |
| False Positive Rate (FPR) | 5.0% (50/1000) |
| Normal score mean ± std | 0.110 ± 0.576 |
| Anomaly score mean ± std | 83,392 ± 133,668 |
| Score separation ratio | 759,178x |
| Min anomaly score | 10.83 |
| Max normal score | 6.47 |
| Clean separation gap | **4.36** |

### Per-Type Detection

| Type | Detected | Rate | Mean Score |
|------|----------|------|------------|
| Burst maneuvers | 13/13 | 100% | 160,332 |
| High delta-V | 13/13 | 100% | 68,740 |
| Regime mismatch | 12/12 | 100% | 49,656 |
| Erratic timing | 12/12 | 100% | 49,652 |

### Configurable Threshold

The FPR is directly controlled by the threshold percentile. All anomaly types remain detectable at higher thresholds:

| Percentile | Threshold | FPR | Min Anomaly Score | Detected |
|-----------|-----------|-----|-------------------|----------|
| P95 | 0.129 | 5.0% | 10.83 | 50/50 |
| P99 | 3.886 | 1.0% | 10.83 | 50/50 |
| P99.5 | 4.686 | 0.5% | 10.83 | 50/50 |

### Checkpoint

```
checkpoints/phase3_anomaly/
├── autoencoder.pt          (18 KB - model weights)
├── anomaly_meta.json       (1.2 KB - config, threshold, normalization stats)
├── training_errors.npy     (4.1 KB - training error distribution)
└── training_summary.json   (870 B - full training metrics)
```

---

## GEO Zero-Activity Bug

### Discovery

During initial training evaluation, 8 out of 50 synthetic anomalies were missed (84% TPR instead of expected ~100%). Investigation revealed all 8 missed anomalies shared a common pattern:

- All based on GEO satellite profiles (~35,786 km altitude)
- All had `maneuver_count=0`, `maneuver_rate=0`, `mean_delta_v=0`
- Anomaly scores were extremely low (0.001 - 0.073), well below the 0.129 threshold

### Root Cause

The synthetic anomaly generator perturbed individual features in isolation using multiplicative scaling or direct overwrites. However, behavioral features are **correlated** — for GEO satellites with zero maneuver activity, many features were already 0.0:

```
maneuver_count = 0.0
maneuver_rate  = 0.0
mean_delta_v   = 0.0
max_delta_v    = 0.0
mean_interval  = 0.0
regularity     = 0.0
```

Perturbations like `count *= 10` (type 0) produced `0 * 10 = 0`, and perturbations like "set regularity to 0.05" modified features that were irrelevant when maneuver count was zero. The resulting "anomalous" profiles were nearly identical to normal GEO profiles, and the model correctly scored them as normal.

### Missed Anomaly Details (Before Fix)

| ID | Type | Score | Threshold | maneuver_count | altitude |
|----|------|-------|-----------|---------------|----------|
| ANOMALY-003 | Erratic timing | 0.0016 | 0.1291 | 0.0 | 35,786 km |
| ANOMALY-005 | High delta-V | 0.0016 | 0.1291 | 0.0* | 35,786 km |
| ANOMALY-006 | Regime mismatch | 0.0017 | 0.1291 | 0.0* | 35,786 km |
| ANOMALY-007 | Erratic timing | 0.0079 | 0.1291 | 0.0 | 35,800 km |
| ANOMALY-017 | High delta-V | 0.0030 | 0.1291 | 0.0 | 35,756 km |
| ANOMALY-028 | Burst | 0.0730 | 0.1291 | 0.0 | 35,746 km |
| ANOMALY-033 | High delta-V | 0.0015 | 0.1291 | 0.0 | 35,806 km |
| ANOMALY-043 | Erratic timing | 0.0015 | 0.1291 | 0.0 | 35,797 km |

*\*Some types set count to a non-zero value but other correlated features remained zero, producing an incoherent profile that the autoencoder still reconstructed easily.*

### Solution

Replaced isolated feature perturbations with **coherent multi-feature signatures** that ensure all correlated features are set together. Each anomaly type now guarantees a non-zero maneuver baseline:

| Type | Before (Broken) | After (Fixed) |
|------|-----------------|---------------|
| **Burst** | `count *= 10` | `count = max(count*10, 15)`, `rate = max(rate*10, 20)`, `mean_dv ≥ 0.05`, `interval ≥ 600s` |
| **High delta-V** | Set dv only | Also set `count ≥ 5`, `rate ≥ 5/day`, `interval ≥ 1800s`, `regularity = 0.5` |
| **Regime mismatch** | Set regime + altitude | Also set `count = 20`, `rate = 15`, `mean_dv = 0.1`, `max_dv = 0.5`, `interval = 1200s` |
| **Erratic timing** | Set timing features | Also set `count ≥ 8`, `rate ≥ 10/day`, `mean_dv ≥ 0.03`, `max_dv ≥ 0.2` |

### Results After Fix

All 8 previously-missed anomalies are now detected with strong scores:

| ID | Type | Score Before | Score After | Status |
|----|------|-------------|-------------|--------|
| ANOMALY-003 | Erratic timing | 0.0016 | **18.50** | DETECTED |
| ANOMALY-005 | High delta-V | 0.0016 | **76.22** | DETECTED |
| ANOMALY-006 | Regime mismatch | 0.0017 | **76.22** | DETECTED |
| ANOMALY-007 | Erratic timing | 0.0079 | **18.42** | DETECTED |
| ANOMALY-017 | High delta-V | 0.0030 | **18.45** | DETECTED |
| ANOMALY-028 | Burst | 0.0730 | **18.32** | DETECTED |
| ANOMALY-033 | High delta-V | 0.0015 | **18.50** | DETECTED |
| ANOMALY-043 | Erratic timing | 0.0015 | **18.50** | DETECTED |

### Impact

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| TPR | 84% (42/50) | **100% (50/50)** |
| FPR | 5.0% | 5.0% (unchanged) |
| Min anomaly score | 0.001 (overlapping with normal) | **10.83** (clean gap) |
| Max normal score | 6.47 | 6.47 (unchanged) |

### Lesson Learned

Synthetic anomaly generation for multi-dimensional feature vectors must maintain **internal consistency** across correlated features. Perturbing individual features in isolation can produce incoherent profiles that fall within the normal distribution. Each anomaly type should define a **complete behavioral signature** spanning all feature groups that contribute to that anomaly pattern.
