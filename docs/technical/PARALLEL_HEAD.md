# Phase 3: Parallel Prediction Head — Results Summary

**Date:** 2026-02-07
**Status:** Complete

## Problem

The scaled Transformer (trained on 1.4M sequences) suffered from a severe autoregressive (AR) train/inference gap:

| Mode | Position RMSE | Velocity RMSE |
|------|--------------|---------------|
| Teacher Forcing (training) | 5.39 | 1.09 |
| Autoregressive (inference) | 14.94 | 2.24 |
| **Degradation** | **+177%** | **+106%** |

**Root cause:** During training, the decoder sees ground-truth targets for all 30 timesteps (teacher forcing). During inference, it auto-regressively generates one timestep at a time using its own imperfect predictions, causing error accumulation over 30 steps.

## Solution

Added a DETR-style **parallel prediction head** with learned query tokens. The head produces all 30 future timesteps simultaneously via cross-attention to the encoder output. Training and inference use the identical code path — no mismatch by construction.

### Architecture

```
History (20 steps) → Encoder → [encoder output]
                                      ↓
                      30 Learned Query Tokens + Positional Encoding
                                      ↓
                      2x Cross-Attention Decoder Layers
                                      ↓
                      Output Projection → 30 Predictions (pos + vel)
```

### Parameters

| Component | Params |
|-----------|--------|
| Original model (encoder + AR decoder) | 235,462 |
| Parallel prediction head | 135,814 |
| **Total** | **371,276** |

## Training

- **Data:** 1,112,800 train / 278,200 val sequences (from 1.4M total)
- **Pretrained encoder:** Loaded from `checkpoints/phase3_scaled/best_model.pt`
- **Phased training:**
  - Epochs 1-3: Encoder frozen, head-only training (LR = 3e-4)
  - Epochs 4-10: Full fine-tuning, encoder LR x0.1 (differential)
- **Hardware:** NVIDIA RTX 4080 Laptop GPU
- **Time:** ~13 minutes (10 epochs)
- **Best val loss:** 37.93 (epoch 9)

### Training Curve

| Epoch | Phase | Train Loss | Val Loss |
|-------|-------|-----------|----------|
| 1 | Frozen | 43.14 | 37.97 |
| 2 | Frozen | 42.47 | 37.95 |
| 3 | Frozen | 42.39 | 37.94 |
| 4 | Full | 41.21 | 37.98 |
| 5 | Full | 39.67 | 37.94 |
| 7 | Full | 39.02 | 37.94 |
| 9 | Full | 38.89 | **37.93** |
| 10 | Full | 38.86 | 37.94 |

## Evaluation Results

Evaluated on 5,000 held-out validation sequences (20-step history, 30-step prediction horizon).

### Position Metrics

| Model | RMSE | MAE | Mean 3D Error | Median 3D Error | P95 3D Error |
|-------|------|-----|---------------|-----------------|-------------|
| Baseline (AR) | 7.61 | 4.96 | 11.24 | 10.81 | 21.88 |
| Scaled (TF) | 5.39 | 4.12 | 8.09 | 6.70 | 20.29 |
| Scaled (AR) | 14.94 | 13.58 | 25.10 | 24.78 | 36.41 |
| **Parallel** | **7.57** | **4.64** | **11.07** | **10.63** | **21.84** |

### Velocity Metrics

| Model | RMSE | MAE | Mean 3D Error |
|-------|------|-----|---------------|
| Baseline (AR) | 1.13 | 0.90 | 1.80 |
| Scaled (TF) | 1.09 | 0.86 | 1.77 |
| Scaled (AR) | 2.24 | 1.71 | 3.61 |
| **Parallel** | **1.00** | **0.71** | **1.58** |

### Key Improvements (Parallel vs Scaled AR)

| Metric | Scaled AR | Parallel | Reduction |
|--------|-----------|----------|-----------|
| Position RMSE | 14.94 | 7.57 | **-49.4%** |
| Velocity RMSE | 2.24 | 1.00 | **-55.2%** |
| Position MAE | 13.58 | 4.64 | **-65.8%** |
| Mean 3D Pos Error | 25.10 | 11.07 | **-55.9%** |

### Per-Timestep Stability

The parallel head shows **flat error across all 30 timesteps** (no error accumulation):
- Position RMSE t=1: 7.57, t=15: 7.57, t=30: 7.56
- Velocity RMSE t=1: 1.00, t=15: 1.00, t=30: 1.00

In contrast, the scaled AR model shows severe degradation over time:
- Position RMSE t=1: 8.47, t=15: 15.48, t=30: 16.32

## Files

| File | Description |
|------|-------------|
| `src/ml/models/trajectory_transformer.py` | `ParallelPredictionHead` class, `forward_parallel()`, updated config/predict |
| `src/ml/training/trainer.py` | Fixed scheduler bug (ReduceLROnPlateau vs CosineAnnealingLR) |
| `scripts/train_trajectory_parallel.py` | Training script with PreloadedDataset, phased training |
| `scripts/evaluate_ml_comparison.py` | Evaluation with parallel model comparison |
| `results/phase3_evaluation/evaluation_report.json` | Full metrics report |
| `checkpoints/phase3_parallel/best_model.pt` | Best model checkpoint |

## Backward Compatibility

- All 80 existing tests pass
- `predict()` API unchanged — auto-dispatches to parallel head when available, falls back to AR otherwise
- `get_config()` / `from_config()` roundtrip preserves new config fields
- Old checkpoints (without parallel head) load and work identically
