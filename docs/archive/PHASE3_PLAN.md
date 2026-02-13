# 📋 Phase 3: ML Prediction & Classification - Implementation Plan

**Created**: 2026-02-07  
**Status**: 🔄 RESTORATION IN PROGRESS  
**Priority**: CRITICAL

---

## 🎯 Phase 3 Objectives

Build production-grade ML models for:
1. **Trajectory Prediction**: Forecast satellite positions 1-24 hours ahead
2. **Maneuver Classification**: Detect and classify orbital maneuvers
3. **Uncertainty Quantification**: Provide calibrated confidence intervals
4. **Operational Integration**: CLI tools for inference and evaluation

---

## 📊 Current Status (Post-Crash Recovery)

### ✅ What Survived
- **Feature Engineering**: trajectory_features.py, sequence_builder.py
- **Trained Checkpoints**: 
  - Trajectory Predictor (14 epochs, best val loss: 464M)
  - Maneuver Classifier (20 epochs, 84.5% val acc)
- **Training Data**: 15 GB processed features in chunks
- **Feature Extraction Script**: extract_features_1k_chunked.py

### ❌ What Was Lost
- Model architectures (trajectory_transformer.py, maneuver_classifier.py, collision_predictor.py)
- Training infrastructure (trainer.py, losses.py)
- Uncertainty quantification (monte_carlo.py, ensemble.py, conformal.py)
- Inference pipeline (inference.py)
- Tests and additional documentation

---

## 🏗️ Architecture Overview

### Model 1: Trajectory Transformer (Seq2Seq Predictor)

**Architecture** (Recovered from checkpoint):
```
Input: (batch, seq_len=20, features=24)
  ↓
Input Projection: 24 → 64
  ↓
Positional Encoding (sine/cosine)
  ↓
Transformer Encoder (2 layers)
  ├─ Multi-Head Attention (4 heads, d_model=64)
  ├─ Feed-Forward (64 → 256 → 64)
  └─ Layer Norm + Residual
  ↓
Transformer Decoder (2 layers)
  ├─ Masked Self-Attention
  ├─ Cross-Attention to Encoder
  ├─ Feed-Forward (64 → 256 → 64)
  └─ Layer Norm + Residual
  ↓
Output Projection: 64 → 6 (position + velocity)
  ↓
Output: (batch, pred_horizon=30, features=6)
```

**Hyperparameters**:
- d_model: 64
- n_heads: 4
- n_encoder_layers: 2
- n_decoder_layers: 2
- d_ff: 256
- dropout: 0.1
- input_dim: 24
- output_dim: 6

**Training Config**:
- Optimizer: AdamW
- Initial LR: 1e-4
- LR Schedule: Cosine annealing
- Batch size: ~11 (inferred from dataset)
- Epochs: 14 (best at epoch 8)
- Best val loss: 464,988,608

**Performance**:
- Position RMSE: 2.95 km
- Velocity RMSE: 3.27 km/s
- Position MAE: 2.37 km

### Model 2: Maneuver Classifier

**Architecture** (To be reverse-engineered):
```
Input: (batch, seq_len, features) or (batch, features)
  ↓
Feature Aggregation (mean/max pooling or LSTM)
  ↓
MLP Layers:
  ├─ Linear(input → 256)
  ├─ ReLU + Dropout(0.3)
  ├─ Linear(256 → 128)
  ├─ ReLU + Dropout(0.3)
  ├─ Linear(128 → 64)
  ├─ ReLU + Dropout(0.3)
  └─ Linear(64 → 4)
  ↓
Softmax → 4 classes
```

**Classes**:
1. Normal orbit
2. Drift/decay
3. Station-keeping
4. Active maneuver

**Training Config**:
- Optimizer: Adam
- Epochs: 20
- Best val accuracy: 84.5%
- Loss: CrossEntropyLoss

---

## 📦 Deliverables

### Core Modules (9 files)

| File | LOC | Status | Description |
|------|-----|--------|-------------|
| `models/trajectory_transformer.py` | 400 | 🔄 In progress | Seq2seq transformer for prediction |
| `models/maneuver_classifier.py` | 200 | ⏳ Pending | Behavior classification |
| `models/collision_predictor.py` | 200 | ⏳ Pending | Close approach detection |
| `training/trainer.py` | 350 | ⏳ Pending | Generic training loop |
| `training/losses.py` | 150 | ⏳ Pending | Custom loss functions |
| `uncertainty/monte_carlo.py` | 150 | ⏳ Pending | MC Dropout uncertainty |
| `uncertainty/ensemble.py` | 200 | ⏳ Pending | Ensemble methods |
| `uncertainty/conformal.py` | 150 | ⏳ Pending | Conformal prediction |
| `inference.py` | 300 | ⏳ Pending | Production inference |

### CLI Scripts (3 files)

| Script | LOC | Status | Description |
|--------|-----|--------|-------------|
| `train_trajectory_predictor.py` | 200 | ⏳ Pending | Train trajectory model |
| `train_maneuver_classifier.py` | 200 | ⏳ Pending | Train classifier |
| `evaluate_ml_models.py` | 400 | ⏳ Pending | Comprehensive evaluation |

### Tests (2 files)

| Test File | Tests | Status | Description |
|-----------|-------|--------|-------------|
| `test_ml_models.py` | 20+ | ⏳ Pending | Model architecture tests |
| `test_ml_features.py` | 15+ | ⏳ Pending | Feature extraction tests |

### Documentation (3 files)

| Document | Status | Description |
|----------|--------|-------------|
| `PHASE3_PLAN.md` | ✅ Done | This file |
| `PHASE3_PROGRESS.md` | ⏳ Pending | Daily progress log |
| `PHASE3_COMPLETE.md` | ⏳ Pending | Completion report |

---

## 🚀 Implementation Timeline

### Day 1: Critical Path Restoration
- [x] Analyze checkpoints and recover architecture
- [ ] Implement trajectory_transformer.py
- [ ] Test checkpoint loading
- [ ] Verify predictions match expected performance
- [ ] Git commit: "Restore trajectory transformer model"

### Day 2: Classification & Inference
- [ ] Implement maneuver_classifier.py
- [ ] Test checkpoint loading
- [ ] Implement inference.py
- [ ] End-to-end prediction demo
- [ ] Git commit: "Restore classifier and inference pipeline"

### Day 3: Training Infrastructure
- [ ] Implement trainer.py
- [ ] Implement losses.py
- [ ] Create training scripts
- [ ] Git commit: "Restore training infrastructure"

### Day 4: Uncertainty & Advanced Features
- [ ] Implement uncertainty quantification modules
- [ ] Implement collision_predictor.py
- [ ] Git commit: "Add uncertainty quantification and collision prediction"

### Day 5: Testing & Documentation
- [ ] Implement comprehensive tests
- [ ] Complete evaluation script
- [ ] Update documentation
- [ ] Git commit: "Complete Phase 3 testing and documentation"

---

## ✅ Success Criteria

| Criterion | Target | Verification |
|-----------|--------|--------------|
| Checkpoint Loading | No errors | Load both models successfully |
| Trajectory Prediction | RMSE ≈ 2.95 km | Evaluate on validation set |
| Maneuver Classification | Acc ≈ 84.5% | Evaluate on validation set |
| Inference Speed | <100ms per object | Benchmark inference.py |
| Tests Pass | 100% pass rate | pytest tests/unit/test_ml_* |
| Memory Safety | <2 GB per chunk | Process 1k objects |
| Documentation | Complete | All 3 markdown files |

---

## 🔧 Technical Notes

### Memory Optimization Strategy
- Process in chunks of 100 objects
- Save intermediate results immediately
- Use `gc.collect()` between chunks
- Monitor RAM usage with `memory_profiler`

### Checkpoint Compatibility
- Trajectory model uses custom state dict keys
- Must implement exact layer names for loading
- Config stored in checkpoint for reproducibility

### Feature Engineering
- Input features: 24D (position, velocity, orbital elements, derived)
- Output features: 6D (position + velocity prediction)
- Sequence length: 20 timesteps (history)
- Prediction horizon: 30 timesteps (future)

---

## 📝 Development Log

**2026-02-07**: Phase 3 restoration initiated after system crash. Created implementation plan and began checkpoint analysis.

---

**Next Steps**: Begin implementation of trajectory_transformer.py with exact architecture matching checkpoint.
