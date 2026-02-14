#!/usr/bin/env python3
"""
Day 4 — Train Anomaly Detection Autoencoder.

Pipeline:
    1. Load ground truth parquet in chunks (100 objects per chunk)
    2. Derive maneuver events from velocity changes per object
    3. Extract 19D behavior profiles via BehaviorFeatureExtractor
    4. Generate synthetic anomalous profiles for validation
    5. Train autoencoder on GPU
    6. Evaluate: normal vs anomalous separation
    7. Save checkpoint

GPU: Used for autoencoder training.
Chunking: Ground truth processed in 10 chunks of 100 objects each.

Usage:
    python scripts/train_anomaly_autoencoder.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.anomaly.behavior_features import (
    BehaviorFeatureExtractor,
    BehaviorProfile,
    ManeuverRecord,
)
from src.ml.anomaly.autoencoder import AutoencoderConfig
from src.ml.anomaly.anomaly_detector import AnomalyDetector

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

DATA_PATH = Path("data/processed/ml_train_1k/ground_truth.parquet")
CHECKPOINT_DIR = Path("checkpoints/phase3_anomaly")
CHUNK_SIZE = 100          # objects per chunk
DV_THRESHOLD = 0.005      # km/s — delta-V above orbital mechanics noise
EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3
THRESHOLD_PERCENTILE = 95.0
SEED = 42

# Maneuver classification thresholds (heuristic from delta-V magnitude)
# These approximate the 6-class scheme used by the CNN-LSTM classifier
DV_BINS = {
    # delta_v range (km/s) → maneuver class
    "normal": (0.0, 0.005),        # class 0
    "drift": (0.005, 0.01),        # class 1
    "station_keeping": (0.01, 0.02),  # class 2
    "minor": (0.02, 0.1),          # class 3
    "major": (0.1, 1.0),           # class 4
    "deorbit": (1.0, float("inf")),   # class 5
}


def classify_delta_v(dv: float) -> int:
    """Classify a delta-V magnitude into maneuver class 0-5."""
    if dv < DV_BINS["normal"][1]:
        return 0
    elif dv < DV_BINS["drift"][1]:
        return 1
    elif dv < DV_BINS["station_keeping"][1]:
        return 2
    elif dv < DV_BINS["minor"][1]:
        return 3
    elif dv < DV_BINS["major"][1]:
        return 4
    else:
        return 5


# -----------------------------------------------------------------------
# Profile extraction
# -----------------------------------------------------------------------

def extract_profiles_from_chunk(
    df_chunk: pd.DataFrame,
    extractor: BehaviorFeatureExtractor,
) -> list[BehaviorProfile]:
    """
    Extract behavior profiles for all objects in a DataFrame chunk.

    For each object:
        1. Compute per-timestep delta-V from velocity differences
        2. Classify each timestep into maneuver class 0-5
        3. Build ManeuverRecord list
        4. Extract 19D behavior profile
    """
    profiles = []
    object_ids = df_chunk["object_id"].unique()

    for obj_id in object_ids:
        obj_df = df_chunk[df_chunk["object_id"] == obj_id].sort_values("time")

        if len(obj_df) < 10:
            continue

        # Velocity columns
        vx = obj_df["velocity_x"].values
        vy = obj_df["velocity_y"].values
        vz = obj_df["velocity_z"].values

        # Delta-V between consecutive timesteps
        dvx = np.diff(vx)
        dvy = np.diff(vy)
        dvz = np.diff(vz)
        dv_mag = np.sqrt(dvx**2 + dvy**2 + dvz**2)

        # Timestamps (seconds since first observation)
        times = (obj_df["time"] - obj_df["time"].iloc[0]).dt.total_seconds().values

        # Build maneuver records (skip first timestep, no delta-V)
        maneuvers = []
        for i, dv in enumerate(dv_mag):
            mc = classify_delta_v(dv)
            maneuvers.append(ManeuverRecord(
                timestamp=times[i + 1],
                maneuver_class=mc,
                delta_v_magnitude=dv,
            ))

        # Current state = last observation
        last = obj_df.iloc[-1]
        position_km = (last["position_x"], last["position_y"], last["position_z"])
        velocity_km_s = (last["velocity_x"], last["velocity_y"], last["velocity_z"])

        profile = extractor.extract(
            object_id=str(obj_id),
            maneuvers=maneuvers,
            position_km=position_km,
            velocity_km_s=velocity_km_s,
        )
        profiles.append(profile)

    return profiles


def generate_anomalous_profiles(
    normal_profiles: list[BehaviorProfile],
    n_anomalies: int = 50,
    seed: int = 42,
) -> list[BehaviorProfile]:
    """
    Generate synthetic anomalous profiles by perturbing normal ones.

    Anomaly types:
        1. Burst maneuvers: extreme maneuver count/rate
        2. High delta-V: abnormally large thrust
        3. Regime mismatch: LEO features with GEO encoding
        4. Erratic timing: low regularity + high entropy
    """
    rng = np.random.RandomState(seed)
    anomalies = []

    for i in range(n_anomalies):
        # Pick a random normal profile as base
        base = normal_profiles[rng.randint(len(normal_profiles))]
        feat = base.features.copy()

        anomaly_type = rng.randint(4)

        if anomaly_type == 0:
            # Burst maneuvers: extreme maneuver count/rate
            # Ensure absolute minimums so zero-activity bases still anomalous
            feat[0] = max(feat[0] * 10.0, 15.0)   # maneuver_count
            feat[1] = max(feat[1] * 10.0, 20.0)   # maneuver_rate (per day)
            feat[2] = max(feat[2], 0.05)           # mean_delta_v
            feat[3] = max(feat[3], 0.15)           # max_delta_v
            feat[5] = max(feat[5], 600.0)          # mean_interval_s (10 min)
            feat[6] = max(feat[6], 0.3)            # some regularity

        elif anomaly_type == 1:
            # High delta-V with coherent maneuver activity
            feat[0] = max(feat[0], 5.0)    # at least 5 maneuvers
            feat[1] = max(feat[1], 5.0)    # at least 5/day rate
            feat[2] = 2.0                  # mean_delta_v (km/s) — very high
            feat[3] = 5.0                  # max_delta_v
            feat[4] = 3.0                  # delta_v_variance
            feat[5] = max(feat[5], 1800.0) # 30 min interval
            feat[6] = 0.5                  # moderate regularity

        elif anomaly_type == 2:
            # Regime mismatch: GEO position with LEO-like maneuver activity
            feat[15:19] = [0, 0, 1, 0]    # GEO regime
            feat[7] = 35786.0             # GEO altitude
            feat[10] = 3.07               # GEO speed
            feat[0] = 20.0                # high maneuver count (unusual for GEO)
            feat[1] = 15.0                # high rate
            feat[2] = 0.1                 # non-trivial delta-v
            feat[3] = 0.5                 # significant max thrust
            feat[5] = 1200.0              # 20 min intervals
            feat[6] = 0.8                 # very regular (phasing-like)

        elif anomaly_type == 3:
            # Erratic timing + diverse behavior with maneuver activity
            feat[0] = max(feat[0], 8.0)   # enough maneuvers for timing stats
            feat[1] = max(feat[1], 10.0)  # ~10/day
            feat[2] = max(feat[2], 0.03)  # some delta-v
            feat[3] = max(feat[3], 0.2)   # moderate max
            feat[5] = 120.0               # mean_interval very short (2 min)
            feat[6] = 0.05                # very irregular
            feat[12] = 0.2                # low dominant fraction (erratic)
            feat[13] = 0.95               # very high entropy
            feat[14] = 1.0                # all unique classes

        anomalies.append(BehaviorProfile(
            object_id=f"ANOMALY-{i:03d}",
            features=feat,
            observation_window_s=base.observation_window_s,
            num_observations=base.num_observations,
        ))

    return anomalies


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\n{'='*60}")
    print("Day 4 — Anomaly Autoencoder Training")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 1: Load ground truth and extract profiles in chunks
    # ------------------------------------------------------------------
    print(f"\n[Step 1] Loading ground truth from {DATA_PATH}")
    t0 = time.time()

    df = pd.read_parquet(DATA_PATH)
    object_ids = sorted(df["object_id"].unique())
    n_objects = len(object_ids)
    n_chunks = (n_objects + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"  Objects: {n_objects}")
    print(f"  Timesteps per object: {len(df) // n_objects}")
    print(f"  Processing in {n_chunks} chunks of {CHUNK_SIZE} objects")

    extractor = BehaviorFeatureExtractor()
    all_profiles: list[BehaviorProfile] = []

    for chunk_idx in range(n_chunks):
        start_id = chunk_idx * CHUNK_SIZE
        end_id = min(start_id + CHUNK_SIZE, n_objects)
        chunk_obj_ids = object_ids[start_id:end_id]

        df_chunk = df[df["object_id"].isin(chunk_obj_ids)]
        profiles = extract_profiles_from_chunk(df_chunk, extractor)
        all_profiles.extend(profiles)

        print(f"  Chunk {chunk_idx+1}/{n_chunks}: {len(profiles)} profiles "
              f"(objects {start_id}-{end_id-1})")

    load_time = time.time() - t0
    print(f"  Total profiles: {len(all_profiles)} in {load_time:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Feature statistics
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Feature statistics")
    features = np.stack([p.features for p in all_profiles])
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Feature means: min={features.mean(0).min():.4f}, max={features.mean(0).max():.4f}")
    print(f"  Feature stds:  min={features.std(0).min():.4f}, max={features.std(0).max():.4f}")

    # Maneuver class distribution across all objects
    class_counts = np.zeros(6)
    for p in all_profiles:
        # Approximate from features: maneuver_count is feat[0]
        class_counts[0] += 1  # all have normal periods
    print(f"  Altitude range: {features[:, 7].min():.0f} - {features[:, 7].max():.0f} km")

    # ------------------------------------------------------------------
    # Step 3: Generate anomalous profiles for validation
    # ------------------------------------------------------------------
    print(f"\n[Step 3] Generating synthetic anomalous profiles")
    anomalous_profiles = generate_anomalous_profiles(all_profiles, n_anomalies=50)
    print(f"  Generated {len(anomalous_profiles)} anomalous profiles (4 types)")

    # ------------------------------------------------------------------
    # Step 4: Train autoencoder on GPU
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Training autoencoder on {device}")
    config = AutoencoderConfig(
        input_dim=19,
        hidden_dims=(32, 16),
        latent_dim=6,
        dropout=0.1,
    )

    detector = AnomalyDetector(
        config=config,
        threshold_percentile=THRESHOLD_PERCENTILE,
        device=device,
    )

    print(f"  Architecture: 19 → 32 → 16 → 6 → 16 → 32 → 19")
    print(f"  Parameters: {detector.model.param_count()}")
    print(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")

    if HAS_MLFLOW:
        mlflow.set_experiment("sda-anomaly-autoencoder")
        mlflow.start_run()
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "n_profiles": len(all_profiles),
            "n_params": detector.model.param_count(),
        })

    t0 = time.time()
    metrics = detector.fit(
        all_profiles,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        verbose=True,
    )
    train_time = time.time() - t0

    print(f"\n  Training complete in {train_time:.1f}s")
    print(f"  Final loss: {metrics['final_loss']:.6f}")
    print(f"  Threshold ({THRESHOLD_PERCENTILE}th %ile): {metrics['threshold']:.6f}")

    # ------------------------------------------------------------------
    # Step 5: Evaluate — normal vs anomalous separation
    # ------------------------------------------------------------------
    print(f"\n[Step 5] Evaluating anomaly detection")

    # Score normal profiles
    normal_scores = detector.score_batch(all_profiles)
    print(f"  Normal scores:  mean={normal_scores.mean():.6f}, "
          f"std={normal_scores.std():.6f}, "
          f"max={normal_scores.max():.6f}")

    # Score anomalous profiles
    anomaly_scores = detector.score_batch(anomalous_profiles)
    print(f"  Anomaly scores: mean={anomaly_scores.mean():.6f}, "
          f"std={anomaly_scores.std():.6f}, "
          f"max={anomaly_scores.max():.6f}")

    # Detection rates
    normal_flagged = (normal_scores > detector.threshold).sum()
    anomaly_flagged = (anomaly_scores > detector.threshold).sum()
    n_normal = len(normal_scores)
    n_anomaly = len(anomaly_scores)

    fpr = normal_flagged / n_normal * 100
    tpr = anomaly_flagged / n_anomaly * 100
    separation = anomaly_scores.mean() / max(normal_scores.mean(), 1e-12)

    print(f"\n  Threshold: {detector.threshold:.6f}")
    print(f"  Normal flagged (FPR):  {normal_flagged}/{n_normal} = {fpr:.1f}%")
    print(f"  Anomaly flagged (TPR): {anomaly_flagged}/{n_anomaly} = {tpr:.1f}%")
    print(f"  Score separation ratio: {separation:.1f}x")

    # Per-anomaly-type detection
    print(f"\n  Per-anomaly-type detection:")
    type_names = ["Burst maneuvers", "High delta-V", "Regime mismatch", "Erratic timing"]
    for t_idx, t_name in enumerate(type_names):
        type_mask = np.arange(n_anomaly) % 4 == t_idx
        type_scores = anomaly_scores[type_mask]
        type_detected = (type_scores > detector.threshold).sum()
        type_total = type_mask.sum()
        print(f"    {t_name}: {type_detected}/{type_total} detected "
              f"(mean score {type_scores.mean():.6f})")

    # Full detection results for a few examples
    print(f"\n  Example detections:")
    for profile in anomalous_profiles[:3]:
        result = detector.detect(profile)
        print(f"    {result.object_id}: score={result.anomaly_score:.4f}, "
              f"anomaly={result.is_anomaly}, "
              f"top_features={result.top_features}")

    # ------------------------------------------------------------------
    # Step 6: Save checkpoint
    # ------------------------------------------------------------------
    print(f"\n[Step 6] Saving checkpoint to {CHECKPOINT_DIR}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    detector.save(CHECKPOINT_DIR)

    # Save training summary
    summary = {
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
        "config": config.to_dict(),
        "training": {
            "n_profiles": len(all_profiles),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "final_loss": metrics["final_loss"],
            "training_time_s": train_time,
        },
        "threshold": {
            "percentile": THRESHOLD_PERCENTILE,
            "value": detector.threshold,
        },
        "evaluation": {
            "n_normal": n_normal,
            "n_anomalous": n_anomaly,
            "false_positive_rate": float(fpr),
            "true_positive_rate": float(tpr),
            "score_separation_ratio": float(separation),
            "normal_score_mean": float(normal_scores.mean()),
            "normal_score_std": float(normal_scores.std()),
            "anomaly_score_mean": float(anomaly_scores.mean()),
            "anomaly_score_std": float(anomaly_scores.std()),
        },
    }

    with open(CHECKPOINT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if HAS_MLFLOW:
        mlflow.log_metrics({
            "final_loss": metrics["final_loss"],
            "fpr": float(fpr),
            "tpr": float(tpr),
            "separation_ratio": float(separation),
        })
        mlflow.log_artifact(str(CHECKPOINT_DIR / "training_summary.json"))
        mlflow.end_run()

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"  Profiles:  {n_normal} normal + {n_anomaly} anomalous")
    print(f"  Params:    {detector.model.param_count()}")
    print(f"  FPR:       {fpr:.1f}%")
    print(f"  TPR:       {tpr:.1f}%")
    print(f"  Separation: {separation:.1f}x")
    print(f"  Checkpoint: {CHECKPOINT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
