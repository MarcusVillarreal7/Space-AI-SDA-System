#!/usr/bin/env python3
"""
Train the CollisionRiskPredictor on synthetic conjunction data.

Generates positive pairs (close approaches at various distances/closing rates)
and negative pairs (random far-apart objects), extracts 24D features via
TrajectoryFeatureExtractor, and trains the 58D→256→128→64→3 MLP.

Uses GPU with chunked data generation for efficiency.

Usage:
    python scripts/train_collision_predictor.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ml.models.collision_predictor import CollisionPredictor, CollisionPredictorConfig
from src.ml.features.trajectory_features import TrajectoryFeatureExtractor, FeatureConfig

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
N_POSITIVE = 10_000
N_NEGATIVE = 10_000
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
CHECKPOINT_DIR = Path("checkpoints/collision_predictor")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MU = 398600.4418  # km^3/s^2
EARTH_RADIUS = 6371.0
FEATURE_DIM = 24

# Feature extraction config (must match training config used elsewhere)
FEATURE_CONFIG = FeatureConfig(
    include_position=True,
    include_velocity=True,
    include_orbital_elements=True,
    include_derived_features=True,
    include_temporal_features=True,
    include_uncertainty=False,
)


def circular_velocity(r_km: float) -> float:
    return np.sqrt(MU / r_km)


def generate_random_orbit_state(alt_range=(200, 40000)) -> tuple[np.ndarray, np.ndarray]:
    """Generate random orbital state (position, velocity) in ECI."""
    alt = np.random.uniform(*alt_range)
    r = EARTH_RADIUS + alt

    # Random orientation
    inc = np.random.uniform(0, np.pi)
    raan = np.random.uniform(0, 2 * np.pi)
    ta = np.random.uniform(0, 2 * np.pi)

    v_circ = circular_velocity(r)

    # In orbital plane
    r_orb = np.array([r * np.cos(ta), r * np.sin(ta), 0.0])
    v_orb = np.array([-v_circ * np.sin(ta), v_circ * np.cos(ta), 0.0])

    # Rotation matrices
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    Rz = np.array([[cO, -sO, 0], [sO, cO, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    R = Rz @ Rx

    return R @ r_orb, R @ v_orb


def propagate_simple(r0, v0, n_steps=20, dt=60.0):
    """Simple Kepler propagation for feature extraction."""
    positions = np.zeros((n_steps, 3))
    velocities = np.zeros((n_steps, 3))
    positions[0] = r0
    velocities[0] = v0
    for i in range(1, n_steps):
        r_mag = np.linalg.norm(positions[i - 1])
        if r_mag < 100:
            positions[i] = positions[i - 1]
            velocities[i] = velocities[i - 1]
            continue
        a_grav = -MU * positions[i - 1] / (r_mag ** 3)
        velocities[i] = velocities[i - 1] + a_grav * dt
        positions[i] = positions[i - 1] + velocities[i] * dt
    return positions, velocities


def extract_24d_features(positions, velocities):
    """Extract 24D feature vector from trajectory using TrajectoryFeatureExtractor."""
    extractor = TrajectoryFeatureExtractor(FEATURE_CONFIG)
    timestamps = np.arange(len(positions)) * 60.0
    features = extractor.extract_features(positions, velocities, timestamps)
    # Average over timesteps to get single 24D vector
    return features.mean(axis=0)


def generate_positive_pairs_chunked(n_pairs: int, chunk_size: int = 500):
    """
    Generate close-approach pairs in chunks for memory efficiency.
    Returns (features1, features2, labels) arrays.
    """
    all_f1, all_f2, all_labels = [], [], []

    for chunk_start in range(0, n_pairs, chunk_size):
        chunk_n = min(chunk_size, n_pairs - chunk_start)
        f1_chunk = np.zeros((chunk_n, FEATURE_DIM))
        f2_chunk = np.zeros((chunk_n, FEATURE_DIM))
        labels_chunk = np.zeros((chunk_n, 3))  # risk, log_ttca, log_miss_dist

        for i in range(chunk_n):
            # Object 1: random orbit
            r1, v1 = generate_random_orbit_state()
            pos1, vel1 = propagate_simple(r1, v1)

            # Object 2: close to object 1 (close approach scenario)
            miss_distance = np.random.uniform(5, 500)  # 5-500 km
            closing_rate = np.random.uniform(0.01, 5.0)  # km/s

            # Place obj2 offset from obj1
            r_hat = r1 / (np.linalg.norm(r1) + 1e-10)
            offset_dir = np.cross(r_hat, np.array([0, 0, 1]))
            offset_dir = offset_dir / (np.linalg.norm(offset_dir) + 1e-10)

            r2 = r1 + offset_dir * miss_distance
            v2 = v1 - offset_dir * closing_rate
            pos2, vel2 = propagate_simple(r2, v2)

            f1_chunk[i] = extract_24d_features(pos1, vel1)
            f2_chunk[i] = extract_24d_features(pos2, vel2)

            # Ground truth
            rel_pos = r2 - r1
            rel_vel = v2 - v1
            dist = np.linalg.norm(rel_pos)
            rel_speed = np.linalg.norm(rel_vel)
            dot_rv = np.dot(rel_pos, rel_vel)
            ttca = max(0, -dot_rv / (rel_speed ** 2 + 1e-10))
            # Miss distance at closest approach
            miss_at_tca = np.linalg.norm(np.cross(rel_pos, rel_vel)) / (rel_speed + 1e-10)

            # Risk: sigmoid-like based on distance and closing rate
            risk = 1.0 / (1.0 + (dist / 50.0) ** 2) * min(1.0, closing_rate / 1.0)
            risk = np.clip(risk, 0.01, 0.99)

            labels_chunk[i] = [risk, np.log1p(ttca), np.log1p(miss_at_tca)]

        all_f1.append(f1_chunk)
        all_f2.append(f2_chunk)
        all_labels.append(labels_chunk)

        progress = min(chunk_start + chunk_n, n_pairs)
        print(f"  Positive pairs: {progress}/{n_pairs}", end="\r")

    print()
    return np.vstack(all_f1), np.vstack(all_f2), np.vstack(all_labels)


def generate_negative_pairs_chunked(n_pairs: int, chunk_size: int = 500):
    """
    Generate far-apart pairs (no collision risk) in chunks.
    """
    all_f1, all_f2, all_labels = [], [], []

    for chunk_start in range(0, n_pairs, chunk_size):
        chunk_n = min(chunk_size, n_pairs - chunk_start)
        f1_chunk = np.zeros((chunk_n, FEATURE_DIM))
        f2_chunk = np.zeros((chunk_n, FEATURE_DIM))
        labels_chunk = np.zeros((chunk_n, 3))

        for i in range(chunk_n):
            # Two random, unrelated orbits
            r1, v1 = generate_random_orbit_state()
            r2, v2 = generate_random_orbit_state()
            pos1, vel1 = propagate_simple(r1, v1)
            pos2, vel2 = propagate_simple(r2, v2)

            f1_chunk[i] = extract_24d_features(pos1, vel1)
            f2_chunk[i] = extract_24d_features(pos2, vel2)

            # Ground truth: low risk, high TTCA, high miss distance
            rel_pos = r2 - r1
            rel_vel = v2 - v1
            dist = np.linalg.norm(rel_pos)
            rel_speed = np.linalg.norm(rel_vel)
            miss = np.linalg.norm(np.cross(rel_pos, rel_vel)) / (rel_speed + 1e-10)

            labels_chunk[i] = [0.01, np.log1p(86400.0), np.log1p(miss)]

        all_f1.append(f1_chunk)
        all_f2.append(f2_chunk)
        all_labels.append(labels_chunk)

        progress = min(chunk_start + chunk_n, n_pairs)
        print(f"  Negative pairs: {progress}/{n_pairs}", end="\r")

    print()
    return np.vstack(all_f1), np.vstack(all_f2), np.vstack(all_labels)


def train():
    print(f"Device: {DEVICE}")
    print(f"Generating {N_POSITIVE} positive + {N_NEGATIVE} negative pairs...\n")

    t0 = time.time()

    # Generate data in chunks
    print("Generating positive (close approach) pairs...")
    pos_f1, pos_f2, pos_labels = generate_positive_pairs_chunked(N_POSITIVE)

    print("Generating negative (far apart) pairs...")
    neg_f1, neg_f2, neg_labels = generate_negative_pairs_chunked(N_NEGATIVE)

    # Combine
    f1 = np.vstack([pos_f1, neg_f1]).astype(np.float32)
    f2 = np.vstack([pos_f2, neg_f2]).astype(np.float32)
    labels = np.vstack([pos_labels, neg_labels]).astype(np.float32)

    # Handle NaN/Inf from feature extraction
    f1 = np.nan_to_num(f1, nan=0.0, posinf=1e6, neginf=-1e6)
    f2 = np.nan_to_num(f2, nan=0.0, posinf=1e6, neginf=-1e6)
    labels = np.nan_to_num(labels, nan=0.0, posinf=10.0, neginf=0.0)

    gen_time = time.time() - t0
    print(f"\nData generation: {gen_time:.1f}s")
    print(f"Features shape: {f1.shape}, Labels shape: {labels.shape}")

    # Shuffle
    perm = np.random.permutation(len(f1))
    f1, f2, labels = f1[perm], f2[perm], labels[perm]

    # Train/val split (80/20)
    split = int(0.8 * len(f1))
    train_f1, val_f1 = f1[:split], f1[split:]
    train_f2, val_f2 = f2[:split], f2[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    # Create DataLoaders — move data to GPU via pin_memory for efficiency
    train_ds = TensorDataset(
        torch.from_numpy(train_f1), torch.from_numpy(train_f2), torch.from_numpy(train_labels)
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_f1), torch.from_numpy(val_f2), torch.from_numpy(val_labels)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(DEVICE == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=(DEVICE == "cuda"))

    # Create model
    config = CollisionPredictorConfig(
        input_dim=48,
        hidden_dims=[256, 128, 64],
        output_dim=3,
        dropout=0.2,
    )
    model = CollisionPredictor(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: CollisionPredictor ({n_params:,} params) on {DEVICE}")

    # Loss: BCE(risk) + MSE(log_TTCA) + MSE(log_miss_distance)
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\nTraining for {EPOCHS} epochs (batch_size={BATCH_SIZE}, lr={LR})...\n")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_f1, batch_f2, batch_labels in train_loader:
            batch_f1 = batch_f1.to(DEVICE, non_blocking=True)
            batch_f2 = batch_f2.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            output = model(batch_f1, batch_f2)  # (B, 3)

            # Split outputs and labels
            pred_risk = output[:, 0]        # sigmoid already applied in model
            pred_ttca = output[:, 1]        # softplus
            pred_miss = output[:, 2]        # softplus

            true_risk = batch_labels[:, 0]
            true_ttca = batch_labels[:, 1]  # log1p(ttca)
            true_miss = batch_labels[:, 2]  # log1p(miss_dist)

            loss_risk = bce(pred_risk, true_risk)
            loss_ttca = mse(torch.log1p(pred_ttca), true_ttca)
            loss_miss = mse(torch.log1p(pred_miss), true_miss)

            loss = loss_risk + 0.5 * loss_ttca + 0.5 * loss_miss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_f1, batch_f2, batch_labels in val_loader:
                batch_f1 = batch_f1.to(DEVICE, non_blocking=True)
                batch_f2 = batch_f2.to(DEVICE, non_blocking=True)
                batch_labels = batch_labels.to(DEVICE, non_blocking=True)

                output = model(batch_f1, batch_f2)
                pred_risk = output[:, 0]
                pred_ttca = output[:, 1]
                pred_miss = output[:, 2]

                true_risk = batch_labels[:, 0]
                true_ttca = batch_labels[:, 1]
                true_miss = batch_labels[:, 2]

                loss_risk = bce(pred_risk, true_risk)
                loss_ttca = mse(torch.log1p(pred_ttca), true_ttca)
                loss_miss = mse(torch.log1p(pred_miss), true_miss)

                loss = loss_risk + 0.5 * loss_ttca + 0.5 * loss_miss
                val_loss += loss.item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)
        scheduler.step(avg_val)

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": model.get_config(),
                "epoch": epoch,
                "val_loss": avg_val,
            }, CHECKPOINT_DIR / "best_model.pt")
            marker = " *"
        else:
            marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0 or marker:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | train={avg_train:.4f} val={avg_val:.4f}{marker}")

    # Final save
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "epoch": EPOCHS,
        "val_loss": avg_val,
    }, CHECKPOINT_DIR / "final_model.pt")

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    train()
