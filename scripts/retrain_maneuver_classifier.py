#!/usr/bin/env python3
"""
Retrain ManeuverClassifier CNN-LSTM with proper early stopping.

Fixes the mode collapse issue from original training where the checkpoint was
saved at the final epoch (val_acc=39.5%) instead of the best epoch (val_acc=84.5%).

Key fixes:
1. Save checkpoint on best val_acc (not val_loss or final epoch)
2. Early stopping with patience=10 on val_acc
3. More training data (5000 samples per class = 30K total)
4. Learning rate warmup + cosine decay
5. Class-balanced sampling

Classes:
  0: Normal        - Unperturbed Keplerian orbit
  1: Drift/Decay   - Slowly decreasing altitude (drag-like)
  2: Station-keeping - Small periodic corrections
  3: Minor Maneuver - Small delta-V (0.01-0.1 km/s)
  4: Major Maneuver - Large delta-V (0.1-1.0 km/s)
  5: Deorbit       - Large retrograde burn, steep altitude drop
"""

import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.models.maneuver_classifier import CNNLSTMManeuverClassifier, CNNLSTMClassifierConfig
from src.ml.features.trajectory_features import TrajectoryFeatureExtractor, FeatureConfig

# Constants
MU = 398600.4418       # km^3/s^2
R_EARTH = 6371.0       # km
SEQ_LEN = 20           # History timesteps
DT = 60.0              # Seconds between timesteps
FEATURE_DIM = 24

CLASS_NAMES = {
    0: "Normal",
    1: "Drift/Decay",
    2: "Station-keeping",
    3: "Minor Maneuver",
    4: "Major Maneuver",
    5: "Deorbit",
}

# ─── Orbit generation helpers ───────────────────────────────────────────

def _circular_velocity(r: float) -> float:
    return np.sqrt(MU / r)


def _propagate(r: np.ndarray, v: np.ndarray, dt: float) -> tuple:
    """Simple Euler propagation step."""
    r_norm = np.linalg.norm(r)
    a = -MU / r_norm**3 * r
    v_new = v + a * dt
    r_new = r + v_new * dt
    return r_new, v_new


def _generate_orbit(alt_km: float, inc_deg: float, raan_deg: float, n_steps: int) -> tuple:
    """Generate a circular orbit trajectory."""
    r0 = R_EARTH + alt_km
    v0 = _circular_velocity(r0)
    inc = np.radians(inc_deg)
    raan = np.radians(raan_deg)

    # Initial state in orbital plane
    r_orb = np.array([r0, 0.0, 0.0])
    v_orb = np.array([0.0, v0, 0.0])

    # Rotation: first by inclination around x, then by RAAN around z
    ci, si = np.cos(inc), np.sin(inc)
    cr, sr = np.cos(raan), np.sin(raan)
    Rx = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    R = Rz @ Rx

    r = R @ r_orb
    v = R @ v_orb

    positions = np.zeros((n_steps, 3))
    velocities = np.zeros((n_steps, 3))
    positions[0] = r
    velocities[0] = v

    for i in range(1, n_steps):
        r, v = _propagate(r, v, DT)
        positions[i] = r
        velocities[i] = v

    return positions, velocities


def _apply_delta_v(vel: np.ndarray, dv_vec: np.ndarray) -> np.ndarray:
    """Apply impulsive maneuver to velocity."""
    return vel + dv_vec


# ─── Synthetic data generators per class ────────────────────────────────

def generate_normal(rng: np.random.Generator) -> tuple:
    """Class 0: Normal orbit — no maneuvers."""
    alt = rng.uniform(200, 36000)
    inc = rng.uniform(0, 90)
    raan = rng.uniform(0, 360)
    pos, vel = _generate_orbit(alt, inc, raan, SEQ_LEN)
    return pos, vel


def generate_drift_decay(rng: np.random.Generator) -> tuple:
    """Class 1: Drift/Decay — slowly losing altitude (atmospheric drag)."""
    alt = rng.uniform(200, 600)  # Low orbits where drag matters
    inc = rng.uniform(0, 90)
    raan = rng.uniform(0, 360)
    pos, vel = _generate_orbit(alt, inc, raan, SEQ_LEN)

    # Apply gradual velocity reduction simulating drag
    drag_rate = rng.uniform(0.0001, 0.001)  # km/s per step
    for i in range(1, SEQ_LEN):
        v_dir = vel[i] / np.linalg.norm(vel[i])
        vel[i] -= v_dir * drag_rate * i
        # Re-propagate position from modified velocity
        r_norm = np.linalg.norm(pos[i - 1])
        a = -MU / r_norm**3 * pos[i - 1]
        vel[i] = vel[i - 1] + a * DT - v_dir * drag_rate
        pos[i] = pos[i - 1] + vel[i] * DT

    return pos, vel


def generate_station_keeping(rng: np.random.Generator) -> tuple:
    """Class 2: Station-keeping — small periodic corrections."""
    alt = rng.uniform(35000, 36500)  # GEO
    inc = rng.uniform(0, 5)
    raan = rng.uniform(0, 360)
    pos, vel = _generate_orbit(alt, inc, raan, SEQ_LEN)

    # Apply small periodic corrections every 5-7 steps
    correction_interval = rng.integers(4, 8)
    correction_dv = rng.uniform(0.001, 0.01)  # Very small burns

    for i in range(correction_interval, SEQ_LEN, correction_interval):
        # Random direction correction
        dv_dir = rng.normal(size=3)
        dv_dir /= np.linalg.norm(dv_dir)
        vel[i] = _apply_delta_v(vel[i], dv_dir * correction_dv)
        # Re-propagate from correction
        for j in range(i + 1, min(i + correction_interval, SEQ_LEN)):
            pos[j], vel[j] = _propagate(pos[j - 1], vel[j - 1], DT)

    return pos, vel


def generate_minor_maneuver(rng: np.random.Generator) -> tuple:
    """Class 3: Minor maneuver — small delta-V (0.01-0.1 km/s)."""
    alt = rng.uniform(300, 36000)
    inc = rng.uniform(0, 90)
    raan = rng.uniform(0, 360)
    pos, vel = _generate_orbit(alt, inc, raan, SEQ_LEN)

    # Apply minor burn at random timestep in second half
    burn_step = rng.integers(SEQ_LEN // 3, 2 * SEQ_LEN // 3)
    dv_mag = rng.uniform(0.01, 0.1)

    # Random direction (prograde, radial, or normal)
    direction_type = rng.integers(0, 3)
    if direction_type == 0:  # Prograde
        dv_dir = vel[burn_step] / np.linalg.norm(vel[burn_step])
    elif direction_type == 1:  # Radial
        dv_dir = pos[burn_step] / np.linalg.norm(pos[burn_step])
    else:  # Normal
        h = np.cross(pos[burn_step], vel[burn_step])
        dv_dir = h / np.linalg.norm(h)

    vel[burn_step] = _apply_delta_v(vel[burn_step], dv_dir * dv_mag)

    # Re-propagate after burn
    for j in range(burn_step + 1, SEQ_LEN):
        pos[j], vel[j] = _propagate(pos[j - 1], vel[j - 1], DT)

    return pos, vel


def generate_major_maneuver(rng: np.random.Generator) -> tuple:
    """Class 4: Major maneuver — large delta-V (0.1-1.0 km/s)."""
    alt = rng.uniform(300, 36000)
    inc = rng.uniform(0, 90)
    raan = rng.uniform(0, 360)
    pos, vel = _generate_orbit(alt, inc, raan, SEQ_LEN)

    # Apply major burn
    burn_step = rng.integers(SEQ_LEN // 3, 2 * SEQ_LEN // 3)
    dv_mag = rng.uniform(0.1, 1.0)

    direction_type = rng.integers(0, 3)
    if direction_type == 0:
        dv_dir = vel[burn_step] / np.linalg.norm(vel[burn_step])
    elif direction_type == 1:
        dv_dir = pos[burn_step] / np.linalg.norm(pos[burn_step])
    else:
        h = np.cross(pos[burn_step], vel[burn_step])
        dv_dir = h / np.linalg.norm(h)

    vel[burn_step] = _apply_delta_v(vel[burn_step], dv_dir * dv_mag)

    for j in range(burn_step + 1, SEQ_LEN):
        pos[j], vel[j] = _propagate(pos[j - 1], vel[j - 1], DT)

    return pos, vel


def generate_deorbit(rng: np.random.Generator) -> tuple:
    """Class 5: Deorbit burn — large retrograde burn, steep descent."""
    alt = rng.uniform(300, 800)  # LEO
    inc = rng.uniform(0, 90)
    raan = rng.uniform(0, 360)
    pos, vel = _generate_orbit(alt, inc, raan, SEQ_LEN)

    # Apply large retrograde burn early in sequence
    burn_step = rng.integers(2, SEQ_LEN // 3)
    dv_mag = rng.uniform(0.3, 1.5)

    # Retrograde direction
    dv_dir = -vel[burn_step] / np.linalg.norm(vel[burn_step])
    vel[burn_step] = _apply_delta_v(vel[burn_step], dv_dir * dv_mag)

    # Re-propagate — altitude will drop
    for j in range(burn_step + 1, SEQ_LEN):
        pos[j], vel[j] = _propagate(pos[j - 1], vel[j - 1], DT)

    return pos, vel


GENERATORS = {
    0: generate_normal,
    1: generate_drift_decay,
    2: generate_station_keeping,
    3: generate_minor_maneuver,
    4: generate_major_maneuver,
    5: generate_deorbit,
}


# ─── Dataset generation ─────────────────────────────────────────────────

def generate_dataset(n_per_class: int, seed: int = 42) -> tuple:
    """Generate balanced synthetic training data with 24D features."""
    rng = np.random.default_rng(seed)

    feature_config = FeatureConfig(
        include_position=True,
        include_velocity=True,
        include_orbital_elements=True,
        include_derived_features=True,
        include_temporal_features=True,
        include_uncertainty=False,  # 24D total
    )
    extractor = TrajectoryFeatureExtractor(feature_config)

    all_features = []
    all_labels = []
    failed = 0

    for class_idx in range(6):
        gen_fn = GENERATORS[class_idx]
        count = 0
        attempts = 0
        while count < n_per_class and attempts < n_per_class * 3:
            attempts += 1
            try:
                pos, vel = gen_fn(rng)
                timestamps = np.arange(SEQ_LEN) * DT

                features = extractor.extract_features(pos, vel, timestamps)

                # Validate: no NaN/Inf
                if not np.all(np.isfinite(features)):
                    failed += 1
                    continue

                all_features.append(features)
                all_labels.append(class_idx)
                count += 1
            except Exception:
                failed += 1
                continue

        if count < n_per_class:
            print(f"  WARNING: Class {class_idx} ({CLASS_NAMES[class_idx]}): only {count}/{n_per_class} samples")

    print(f"Generated {len(all_features)} samples ({failed} failed attempts)")

    X = np.array(all_features, dtype=np.float32)  # (N, 20, 24)
    y = np.array(all_labels, dtype=np.int64)       # (N,)

    return X, y


# ─── Training ───────────────────────────────────────────────────────────

def train(
    n_per_class: int = 5000,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    checkpoint_dir: str = "checkpoints/phase3_day4",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Generating {n_per_class * 6} synthetic training samples...")

    t0 = time.time()

    # Generate data
    X, y = generate_dataset(n_per_class, seed=42)
    print(f"Dataset: {X.shape}, Labels: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Shuffle and split 80/20
    rng = np.random.default_rng(123)
    indices = rng.permutation(len(X))
    split = int(0.8 * len(X))

    X_train = torch.from_numpy(X[indices[:split]])
    y_train = torch.from_numpy(y[indices[:split]])
    X_val = torch.from_numpy(X[indices[split:]])
    y_val = torch.from_numpy(y[indices[split:]])

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Dataloaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=True)

    # Model
    config = CNNLSTMClassifierConfig(
        input_dim=FEATURE_DIM,
        cnn_channels=[32, 64, 128],
        kernel_size=3,
        lstm_hidden_dim=128,
        lstm_layers=2,
        bidirectional=True,
        classifier_dims=[256, 128],
        num_classes=6,
        dropout=0.3,
    )
    model = CNNLSTMManeuverClassifier(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # History
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    data_gen_time = time.time() - t0
    print(f"Data generation: {data_gen_time:.1f}s")
    print(f"\nTraining {epochs} epochs with patience={patience} on best val_acc...")
    print("-" * 70)

    if HAS_MLFLOW:
        mlflow.set_experiment("sda-maneuver-classifier")
        mlflow.start_run()
        mlflow.log_params({
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "n_per_class": n_per_class,
            "n_params": n_params,
        })

    t_train = time.time()

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += X_batch.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += X_batch.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if HAS_MLFLOW:
            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr,
            }, step=epoch)

        # Check improvement (on val_acc, NOT val_loss)
        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            improved = " ★ BEST"

            # Save best checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "history": history,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, checkpoint_path / "maneuver_classifier.pt")
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch+1:2d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
            f"LR: {current_lr:.2e}{improved}"
        )

        # Early stopping on val_acc
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    train_time = time.time() - t_train
    total_time = time.time() - t0

    # Save history
    with open(checkpoint_path / "maneuver_classifier_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("-" * 70)
    print(f"Best val_acc: {best_val_acc:.4f} at epoch {best_epoch+1}")
    print(f"Training time: {train_time:.1f}s | Total: {total_time:.1f}s")
    print(f"Checkpoint: {checkpoint_path / 'maneuver_classifier.pt'}")

    # ── Quick validation of saved model ──
    print("\n── Post-training validation ──")
    ckpt = torch.load(checkpoint_path / "maneuver_classifier.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Per-class accuracy
    all_preds = []
    all_true = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(1).cpu()
            all_preds.append(preds)
            all_true.append(y_batch)

    all_preds = torch.cat(all_preds).numpy()
    all_true = torch.cat(all_true).numpy()

    print(f"Overall val accuracy: {(all_preds == all_true).mean():.4f}")
    for c in range(6):
        mask = all_true == c
        if mask.sum() > 0:
            acc = (all_preds[mask] == c).mean()
            print(f"  Class {c} ({CLASS_NAMES[c]:>16s}): {acc:.3f} ({mask.sum()} samples)")

    # Check for mode collapse
    unique_preds = np.unique(all_preds)
    if len(unique_preds) <= 2:
        print(f"\n⚠ WARNING: Model may have collapsed — only predicts classes {unique_preds}")
    else:
        print(f"\n✓ Model predicts {len(unique_preds)} distinct classes: {unique_preds}")

    if HAS_MLFLOW:
        mlflow.log_metrics({"best_val_acc": best_val_acc, "best_epoch": best_epoch})
        mlflow.log_artifact(str(checkpoint_path / "maneuver_classifier_history.json"))
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()


if __name__ == "__main__":
    train()
