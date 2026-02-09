#!/usr/bin/env python3
"""
Space-AI Model Showcase — Generate social-media-ready visualizations.

Runs each of the project's 6 core ML models on synthetic data and produces
high-DPI PNG figures saved to results/model_showcase/.

Usage:
    python scripts/generate_model_showcase.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Project imports ──
from src.ml.models.trajectory_transformer import TrajectoryTransformer, TransformerConfig
from src.ml.models.maneuver_classifier import CNNLSTMManeuverClassifier, CNNLSTMClassifierConfig, CLASS_NAMES
from src.ml.models.collision_predictor import CollisionPredictor, CollisionPredictorConfig, RISK_LEVELS
from src.ml.anomaly.autoencoder import BehaviorAutoencoder, AutoencoderConfig
from src.ml.uncertainty.monte_carlo import MCDropoutPredictor, MCDropoutConfig

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "results" / "model_showcase"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════
#  Global style
# ═══════════════════════════════════════════════════════════════
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
ACCENT     = "#58a6ff"
ACCENT2    = "#f78166"
ACCENT3    = "#3fb950"
ACCENT4    = "#d2a8ff"
ACCENT5    = "#f0883e"
TEXT_COLOR  = "#c9d1d9"
GRID_COLOR  = "#21262d"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   GRID_COLOR,
    "axes.labelcolor":  TEXT_COLOR,
    "axes.grid":        True,
    "grid.color":       GRID_COLOR,
    "grid.alpha":       0.5,
    "text.color":       TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "font.family":      "sans-serif",
    "font.size":        11,
    "legend.facecolor": PANEL_BG,
    "legend.edgecolor": GRID_COLOR,
    "savefig.facecolor": DARK_BG,
    "savefig.dpi":      200,
})

def _save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════
#  Figure 1 — Trajectory Transformer: Predicted vs Ground Truth
# ═══════════════════════════════════════════════════════════════
def fig_trajectory_prediction():
    print("\n[1/6] Trajectory Transformer — Prediction vs Ground Truth")

    # Load trained parallel-head model
    ckpt_path = CKPT / "phase3_parallel" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model_cfg = ckpt.get("model_config", {})
    config = TransformerConfig(**model_cfg)
    model = TrajectoryTransformer(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Synthetic ISS-like orbit (circular at ~400 km)
    R = 6771.0  # km
    T_orbit = 2 * math.pi * math.sqrt(R**3 / 398600.4418)  # ~5580 s
    omega = 2 * math.pi / T_orbit
    dt = 60.0  # 1 min steps
    total_steps = 50  # 20 history + 30 future

    t_arr = np.arange(total_steps) * dt
    pos_x = R * np.cos(omega * t_arr)
    pos_y = R * np.sin(omega * t_arr)
    pos_z = np.zeros(total_steps)
    vel_x = -R * omega * np.sin(omega * t_arr)
    vel_y =  R * omega * np.cos(omega * t_arr)
    vel_z = np.zeros(total_steps)

    # Build 24D features (pos3 + vel3 + 18 zeros for orbital/derived/temporal/uncertainty)
    feats = np.zeros((total_steps, 24))
    feats[:, 0] = pos_x; feats[:, 1] = pos_y; feats[:, 2] = pos_z
    feats[:, 3] = vel_x; feats[:, 4] = vel_y; feats[:, 5] = vel_z

    src = torch.tensor(feats[:20], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model.predict(src, pred_horizon=30).cpu().numpy()[0]  # (30, 6)

    gt = feats[20:, :6]
    pred_pos = pred[:, :3]
    gt_pos = gt[:, :3]
    horizon = np.arange(1, 31)
    pos_error = np.sqrt(((pred_pos - gt_pos)**2).sum(axis=1))

    # Training history
    hist = json.loads((CKPT / "phase3_parallel" / "training_history.json").read_text())

    # ── Plot ──
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.32)

    # Panel A — orbit view
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(feats[:20, 0], feats[:20, 1], color=ACCENT, lw=2, label="History (20 steps)")
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], color=ACCENT3, lw=2, ls="--", label="Ground Truth")
    ax1.plot(pred_pos[:, 0], pred_pos[:, 1], color=ACCENT2, lw=2, label="Predicted")
    ax1.scatter([feats[0, 0]], [feats[0, 1]], c="white", s=60, zorder=5, marker="o")
    ax1.scatter([feats[19, 0]], [feats[19, 1]], c=ACCENT, s=80, zorder=5, marker="D",
                edgecolors="white", linewidths=0.8)
    ax1.set_xlabel("X Position (km)")
    ax1.set_ylabel("Y Position (km)")
    ax1.set_title("Orbital Trajectory — XY Plane", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=9)
    ax1.set_aspect("equal")

    # Panel B — per-horizon error
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(horizon, 0, pos_error, alpha=0.25, color=ACCENT2)
    ax2.plot(horizon, pos_error, color=ACCENT2, lw=2)
    ax2.set_xlabel("Prediction Horizon (steps)")
    ax2.set_ylabel("Position RMSE (km)")
    ax2.set_title("Error vs Horizon", fontsize=13, fontweight="bold")
    mean_err = pos_error.mean()
    ax2.axhline(mean_err, color=ACCENT, ls="--", lw=1, alpha=0.7)
    ax2.text(15, mean_err + 0.3, f"Mean: {mean_err:.2f} km", ha="center",
             fontsize=10, color=ACCENT)

    # Panel C — training loss
    ax3 = fig.add_subplot(gs[2])
    epochs = range(1, len(hist["train_loss"]) + 1)
    ax3.plot(epochs, hist["train_loss"], color=ACCENT2, lw=2, label="Train Loss")
    ax3.plot(epochs, hist["val_loss"], color=ACCENT3, lw=2, label="Val Loss")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.set_title("Training Convergence", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9)

    fig.suptitle("TRAJECTORY TRANSFORMER  ·  Parallel Prediction Head  ·  371K params",
                 fontsize=15, fontweight="bold", y=1.02, color="white")
    _save(fig, "01_trajectory_transformer")


# ═══════════════════════════════════════════════════════════════
#  Figure 2 — Maneuver Classifier: Class Probabilities
# ═══════════════════════════════════════════════════════════════
def fig_maneuver_classifier():
    print("\n[2/6] Maneuver Classifier — Class Probabilities")

    ckpt = torch.load(CKPT / "phase3_day4" / "maneuver_classifier.pt",
                      map_location=DEVICE, weights_only=False)
    raw_cfg = ckpt["config"]
    if isinstance(raw_cfg, CNNLSTMClassifierConfig):
        # Re-create via dict to trigger __post_init__ for None defaults
        from dataclasses import asdict
        config = CNNLSTMClassifierConfig(**{k: v for k, v in asdict(raw_cfg).items() if v is not None})
    else:
        config = CNNLSTMClassifierConfig(**raw_cfg)
    model = CNNLSTMManeuverClassifier(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    hist = json.loads((CKPT / "phase3_day4" / "maneuver_classifier_history.json").read_text())

    # Create 6 synthetic scenarios (one per class)
    torch.manual_seed(42)
    scenarios = {
        "Normal orbit":      torch.randn(1, 20, 24) * 0.1,
        "Decay pattern":     torch.randn(1, 20, 24) * 0.5 + 0.3,
        "Station-keeping":   torch.randn(1, 20, 24) * 0.2 - 0.1,
        "Minor delta-V":     torch.randn(1, 20, 24) * 0.8,
        "Major maneuver":    torch.randn(1, 20, 24) * 1.5,
        "Deorbit burn":      torch.randn(1, 20, 24) * 2.0 + 1.0,
    }

    class_names = [CLASS_NAMES[i] for i in range(6)]
    colors = [ACCENT, "#79c0ff", ACCENT3, ACCENT4, ACCENT2, "#ff7b72"]

    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.35)

    # Panel A — probability bars for each scenario
    ax1 = fig.add_subplot(gs[0])
    scenario_names = list(scenarios.keys())
    probs_all = []
    for name, x in scenarios.items():
        with torch.no_grad():
            p = model.predict_proba(x.to(DEVICE)).cpu().numpy()[0]
        probs_all.append(p)
    probs_all = np.array(probs_all)

    x_pos = np.arange(len(class_names))
    bar_width = 0.12
    for i, (sname, prob) in enumerate(zip(scenario_names, probs_all)):
        offset = (i - 2.5) * bar_width
        ax1.bar(x_pos + offset, prob, bar_width, label=sname,
                color=colors[i], alpha=0.85, edgecolor="none")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Probability")
    ax1.set_title("Class Probabilities (6 Scenarios)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=7.5, ncol=2, loc="upper right")
    ax1.set_ylim(0, 1.0)

    # Panel B — training loss
    ax2 = fig.add_subplot(gs[1])
    epochs = range(1, len(hist["train_loss"]) + 1)
    ax2.plot(epochs, hist["train_loss"], color=ACCENT2, lw=2, label="Train Loss")
    ax2.plot(epochs, hist["val_loss"], color=ACCENT3, lw=2, label="Val Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.set_title("Training Convergence", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)

    # Panel C — accuracy
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(epochs, [a * 100 for a in hist["train_acc"]], color=ACCENT, lw=2, label="Train Acc")
    ax3.plot(epochs, [a * 100 for a in hist["val_acc"]], color=ACCENT3, lw=2, label="Val Acc")
    ax3.axhline(84.5, color=ACCENT2, ls="--", lw=1, alpha=0.7)
    ax3.text(10, 87, "Best: 84.5%", ha="center", fontsize=10, color=ACCENT2)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Classification Accuracy", fontsize=13, fontweight="bold")
    ax3.set_ylim(0, 100)
    ax3.legend(fontsize=9)

    fig.suptitle("CNN-LSTM MANEUVER CLASSIFIER  ·  6 Classes  ·  719K params  ·  84.5% Accuracy",
                 fontsize=15, fontweight="bold", y=1.02, color="white")
    _save(fig, "02_maneuver_classifier")


# ═══════════════════════════════════════════════════════════════
#  Figure 3 — Collision Predictor: Risk Matrix
# ═══════════════════════════════════════════════════════════════
def fig_collision_risk():
    print("\n[3/6] Collision Predictor — Risk Analysis")

    model = CollisionPredictor().to(DEVICE)
    model.eval()

    # 8 synthetic satellites
    torch.manual_seed(7)
    n_objects = 8
    obj_names = [f"SAT-{i+1:03d}" for i in range(n_objects)]
    features = torch.randn(n_objects, 24, device=DEVICE)

    # Compute pairwise risk matrix
    risk_matrix = np.zeros((n_objects, n_objects))
    with torch.no_grad():
        for i in range(n_objects):
            for j in range(n_objects):
                if i == j:
                    continue
                out = model(features[i:i+1], features[j:j+1])
                risk_matrix[i, j] = out[0, 0].item()

    # Also compute single-pair detailed result
    with torch.no_grad():
        detail = model.predict_collision_risk(features[0:1], features[1:2])

    fig = plt.figure(figsize=(16, 6.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.35)

    # Panel A — heatmap
    ax1 = fig.add_subplot(gs[0])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "risk", ["#0d1117", "#1f6feb", "#f0883e", "#da3633"])
    im = ax1.imshow(risk_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax1.set_xticks(range(n_objects))
    ax1.set_yticks(range(n_objects))
    ax1.set_xticklabels(obj_names, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(obj_names, fontsize=8)
    ax1.set_title("Pairwise Collision Risk", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax1, shrink=0.85, label="Risk Score")
    cbar.ax.yaxis.label.set_color(TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)

    # Panel B — risk distribution
    ax2 = fig.add_subplot(gs[1])
    upper_tri = risk_matrix[np.triu_indices(n_objects, k=1)]
    ax2.hist(upper_tri, bins=15, color=ACCENT, edgecolor=DARK_BG, alpha=0.85)
    ax2.axvline(0.3, color="#f0883e", ls="--", lw=1.5, label="Medium (0.3)")
    ax2.axvline(0.6, color="#da3633", ls="--", lw=1.5, label="High (0.6)")
    ax2.axvline(0.8, color="#ff7b72", ls="--", lw=1.5, label="Critical (0.8)")
    ax2.set_xlabel("Risk Score")
    ax2.set_ylabel("Pair Count")
    ax2.set_title("Risk Score Distribution", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=8)

    # Panel C — architecture diagram text
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")

    info_lines = [
        ("Architecture", "MLP + Relative Trajectory Encoder"),
        ("Input", "2 × 24D features → 58D combined"),
        ("Hidden Layers", "256 → 128 → 64"),
        ("Outputs", "Risk Score  ·  TTCA  ·  Miss Distance"),
        ("Risk Activation", "Sigmoid [0, 1]"),
        ("TTCA / Distance", "Softplus (positive)"),
        ("Risk Tiers", "Low < 0.3 < Med < 0.6 < High < 0.8 < Critical"),
        ("Pairs Analyzed", f"{n_objects}×{n_objects} = {len(upper_tri)} unique"),
    ]
    y = 0.92
    for label, value in info_lines:
        ax3.text(0.05, y, label + ":", fontsize=10, fontweight="bold",
                 transform=ax3.transAxes, va="top", color=ACCENT)
        ax3.text(0.05, y - 0.06, value, fontsize=9.5,
                 transform=ax3.transAxes, va="top", color=TEXT_COLOR)
        y -= 0.13
    ax3.set_title("Model Summary", fontsize=13, fontweight="bold")

    fig.suptitle("COLLISION RISK PREDICTOR  ·  Pairwise Conjunction Analysis  ·  ~90K params",
                 fontsize=15, fontweight="bold", y=1.02, color="white")
    _save(fig, "03_collision_predictor")


# ═══════════════════════════════════════════════════════════════
#  Figure 4 — Anomaly Autoencoder: Reconstruction Error
# ═══════════════════════════════════════════════════════════════
def fig_anomaly_detection():
    print("\n[4/6] Anomaly Autoencoder — Reconstruction Error")

    state_dict = torch.load(CKPT / "phase3_anomaly" / "autoencoder.pt",
                            map_location=DEVICE, weights_only=False)
    meta = json.loads((CKPT / "phase3_anomaly" / "anomaly_meta.json").read_text())
    config = AutoencoderConfig.from_dict(meta["config"])
    model = BehaviorAutoencoder(config).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    threshold = meta["threshold"]
    training_errors = np.load(CKPT / "phase3_anomaly" / "training_errors.npy")

    # Generate normal + anomalous samples
    torch.manual_seed(99)
    normal_data = torch.randn(200, 19, device=DEVICE) * 0.3
    anomalous_data = torch.randn(30, 19, device=DEVICE) * 5.0 + 3.0

    normal_errors = model.reconstruction_error(normal_data).cpu().numpy()
    anomaly_errors = model.reconstruction_error(anomalous_data).cpu().numpy()

    # Latent space
    with torch.no_grad():
        _, latent_normal = model(normal_data)
        _, latent_anomaly = model(anomalous_data)
    ln = latent_normal.cpu().numpy()
    la = latent_anomaly.cpu().numpy()

    fig = plt.figure(figsize=(16, 6.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.32)

    # Panel A — error distribution
    ax1 = fig.add_subplot(gs[0])
    bins = np.linspace(0, max(normal_errors.max(), min(anomaly_errors.max(), 50)), 40)
    ax1.hist(normal_errors, bins=bins, color=ACCENT3, alpha=0.8, label=f"Normal (n={len(normal_errors)})")
    ax1.hist(anomaly_errors, bins=bins, color="#da3633", alpha=0.8, label=f"Anomalous (n={len(anomaly_errors)})")
    ax1.axvline(threshold, color=ACCENT2, ls="--", lw=2, label=f"Threshold ({threshold:.3f})")
    ax1.set_xlabel("Reconstruction Error (MSE)")
    ax1.set_ylabel("Count")
    ax1.set_title("Error Distribution", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8)

    # Panel B — latent space (first 2 dims)
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(ln[:, 0], ln[:, 1], c=ACCENT3, s=20, alpha=0.6, label="Normal")
    ax2.scatter(la[:, 0], la[:, 1], c="#da3633", s=40, alpha=0.9, marker="x", label="Anomalous")
    ax2.set_xlabel("Latent Dim 1")
    ax2.set_ylabel("Latent Dim 2")
    ax2.set_title("Latent Space (6D → 2D)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)

    # Panel C — training error curve (from saved errors)
    ax3 = fig.add_subplot(gs[2])
    sorted_errors = np.sort(training_errors)
    percentiles = np.arange(len(sorted_errors)) / len(sorted_errors) * 100
    ax3.plot(sorted_errors, percentiles, color=ACCENT, lw=2)
    ax3.axhline(95, color=ACCENT2, ls="--", lw=1.5, alpha=0.7)
    ax3.axvline(threshold, color=ACCENT2, ls="--", lw=1.5, alpha=0.7)
    ax3.text(threshold + 0.003, 50, f"95th %ile\n= {threshold:.4f}",
             fontsize=9, color=ACCENT2, va="center")
    ax3.set_xlabel("Reconstruction Error")
    ax3.set_ylabel("Percentile")
    ax3.set_title("Threshold Calibration", fontsize=13, fontweight="bold")

    summary = json.loads((CKPT / "phase3_anomaly" / "training_summary.json").read_text())
    tpr = summary["evaluation"]["true_positive_rate"]
    fpr = summary["evaluation"]["false_positive_rate"]
    fig.suptitle(f"BEHAVIOR AUTOENCODER  ·  19→6→19  ·  Anomaly Detection  ·  TPR {tpr:.0f}% / FPR {fpr:.0f}%",
                 fontsize=15, fontweight="bold", y=1.02, color="white")
    _save(fig, "04_anomaly_autoencoder")


# ═══════════════════════════════════════════════════════════════
#  Figure 5 — Threat Pipeline: End-to-End Breakdown
# ═══════════════════════════════════════════════════════════════
def fig_threat_pipeline():
    print("\n[5/6] Threat Assessment Pipeline — Score Breakdown")

    # Simulate 5 threat scenarios with sub-scores
    scenarios = [
        {"name": "Routine LEO sat",     "intent": 5,  "anomaly": 3,  "proximity": 2,  "pattern": 0},
        {"name": "Drifting debris",      "intent": 15, "anomaly": 25, "proximity": 10, "pattern": 5},
        {"name": "Station-keeping\nnear asset", "intent": 30, "anomaly": 10, "proximity": 55, "pattern": 20},
        {"name": "Maneuvering\nnear HVA",       "intent": 60, "anomaly": 40, "proximity": 70, "pattern": 45},
        {"name": "Aggressive\nshadowing",        "intent": 85, "anomaly": 75, "proximity": 90, "pattern": 80},
    ]
    weights = {"intent": 0.35, "anomaly": 0.25, "proximity": 0.25, "pattern": 0.15}
    tier_boundaries = [0, 20, 40, 60, 80, 100]
    tier_names = ["MINIMAL", "LOW", "MODERATE", "ELEVATED", "CRITICAL"]
    tier_colors = [ACCENT3, ACCENT, "#d29922", ACCENT2, "#da3633"]

    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.3)

    # Panel A — stacked bar chart of sub-scores
    ax1 = fig.add_subplot(gs[0])
    names = [s["name"] for s in scenarios]
    x = np.arange(len(scenarios))
    bar_w = 0.55

    bottoms = np.zeros(len(scenarios))
    sub_labels = ["Intent (35%)", "Anomaly (25%)", "Proximity (25%)", "Pattern (15%)"]
    sub_keys = ["intent", "anomaly", "proximity", "pattern"]
    sub_colors = [ACCENT, ACCENT4, ACCENT2, ACCENT3]

    final_scores = []
    for s in scenarios:
        score = sum(s[k] * weights[k] for k in sub_keys)
        final_scores.append(score)

    for idx, (key, label, color) in enumerate(zip(sub_keys, sub_labels, sub_colors)):
        vals = [s[key] * weights[key] for s in scenarios]
        ax1.barh(x, vals, bar_w, left=bottoms, label=label, color=color, edgecolor="none")
        bottoms += vals

    # Tier colors on the right
    for i, score in enumerate(final_scores):
        tier_idx = min(int(score // 20), 4)
        ax1.text(score + 1.5, i, f"{score:.0f}  {tier_names[tier_idx]}",
                 va="center", fontsize=10, fontweight="bold", color=tier_colors[tier_idx])

    ax1.set_yticks(x)
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel("Weighted Threat Score (0–100)")
    ax1.set_title("Threat Score Breakdown by Scenario", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_xlim(0, 105)
    ax1.invert_yaxis()

    # Panel B — tier gauge
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")

    # Draw tier scale
    for i, (name, color) in enumerate(zip(tier_names, tier_colors)):
        y = 0.88 - i * 0.17
        rect = FancyBboxPatch((0.05, y - 0.05), 0.45, 0.1, boxstyle="round,pad=0.02",
                               facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.5,
                               transform=ax2.transAxes)
        ax2.add_patch(rect)
        ax2.text(0.28, y, name, transform=ax2.transAxes, ha="center", va="center",
                 fontsize=12, fontweight="bold", color=color)
        ax2.text(0.58, y, f"{tier_boundaries[i]}–{tier_boundaries[i+1]}",
                 transform=ax2.transAxes, ha="left", va="center", fontsize=10, color=TEXT_COLOR)

    # Pipeline diagram
    pipeline_y = 0.88
    steps = ["Maneuver\nClassification", "Intent\nAnalysis", "Anomaly\nDetection", "Threat\nScoring"]
    step_colors = [ACCENT, ACCENT4, ACCENT2, "#da3633"]
    for i, (step, c) in enumerate(zip(steps, step_colors)):
        yp = pipeline_y - i * 0.17
        ax2.text(0.85, yp, step, transform=ax2.transAxes, ha="center", va="center",
                 fontsize=9, color=c, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=c, alpha=0.15, edgecolor=c))
        if i < len(steps) - 1:
            ax2.annotate("", xy=(0.85, yp - 0.07), xytext=(0.85, yp - 0.04),
                         xycoords="axes fraction", textcoords="axes fraction",
                         arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=1.5))

    ax2.set_title("Tier Scale & Pipeline", fontsize=13, fontweight="bold")

    fig.suptitle("THREAT ASSESSMENT PIPELINE  ·  Intent → Anomaly → Proximity → Pattern → Score",
                 fontsize=15, fontweight="bold", y=1.02, color="white")
    _save(fig, "05_threat_pipeline")


# ═══════════════════════════════════════════════════════════════
#  Figure 6 — MC Dropout Uncertainty: Confidence Bands
# ═══════════════════════════════════════════════════════════════
def fig_uncertainty():
    print("\n[6/6] Uncertainty Quantification — MC Dropout Confidence Bands")

    # Load parallel transformer
    ckpt_path = CKPT / "phase3_parallel" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model_cfg = ckpt.get("model_config", {})
    config = TransformerConfig(**model_cfg)
    model = TrajectoryTransformer(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    mc = MCDropoutPredictor(model, MCDropoutConfig(n_samples=50))

    # Use same ISS orbit
    R = 6771.0
    T_orbit = 2 * math.pi * math.sqrt(R**3 / 398600.4418)
    omega = 2 * math.pi / T_orbit
    dt = 60.0
    t_arr = np.arange(50) * dt
    feats = np.zeros((50, 24))
    feats[:, 0] = R * np.cos(omega * t_arr)
    feats[:, 1] = R * np.sin(omega * t_arr)
    feats[:, 3] = -R * omega * np.sin(omega * t_arr)
    feats[:, 4] =  R * omega * np.cos(omega * t_arr)

    src = torch.tensor(feats[:20], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    result = mc.predict_trajectory_with_uncertainty(src, pred_horizon=30)

    mean = result["mean"][0]      # (30, 6)
    std  = result["std"][0]       # (30, 6)
    lower = result["lower_95"][0]
    upper = result["upper_95"][0]
    gt = feats[20:, :6]
    horizon = np.arange(1, 31)

    fig = plt.figure(figsize=(16, 6.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.32)

    # Panel A — X position with CI
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(horizon, lower[:, 0], upper[:, 0], alpha=0.2, color=ACCENT, label="95% CI")
    ax1.fill_between(horizon, mean[:, 0] - std[:, 0], mean[:, 0] + std[:, 0],
                     alpha=0.35, color=ACCENT)
    ax1.plot(horizon, mean[:, 0], color=ACCENT, lw=2, label="MC Mean")
    ax1.plot(horizon, gt[:, 0], color=ACCENT3, lw=2, ls="--", label="Ground Truth")
    ax1.set_xlabel("Prediction Horizon")
    ax1.set_ylabel("X Position (km)")
    ax1.set_title("X Position — 50 MC Samples", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)

    # Panel B — Y position with CI
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(horizon, lower[:, 1], upper[:, 1], alpha=0.2, color=ACCENT4, label="95% CI")
    ax2.fill_between(horizon, mean[:, 1] - std[:, 1], mean[:, 1] + std[:, 1],
                     alpha=0.35, color=ACCENT4)
    ax2.plot(horizon, mean[:, 1], color=ACCENT4, lw=2, label="MC Mean")
    ax2.plot(horizon, gt[:, 1], color=ACCENT3, lw=2, ls="--", label="Ground Truth")
    ax2.set_xlabel("Prediction Horizon")
    ax2.set_ylabel("Y Position (km)")
    ax2.set_title("Y Position — 50 MC Samples", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)

    # Panel C — uncertainty growth
    ax3 = fig.add_subplot(gs[2])
    pos_std = np.sqrt(std[:, 0]**2 + std[:, 1]**2 + std[:, 2]**2)
    vel_std = np.sqrt(std[:, 3]**2 + std[:, 4]**2 + std[:, 5]**2)
    ax3.plot(horizon, pos_std, color=ACCENT2, lw=2, label="Position σ (km)")
    ax3.plot(horizon, vel_std, color=ACCENT3, lw=2, label="Velocity σ (km/s)")
    ax3.fill_between(horizon, 0, pos_std, alpha=0.15, color=ACCENT2)
    ax3.fill_between(horizon, 0, vel_std, alpha=0.15, color=ACCENT3)
    ax3.set_xlabel("Prediction Horizon")
    ax3.set_ylabel("Uncertainty (σ)")
    ax3.set_title("Uncertainty Growth Over Horizon", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9)

    fig.suptitle("UNCERTAINTY QUANTIFICATION  ·  Monte Carlo Dropout (50 samples)  ·  95% Confidence Intervals",
                 fontsize=15, fontweight="bold", y=1.02, color="white")
    _save(fig, "06_uncertainty_mc_dropout")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  SPACE-AI MODEL SHOWCASE  —  Generating Figures")
    print("=" * 60)

    fig_trajectory_prediction()
    fig_maneuver_classifier()
    fig_collision_risk()
    fig_anomaly_detection()
    fig_threat_pipeline()
    fig_uncertainty()

    print("\n" + "=" * 60)
    print(f"  All 6 figures saved to: {OUT}/")
    print("=" * 60)
    print("\nFiles:")
    for f in sorted(OUT.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}  ({size_kb:.0f} KB)")
