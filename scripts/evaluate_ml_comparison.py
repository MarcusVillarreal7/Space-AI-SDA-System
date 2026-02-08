#!/usr/bin/env python3
"""
Evaluate and Compare Baseline vs Scaled Trajectory Transformer.

Comprehensive evaluation script that:
1. Loads baseline (88 samples) and scaled (1.4M sequences) model checkpoints
2. Evaluates both on held-out validation data from the 1K dataset
3. Computes position/velocity RMSE, MAE, and per-horizon error profiles
4. Runs MC Dropout uncertainty quantification on the scaled model
5. Generates comparison report (JSON + console summary)
6. Saves per-horizon error plots (matplotlib)

Author: Space AI Team
Date: 2026-02-07
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.models.trajectory_transformer import TrajectoryTransformer, TransformerConfig
from src.ml.uncertainty.monte_carlo import MCDropoutPredictor, MCDropoutConfig
from src.utils.logging_config import get_logger

logger = get_logger("evaluate_comparison")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> TrajectoryTransformer:
    """Load a TrajectoryTransformer from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = ckpt.get("model_config", {})
    model = TrajectoryTransformer.from_config(config_dict) if config_dict else TrajectoryTransformer()
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_val_sequences(
    chunk_dir: str,
    max_sequences: int = 10000,
    train_ratio: float = 0.8,
    input_dim: int = 24,
    output_dim: int = 6,
):
    """
    Load validation sequences from chunked features.

    Each chunk file is a dict mapping object_index -> {history, target, mask}.
    We flatten all objects' sequences and take the validation split (last 20%).

    Returns:
        history: (N, 20, input_dim) tensor
        target:  (N, 30, output_dim) tensor
    """
    chunk_dir = Path(chunk_dir)
    with open(chunk_dir / "metadata.json") as f:
        meta = json.load(f)

    total_sequences = meta["totals"]["total_sequences"]
    val_start = int(total_sequences * train_ratio)

    histories, targets = [], []
    running_idx = 0

    chunk_files = sorted(chunk_dir.glob("features_chunk_*.pt"))
    for chunk_file in chunk_files:
        logger.info(f"Loading {chunk_file.name} ...")
        chunk = torch.load(chunk_file, map_location="cpu", weights_only=False)

        for obj_key in sorted(chunk.keys(), key=int):
            obj = chunk[obj_key]
            n_seq = obj["history"].shape[0]
            seq_end = running_idx + n_seq

            # Compute overlap with val range
            overlap_start = max(running_idx, val_start)
            overlap_end = seq_end

            if overlap_start < overlap_end:
                local_start = overlap_start - running_idx
                local_end = overlap_end - running_idx
                h = obj["history"][local_start:local_end, :, :input_dim]
                t = obj["target"][local_start:local_end, :, :output_dim]
                histories.append(h)
                targets.append(t)

            running_idx = seq_end

            if sum(h_.shape[0] for h_ in histories) >= max_sequences:
                break

        del chunk
        gc.collect()

        if sum(h_.shape[0] for h_ in histories) >= max_sequences:
            break

    history = torch.cat(histories, dim=0)[:max_sequences]
    target = torch.cat(targets, dim=0)[:max_sequences]
    logger.info(f"Loaded {history.shape[0]:,} val sequences  history={history.shape}  target={target.shape}")
    return history, target


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Compute comprehensive trajectory metrics.

    Args:
        predictions: (N, T, 6) — first 3 = position, last 3 = velocity
        targets:     (N, T, 6)

    Returns:
        Dictionary of metric values.
    """
    pred_pos = predictions[..., :3]
    pred_vel = predictions[..., 3:6]
    tgt_pos = targets[..., :3]
    tgt_vel = targets[..., 3:6]

    # Overall RMSE / MAE
    pos_rmse = torch.sqrt(F.mse_loss(pred_pos, tgt_pos)).item()
    vel_rmse = torch.sqrt(F.mse_loss(pred_vel, tgt_vel)).item()
    pos_mae = F.l1_loss(pred_pos, tgt_pos).item()
    vel_mae = F.l1_loss(pred_vel, tgt_vel).item()

    # Per-timestep RMSE profile  (N, T, 3) -> (T,)
    per_t_pos = torch.sqrt(((pred_pos - tgt_pos) ** 2).mean(dim=(0, 2)))  # (T,)
    per_t_vel = torch.sqrt(((pred_vel - tgt_vel) ** 2).mean(dim=(0, 2)))

    T = per_t_pos.shape[0]
    initial_idx = slice(0, max(1, T // 6))         # first ~5 timesteps
    mid_idx = slice(T // 3, 2 * T // 3)            # middle third
    final_idx = slice(-max(1, T // 6), None)        # last ~5 timesteps

    # 3D Euclidean distance per sample per timestep
    pos_dist = torch.norm(pred_pos - tgt_pos, dim=-1)  # (N, T)
    vel_dist = torch.norm(pred_vel - tgt_vel, dim=-1)

    return {
        "position_rmse": pos_rmse,
        "position_mae": pos_mae,
        "velocity_rmse": vel_rmse,
        "velocity_mae": vel_mae,
        "position_rmse_initial": per_t_pos[initial_idx].mean().item(),
        "position_rmse_mid": per_t_pos[mid_idx].mean().item(),
        "position_rmse_final": per_t_pos[final_idx].mean().item(),
        "velocity_rmse_initial": per_t_vel[initial_idx].mean().item(),
        "velocity_rmse_mid": per_t_vel[mid_idx].mean().item(),
        "velocity_rmse_final": per_t_vel[final_idx].mean().item(),
        "mean_3d_pos_error": pos_dist.mean().item(),
        "median_3d_pos_error": pos_dist.median().item(),
        "p95_3d_pos_error": torch.quantile(pos_dist, 0.95).item(),
        "mean_3d_vel_error": vel_dist.mean().item(),
        "per_timestep_pos_rmse": per_t_pos.tolist(),
        "per_timestep_vel_rmse": per_t_vel.tolist(),
    }


def evaluate_model_teacher_forcing(
    model: TrajectoryTransformer,
    history: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """Evaluate with teacher forcing (same as training)."""
    model.eval()
    all_preds, all_tgts = [], []
    input_dim = model.config.input_dim

    with torch.no_grad():
        for i in range(0, history.shape[0], batch_size):
            h = history[i : i + batch_size].to(device)
            t = target[i : i + batch_size].to(device)

            # Pad target to input_dim for decoder input (teacher forcing)
            tgt_input = torch.zeros(t.shape[0], t.shape[1], input_dim, device=device)
            tgt_input[..., : t.shape[-1]] = t
            pred = model(h, tgt_input)  # (B, T, 6)

            all_preds.append(pred.cpu())
            all_tgts.append(t.cpu())

    preds = torch.cat(all_preds)
    tgts = torch.cat(all_tgts)
    return compute_metrics(preds, tgts)


def evaluate_model_autoregressive(
    model: TrajectoryTransformer,
    history: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> dict:
    """Evaluate with autoregressive (true inference) decoding."""
    model.eval()
    pred_horizon = target.shape[1]
    all_preds, all_tgts = [], []

    with torch.no_grad():
        for i in range(0, history.shape[0], batch_size):
            h = history[i : i + batch_size].to(device)
            t = target[i : i + batch_size]

            pred = model.predict(h, pred_horizon)  # (B, T, 6)
            all_preds.append(pred.cpu())
            all_tgts.append(t)

    preds = torch.cat(all_preds)
    tgts = torch.cat(all_tgts)
    return compute_metrics(preds, tgts)


def evaluate_uncertainty(
    model: TrajectoryTransformer,
    history: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    n_mc_samples: int = 20,
    batch_size: int = 128,
    max_sequences: int = 2000,
) -> dict:
    """
    Evaluate uncertainty calibration via MC Dropout on autoregressive predictions.

    Returns coverage statistics and sharpness measures.
    """
    mc_cfg = MCDropoutConfig(n_samples=n_mc_samples)
    mc = MCDropoutPredictor(model, mc_cfg)

    history = history[:max_sequences]
    target = target[:max_sequences]
    pred_horizon = target.shape[1]

    all_means, all_stds = [], []

    for i in tqdm(range(0, history.shape[0], batch_size), desc="MC Dropout"):
        h = history[i : i + batch_size].to(device)

        # Collect MC samples
        samples = []
        mc._enable_dropout()
        for _ in range(n_mc_samples):
            with torch.no_grad():
                pred = model.predict(h, pred_horizon)
            samples.append(pred.cpu())

        stacked = torch.stack(samples)  # (S, B, T, 6)
        all_means.append(stacked.mean(dim=0))
        all_stds.append(stacked.std(dim=0))

    means = torch.cat(all_means)  # (N, T, 6)
    stds = torch.cat(all_stds)

    # Coverage: fraction of targets within mean ± k*std
    tgt = target[..., :6]
    errors = torch.abs(tgt - means)

    coverage_1sigma = (errors < 1.0 * stds).float().mean().item()
    coverage_2sigma = (errors < 2.0 * stds).float().mean().item()
    coverage_3sigma = (errors < 3.0 * stds).float().mean().item()

    # Sharpness: average prediction interval width
    avg_std = stds.mean().item()
    avg_std_pos = stds[..., :3].mean().item()
    avg_std_vel = stds[..., 3:6].mean().item()

    model.eval()  # restore eval mode

    return {
        "n_mc_samples": n_mc_samples,
        "n_sequences_evaluated": means.shape[0],
        "coverage_1sigma": coverage_1sigma,
        "coverage_2sigma": coverage_2sigma,
        "coverage_3sigma": coverage_3sigma,
        "ideal_1sigma": 0.6827,
        "ideal_2sigma": 0.9545,
        "ideal_3sigma": 0.9973,
        "avg_prediction_std": avg_std,
        "avg_position_std": avg_std_pos,
        "avg_velocity_std": avg_std_vel,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_comparison_plots(baseline_metrics: dict, scaled_metrics: dict, output_dir: Path):
    """Generate and save comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-horizon position RMSE ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, title in [
        (axes[0], "per_timestep_pos_rmse", "Position RMSE per Timestep"),
        (axes[1], "per_timestep_vel_rmse", "Velocity RMSE per Timestep"),
    ]:
        if key in baseline_metrics and key in scaled_metrics:
            b = baseline_metrics[key]
            s = scaled_metrics[key]
            ax.plot(range(len(b)), b, "r-o", markersize=3, label="Baseline (88 seq)")
            ax.plot(range(len(s)), s, "b-o", markersize=3, label="Scaled (1.4M seq)")
            ax.set_xlabel("Prediction Timestep")
            ax.set_ylabel("RMSE (normalized)")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "per_horizon_comparison.png", dpi=150)
    plt.close()
    logger.info(f"Saved per_horizon_comparison.png")

    # --- Training loss curves ---
    fig, ax = plt.subplots(figsize=(8, 5))
    baseline_hist_path = project_root / "checkpoints" / "phase3_day3" / "training_history.json"
    scaled_hist_path = project_root / "checkpoints" / "phase3_scaled" / "training_history.json"

    if baseline_hist_path.exists() and scaled_hist_path.exists():
        with open(baseline_hist_path) as f:
            bh = json.load(f)
        with open(scaled_hist_path) as f:
            sh = json.load(f)

        # Baseline val loss (very large values, plot on log scale)
        ax.semilogy(bh["val_loss"], "r-o", markersize=4, label="Baseline val loss")
        ax.semilogy(sh["val_loss"], "b-s", markersize=4, label="Scaled val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss (log)")
        ax.set_title("Training Convergence: Baseline vs Scaled")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_loss_comparison.png", dpi=150)
    plt.close()
    logger.info(f"Saved training_loss_comparison.png")

    # --- Summary bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metric_keys = [
        ("position_rmse", "Pos RMSE"),
        ("position_mae", "Pos MAE"),
        ("mean_3d_pos_error", "3D Pos Err"),
    ]
    x = np.arange(len(metric_keys))
    w = 0.35
    b_vals = [baseline_metrics.get(k, 0) for k, _ in metric_keys]
    s_vals = [scaled_metrics.get(k, 0) for k, _ in metric_keys]
    axes[0].bar(x - w / 2, b_vals, w, label="Baseline", color="salmon")
    axes[0].bar(x + w / 2, s_vals, w, label="Scaled", color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n for _, n in metric_keys])
    axes[0].set_ylabel("Error (normalized)")
    axes[0].set_title("Position Metrics")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    metric_keys_v = [
        ("velocity_rmse", "Vel RMSE"),
        ("velocity_mae", "Vel MAE"),
        ("mean_3d_vel_error", "3D Vel Err"),
    ]
    b_vals_v = [baseline_metrics.get(k, 0) for k, _ in metric_keys_v]
    s_vals_v = [scaled_metrics.get(k, 0) for k, _ in metric_keys_v]
    axes[1].bar(x - w / 2, b_vals_v, w, label="Baseline", color="salmon")
    axes[1].bar(x + w / 2, s_vals_v, w, label="Scaled", color="steelblue")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n for _, n in metric_keys_v])
    axes[1].set_ylabel("Error (normalized)")
    axes[1].set_title("Velocity Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "metric_summary_comparison.png", dpi=150)
    plt.close()
    logger.info(f"Saved metric_summary_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate baseline vs scaled Transformer")
    parser.add_argument("--data", default="data/processed/features_1k_chunked",
                        help="Path to chunked features directory")
    parser.add_argument("--baseline-ckpt", default="checkpoints/phase3_day3/best_model.pt",
                        help="Baseline model checkpoint")
    parser.add_argument("--scaled-ckpt", default="checkpoints/phase3_scaled/best_model.pt",
                        help="Scaled model checkpoint")
    parser.add_argument("--parallel-ckpt", default="checkpoints/phase3_parallel/best_model.pt",
                        help="Parallel prediction head model checkpoint")
    parser.add_argument("--output", default="results/phase3_evaluation",
                        help="Output directory for results")
    parser.add_argument("--max-sequences", type=int, default=10000,
                        help="Max validation sequences to evaluate")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for evaluation")
    parser.add_argument("--mc-samples", type=int, default=20,
                        help="MC Dropout samples for uncertainty")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--no-uncertainty", action="store_true",
                        help="Skip uncertainty evaluation")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Load data ----
    logger.info("=" * 70)
    logger.info("LOADING VALIDATION DATA")
    logger.info("=" * 70)
    history, target = load_val_sequences(
        args.data, max_sequences=args.max_sequences, input_dim=24, output_dim=6
    )

    # ---- Load models ----
    logger.info("=" * 70)
    logger.info("LOADING MODELS")
    logger.info("=" * 70)

    baseline_model = load_model(args.baseline_ckpt, device)
    logger.info(f"Baseline model loaded: {sum(p.numel() for p in baseline_model.parameters()):,} params")

    scaled_model = load_model(args.scaled_ckpt, device)
    logger.info(f"Scaled model loaded:   {sum(p.numel() for p in scaled_model.parameters()):,} params")

    # ---- Evaluate: Teacher Forcing ----
    logger.info("=" * 70)
    logger.info("EVALUATION: TEACHER FORCING")
    logger.info("=" * 70)

    logger.info("Evaluating baseline (teacher forcing) ...")
    baseline_tf = evaluate_model_teacher_forcing(baseline_model, history, target, device, args.batch_size)

    logger.info("Evaluating scaled (teacher forcing) ...")
    scaled_tf = evaluate_model_teacher_forcing(scaled_model, history, target, device, args.batch_size)

    # ---- Evaluate: Autoregressive ----
    logger.info("=" * 70)
    logger.info("EVALUATION: AUTOREGRESSIVE (TRUE INFERENCE)")
    logger.info("=" * 70)

    logger.info("Evaluating baseline (autoregressive) ...")
    baseline_ar = evaluate_model_autoregressive(baseline_model, history, target, device, args.batch_size)

    logger.info("Evaluating scaled (autoregressive) ...")
    scaled_ar = evaluate_model_autoregressive(scaled_model, history, target, device, args.batch_size)

    # ---- Evaluate: Parallel Prediction Head ----
    parallel_metrics = {}
    parallel_ckpt = Path(args.parallel_ckpt)
    if parallel_ckpt.exists():
        logger.info("=" * 70)
        logger.info("EVALUATION: PARALLEL PREDICTION HEAD")
        logger.info("=" * 70)
        parallel_model = load_model(str(parallel_ckpt), device)
        logger.info(f"Parallel model loaded: {sum(p.numel() for p in parallel_model.parameters()):,} params")
        logger.info(f"  Has parallel head: {parallel_model.parallel_head is not None}")
        # predict() auto-dispatches to parallel head
        parallel_metrics = evaluate_model_autoregressive(
            parallel_model, history, target, device, args.batch_size
        )
    else:
        logger.info(f"Parallel checkpoint not found at {parallel_ckpt}, skipping parallel eval")

    # ---- Uncertainty (scaled model only) ----
    uncertainty_results = {}
    if not args.no_uncertainty:
        logger.info("=" * 70)
        logger.info("UNCERTAINTY EVALUATION (MC Dropout)")
        logger.info("=" * 70)
        uncertainty_results = evaluate_uncertainty(
            scaled_model, history, target, device,
            n_mc_samples=args.mc_samples,
            batch_size=args.batch_size,
        )

    # ---- Compute improvement factors ----
    def improvement(baseline_val, scaled_val):
        if baseline_val == 0:
            return 0.0
        return ((baseline_val - scaled_val) / baseline_val) * 100.0

    improvements_tf = {
        k: improvement(baseline_tf[k], scaled_tf[k])
        for k in ["position_rmse", "velocity_rmse", "position_mae", "velocity_mae",
                   "mean_3d_pos_error", "mean_3d_vel_error"]
    }
    improvements_ar = {
        k: improvement(baseline_ar[k], scaled_ar[k])
        for k in ["position_rmse", "velocity_rmse", "position_mae", "velocity_mae",
                   "mean_3d_pos_error", "mean_3d_vel_error"]
    }

    # ---- Build report ----
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
            "n_val_sequences": history.shape[0],
            "history_len": history.shape[1],
            "pred_horizon": target.shape[1],
            "baseline_checkpoint": args.baseline_ckpt,
            "scaled_checkpoint": args.scaled_ckpt,
        },
        "teacher_forcing": {
            "baseline": {k: v for k, v in baseline_tf.items() if not k.startswith("per_timestep")},
            "scaled": {k: v for k, v in scaled_tf.items() if not k.startswith("per_timestep")},
            "improvement_pct": improvements_tf,
        },
        "autoregressive": {
            "baseline": {k: v for k, v in baseline_ar.items() if not k.startswith("per_timestep")},
            "scaled": {k: v for k, v in scaled_ar.items() if not k.startswith("per_timestep")},
            "improvement_pct": improvements_ar,
        },
        "parallel": {
            k: v for k, v in parallel_metrics.items() if not k.startswith("per_timestep")
        } if parallel_metrics else {},
        "uncertainty": uncertainty_results,
        "per_timestep": {
            "teacher_forcing": {
                "baseline_pos": baseline_tf.get("per_timestep_pos_rmse", []),
                "scaled_pos": scaled_tf.get("per_timestep_pos_rmse", []),
                "baseline_vel": baseline_tf.get("per_timestep_vel_rmse", []),
                "scaled_vel": scaled_tf.get("per_timestep_vel_rmse", []),
            },
            "autoregressive": {
                "baseline_pos": baseline_ar.get("per_timestep_pos_rmse", []),
                "scaled_pos": scaled_ar.get("per_timestep_pos_rmse", []),
                "baseline_vel": baseline_ar.get("per_timestep_vel_rmse", []),
                "scaled_vel": scaled_ar.get("per_timestep_vel_rmse", []),
            },
        },
    }

    # Save report
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved: {report_path}")

    # ---- Plots ----
    if not args.no_plots:
        save_comparison_plots(baseline_ar, scaled_ar, output_dir)

    # ---- Console summary ----
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY: BASELINE vs SCALED TRANSFORMER")
    print("=" * 70)
    print(f"  Validation sequences: {history.shape[0]:,}")
    print(f"  History / Prediction: {history.shape[1]} / {target.shape[1]} timesteps")
    print()

    print("  TEACHER FORCING:")
    print(f"  {'Metric':<25} {'Baseline':>12} {'Scaled':>12} {'Improvement':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    for k in ["position_rmse", "velocity_rmse", "position_mae", "velocity_mae",
              "mean_3d_pos_error"]:
        bv = baseline_tf[k]
        sv = scaled_tf[k]
        imp = improvements_tf.get(k, 0)
        print(f"  {k:<25} {bv:>12.6f} {sv:>12.6f} {imp:>+11.1f}%")

    print()
    print("  AUTOREGRESSIVE (TRUE INFERENCE):")
    print(f"  {'Metric':<25} {'Baseline':>12} {'Scaled':>12} {'Improvement':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    for k in ["position_rmse", "velocity_rmse", "position_mae", "velocity_mae",
              "mean_3d_pos_error"]:
        bv = baseline_ar[k]
        sv = scaled_ar[k]
        imp = improvements_ar.get(k, 0)
        print(f"  {k:<25} {bv:>12.6f} {sv:>12.6f} {imp:>+11.1f}%")

    if parallel_metrics:
        print()
        print("  PARALLEL PREDICTION HEAD:")
        print(f"  {'Metric':<25} {'Parallel':>12} {'Scaled TF':>12} {'Gap':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
        for k in ["position_rmse", "velocity_rmse", "position_mae", "velocity_mae",
                   "mean_3d_pos_error"]:
            pv = parallel_metrics.get(k, 0)
            sv = scaled_tf.get(k, 0)
            gap_pct = ((pv - sv) / sv * 100) if sv else 0
            print(f"  {k:<25} {pv:>12.6f} {sv:>12.6f} {gap_pct:>+11.1f}%")

    if uncertainty_results:
        print()
        print("  UNCERTAINTY CALIBRATION (MC Dropout):")
        print(f"  {'Metric':<25} {'Actual':>12} {'Ideal':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12}")
        print(f"  {'1-sigma coverage':<25} {uncertainty_results['coverage_1sigma']:>12.4f} {0.6827:>12.4f}")
        print(f"  {'2-sigma coverage':<25} {uncertainty_results['coverage_2sigma']:>12.4f} {0.9545:>12.4f}")
        print(f"  {'3-sigma coverage':<25} {uncertainty_results['coverage_3sigma']:>12.4f} {0.9973:>12.4f}")
        print(f"  {'Avg position std':<25} {uncertainty_results['avg_position_std']:>12.6f}")
        print(f"  {'Avg velocity std':<25} {uncertainty_results['avg_velocity_std']:>12.6f}")

    print()
    print(f"  Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
