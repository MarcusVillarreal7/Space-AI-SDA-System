#!/usr/bin/env python3
"""
Train Trajectory Transformer with Parallel Prediction Head.

Fixes the autoregressive train/inference gap by adding a DETR-style parallel
prediction head.  Training and inference share the exact same code path.

Strategy:
  1. Create model with use_parallel_decoder=True
  2. Load encoder weights from the scaled checkpoint (best_model.pt)
  3. Phased training:
     - Epochs 1-3: Freeze encoder, train prediction head only
     - Epochs 4+:  Unfreeze encoder, fine-tune end-to-end (encoder LR × 0.1)

Author: Space AI Team
Date: 2026-02-07
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
from tqdm import tqdm
import gc

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.models.trajectory_transformer import TrajectoryTransformer, TransformerConfig
from src.ml.training.losses import create_trajectory_loss
from src.utils.logging_config import get_logger

logger = get_logger("train_parallel")


class PreloadedDataset(Dataset):
    """
    RAM-preloaded dataset for maximum GPU throughput.

    Loads all chunks once, concatenates into two contiguous tensors
    (~4.5 GB total), then serves samples with zero disk I/O.
    """

    def __init__(self, chunk_dir, split="train", train_ratio=0.8, input_dim=24, output_dim=6):
        chunk_dir = Path(chunk_dir)

        histories, targets = [], []
        chunk_files = sorted(chunk_dir.glob("features_chunk_*.pt"))

        for chunk_file in chunk_files:
            logger.info(f"Loading {chunk_file.name} ...")
            chunk = torch.load(chunk_file, map_location="cpu", weights_only=False)
            for obj_key in sorted(chunk.keys(), key=int):
                obj = chunk[obj_key]
                histories.append(obj["history"][:, :, :input_dim])
                targets.append(obj["target"][:, :, :output_dim])
            del chunk
            gc.collect()

        all_history = torch.cat(histories, dim=0)  # (N, 20, 24)
        all_target = torch.cat(targets, dim=0)      # (N, 30, 6)
        del histories, targets
        gc.collect()

        total = all_history.shape[0]
        split_idx = int(total * train_ratio)

        if split == "train":
            self.history = all_history[:split_idx]
            self.target = all_target[:split_idx]
        else:
            self.history = all_history[split_idx:]
            self.target = all_target[split_idx:]

        logger.info(f"{split} dataset: {len(self):,} sequences  "
                     f"history={self.history.shape}  target={self.target.shape}")

    def __len__(self):
        return self.history.shape[0]

    def __getitem__(self, idx):
        return {"history": self.history[idx], "target": self.target[idx]}


def load_encoder_weights(model: TrajectoryTransformer, checkpoint_path: str, device: torch.device):
    """
    Load encoder (and shared) weights from a pretrained checkpoint.

    Copies: input_projection, pos_encoding, encoder_layers, (output_projection skipped
    because the parallel head has its own).  Decoder layers are left at init values
    since the parallel head has fresh decoder layers.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pretrained_sd = ckpt.get("model_state_dict", ckpt)

    # Filter to encoder-related keys
    encoder_prefixes = ("input_projection.", "pos_encoding.", "encoder_layers.")
    encoder_sd = {k: v for k, v in pretrained_sd.items() if k.startswith(encoder_prefixes)}

    missing, unexpected = model.load_state_dict(encoder_sd, strict=False)
    logger.info(f"Loaded {len(encoder_sd)} encoder weight tensors from {checkpoint_path}")
    logger.info(f"  Missing (expected — new head): {len(missing)} keys")
    if unexpected:
        logger.warning(f"  Unexpected keys: {unexpected}")


def create_param_groups(model: TrajectoryTransformer, head_lr: float, encoder_lr_scale: float = 0.1):
    """
    Create parameter groups with differential learning rates.

    Returns two groups:
      - Encoder params: LR = head_lr * encoder_lr_scale
      - Parallel head params: LR = head_lr
    """
    encoder_prefixes = ("input_projection.", "pos_encoding.", "encoder_layers.")
    encoder_params = []
    head_params = []

    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in encoder_prefixes):
            encoder_params.append(param)
        else:
            head_params.append(param)

    return [
        {"params": encoder_params, "lr": head_lr * encoder_lr_scale, "name": "encoder"},
        {"params": head_params, "lr": head_lr, "name": "head"},
    ]


def freeze_encoder(model: TrajectoryTransformer):
    """Freeze encoder weights (input projection, positional encoding, encoder layers)."""
    encoder_prefixes = ("input_projection.", "pos_encoding.", "encoder_layers.")
    frozen = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in encoder_prefixes):
            param.requires_grad = False
            frozen += 1
    logger.info(f"Froze {frozen} encoder parameter tensors")


def unfreeze_encoder(model: TrajectoryTransformer):
    """Unfreeze encoder weights."""
    unfrozen = 0
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen += 1
    logger.info(f"Unfroze {unfrozen} parameter tensors")


def train_one_epoch(
    model, train_loader, optimizer, criterion, device, epoch, grad_clip=1.0
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
        for batch in pbar:
            src = batch["history"].to(device)
            tgt = batch["target"].to(device)
            # Trim target to output_dim if needed
            tgt = tgt[..., :model.config.output_dim]

            optimizer.zero_grad()
            pred = model.forward_parallel(src)
            loss = criterion(pred, tgt)
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        src = batch["history"].to(device)
        tgt = batch["target"].to(device)
        tgt = tgt[..., :model.config.output_dim]

        pred = model.forward_parallel(src)
        loss = criterion(pred, tgt)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train parallel prediction head")
    parser.add_argument("--data", default="data/processed/features_1k_chunked",
                        help="Chunked features directory")
    parser.add_argument("--pretrained", default="checkpoints/phase3_scaled/best_model.pt",
                        help="Pretrained scaled model checkpoint")
    parser.add_argument("--output", default="checkpoints/phase3_parallel",
                        help="Output checkpoint directory")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total training epochs")
    parser.add_argument("--freeze-epochs", type=int, default=3,
                        help="Epochs to freeze encoder")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Head learning rate")
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1,
                        help="Encoder LR multiplier relative to head LR")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Datasets ----
    logger.info("=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)
    train_dataset = PreloadedDataset(args.data, split="train")
    val_dataset = PreloadedDataset(args.data, split="val")
    logger.info(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}")

    # Workers=0 since data is preloaded in RAM — no disk I/O bottleneck
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    logger.info("=" * 70)
    logger.info("CREATING MODEL")
    logger.info("=" * 70)
    config = TransformerConfig(
        d_model=64, n_heads=4, n_encoder_layers=2, n_decoder_layers=2,
        d_ff=256, dropout=0.1, input_dim=24, output_dim=6,
        pred_horizon=30, use_parallel_decoder=True,
    )
    model = TrajectoryTransformer(config)

    # Load pretrained encoder
    load_encoder_weights(model, args.pretrained, device)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in model.parallel_head.parameters())
    logger.info(f"Total params: {total_params:,}  (head: {head_params:,})")

    # ---- Loss ----
    criterion = create_trajectory_loss("weighted_mse", position_weight=1.0, velocity_weight=0.1)

    # ---- Training ----
    logger.info("=" * 70)
    logger.info("TRAINING")
    logger.info("=" * 70)

    # MLflow experiment tracking
    if HAS_MLFLOW:
        mlflow.set_experiment("sda-trajectory-transformer")
        mlflow.start_run()
        mlflow.log_params({
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "freeze_epochs": args.freeze_epochs,
            "encoder_lr_scale": args.encoder_lr_scale,
            "total_params": total_params,
            "head_params": head_params,
        })

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        # Phase switching
        if epoch <= args.freeze_epochs:
            if epoch == 1:
                freeze_encoder(model)
                # Only optimize head params during freeze phase
                optimizer = optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=args.lr, weight_decay=1e-5,
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
                )
        else:
            if epoch == args.freeze_epochs + 1:
                unfreeze_encoder(model)
                # Switch to differential LR groups
                param_groups = create_param_groups(model, args.lr, args.encoder_lr_scale)
                optimizer = optim.AdamW(param_groups, weight_decay=1e-5)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - args.freeze_epochs,
                    eta_min=args.lr * 0.01,
                )

        phase = "FROZEN" if epoch <= args.freeze_epochs else "FULL"
        logger.info(f"\n--- Epoch {epoch}/{args.epochs} [{phase}] ---")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
        )
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={current_lr:.6f}"
        )

        if HAS_MLFLOW:
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "lr": current_lr},
                step=epoch,
            )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_config": model.get_config(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            logger.info(f"  -> New best model saved (val_loss={best_val_loss:.6f})")

        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_config": model.get_config(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                output_dir / f"checkpoint_epoch{epoch}.pt",
            )

    # Save final
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
            "best_val_loss": best_val_loss,
            "history": history,
        },
        output_dir / "final_model.pt",
    )

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Summary
    summary = {
        "config": vars(args),
        "model": {"total_params": total_params, "head_params": head_params, "config": config.__dict__},
        "results": {
            "best_val_loss": best_val_loss,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "epochs_trained": args.epochs,
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if HAS_MLFLOW:
        mlflow.log_metrics({"best_val_loss": best_val_loss})
        mlflow.log_artifact(str(output_dir / "training_summary.json"))
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best val loss: {best_val_loss:.6f}")
    logger.info(f"Checkpoints: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
