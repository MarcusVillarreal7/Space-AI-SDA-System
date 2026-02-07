#!/usr/bin/env python3
"""
Train Trajectory Transformer on Scaled Dataset (1.4M sequences).

This script implements Stage 4 of the Performance Optimization plan:
- Load 1.4M sequences from chunked features
- Train Trajectory Transformer with GPU acceleration
- Save checkpoints and training history
- Enable comparison with baseline model (88 sequences)

Memory-safe chunked loading prevents RAM exhaustion.

Author: Space AI Team
Date: 2026-02-07
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import click
import json
from tqdm import tqdm
from datetime import datetime
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.models.trajectory_transformer import TrajectoryTransformer, TransformerConfig
from src.ml.training.trainer import Trainer, TrainerConfig
from src.ml.training.losses import create_trajectory_loss
from src.utils.logging_config import get_logger

logger = get_logger("train_trajectory")


class ChunkedTrajectoryDataset(Dataset):
    """
    Memory-efficient dataset loader for chunked features.
    
    Loads chunks on-demand to prevent RAM exhaustion.
    """
    
    def __init__(
        self,
        chunk_dir: Path,
        split: str = 'train',
        train_ratio: float = 0.8
    ):
        """
        Initialize chunked dataset.
        
        Args:
            chunk_dir: Directory containing feature chunks
            split: 'train' or 'val'
            train_ratio: Fraction of data for training
        """
        self.chunk_dir = Path(chunk_dir)
        self.split = split
        self.train_ratio = train_ratio
        
        # Load metadata
        metadata_path = self.chunk_dir / 'metadata.json'
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        self.total_sequences = self.metadata['totals']['total_sequences']
        self.num_chunks = self.metadata['totals'].get('num_objects', 1000) // 100
        
        # Determine split
        split_idx = int(self.total_sequences * self.train_ratio)
        
        if split == 'train':
            self.start_idx = 0
            self.end_idx = split_idx
        else:  # val
            self.start_idx = split_idx
            self.end_idx = self.total_sequences
        
        self.length = self.end_idx - self.start_idx
        
        # Cache for current chunk
        self.current_chunk = None
        self.current_chunk_idx = -1
        
        logger.info(f"{split} dataset: {self.length:,} sequences ({split_idx=}, {self.end_idx=})")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        Get a single sequence.
        
        Args:
            idx: Index in split
        
        Returns:
            Dictionary with 'history' and 'target' tensors
        """
        # Convert to global index
        global_idx = self.start_idx + idx
        
        # Determine which chunk this belongs to
        sequences_per_chunk = self.metadata['chunks'][0]['num_sequences']
        chunk_idx = global_idx // sequences_per_chunk
        local_idx = global_idx % sequences_per_chunk
        
        # Load chunk if not cached
        if chunk_idx != self.current_chunk_idx:
            self._load_chunk(chunk_idx)
        
        # Extract sequence
        sequence = self.current_chunk[local_idx]
        
        # Split into history and target
        # Assuming sequence format: {'history': (seq_len, D), 'target': (pred_horizon, D)}
        if isinstance(sequence, dict):
            return sequence
        else:
            # If raw tensor, split manually (assume first 20 = history, next 30 = target)
            # This depends on how features were saved
            return {
                'history': sequence['history'],
                'target': sequence['target']
            }
    
    def _load_chunk(self, chunk_idx: int):
        """Load a feature chunk into memory."""
        chunk_file = self.chunk_dir / f'features_chunk_{chunk_idx:04d}.pt'
        
        logger.info(f"Loading chunk {chunk_idx}: {chunk_file}")
        self.current_chunk = torch.load(chunk_file, map_location='cpu')
        self.current_chunk_idx = chunk_idx
        
        # Force garbage collection
        gc.collect()


class TrajectoryTransformerTrainer(Trainer):
    """
    Specialized trainer for Trajectory Transformer.
    
    Handles sequence-to-sequence prediction with custom forward pass.
    """
    
    def _forward_pass(self, batch):
        """
        Forward pass for trajectory transformer.
        
        Args:
            batch: Dictionary with 'history' and 'target'
        
        Returns:
            (loss, predictions, targets)
        """
        src = batch['history']  # (batch, src_len, input_dim)
        tgt = batch['target']  # (batch, tgt_len, output_dim)
        
        # Forward pass
        # Model expects (batch, tgt_len, input_dim) for decoder input
        # Use teacher forcing: use actual target as decoder input
        if tgt.size(-1) < src.size(-1):
            # Pad target to match input dimension
            tgt_input = torch.zeros(tgt.size(0), tgt.size(1), src.size(-1), device=src.device)
            tgt_input[..., :tgt.size(-1)] = tgt
        else:
            tgt_input = tgt
        
        predictions = self.model(src, tgt_input)
        
        # Compute loss (predictions vs target)
        # Target should be position + velocity (6D)
        if tgt.size(-1) != predictions.size(-1):
            # Extract position and velocity from target if needed
            tgt = tgt[..., :predictions.size(-1)]
        
        loss = self.criterion(predictions, tgt)
        
        return loss, predictions, tgt


@click.command()
@click.option('--data', type=click.Path(exists=True), required=True,
              help='Path to chunked features directory')
@click.option('--output', type=click.Path(), default='checkpoints/phase3_scaled',
              help='Output directory for checkpoints')
@click.option('--epochs', type=int, default=20,
              help='Number of training epochs')
@click.option('--batch-size', type=int, default=32,
              help='Training batch size')
@click.option('--lr', type=float, default=1e-4,
              help='Learning rate')
@click.option('--device', type=str, default='cuda',
              help='Device to train on (cuda or cpu)')
@click.option('--workers', type=int, default=4,
              help='Number of data loading workers')
@click.option('--resume', type=click.Path(exists=True), default=None,
              help='Resume from checkpoint')
@click.option('--save-every', type=int, default=5,
              help='Save checkpoint every N epochs')
def main(data, output, epochs, batch_size, lr, device, workers, resume, save_every):
    """
    Train Trajectory Transformer on scaled dataset (1.4M sequences).
    
    Example:
        python train_trajectory_scaled.py \\
            --data data/processed/features_1k_chunked \\
            --output checkpoints/phase3_scaled \\
            --epochs 20 \\
            --device cuda
    """
    logger.info("="*70)
    logger.info("STAGE 4: TRAINING TRANSFORMER ON SCALED DATASET")
    logger.info("="*70)
    logger.info(f"Data: {data}")
    logger.info(f"Output: {output}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Device: {device}")
    logger.info("")
    
    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = ChunkedTrajectoryDataset(data, split='train', train_ratio=0.8)
    val_dataset = ChunkedTrajectoryDataset(data, split='val', train_ratio=0.8)
    
    logger.info(f"Train: {len(train_dataset):,} sequences")
    logger.info(f"Val: {len(val_dataset):,} sequences")
    logger.info("")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device == 'cuda')
    )
    
    logger.info(f"Train batches: {len(train_loader):,}")
    logger.info(f"Val batches: {len(val_loader):,}")
    logger.info("")
    
    # Create model
    logger.info("Creating model...")
    model_config = TransformerConfig(
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=256,
        dropout=0.1,
        input_dim=24,
        output_dim=6
    )
    
    model = TrajectoryTransformer(model_config)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info("")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5
    )
    
    # Create loss function
    criterion = create_trajectory_loss(
        loss_type='weighted_mse',
        position_weight=1.0,
        velocity_weight=0.1
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.01
    )
    
    # Create trainer
    trainer_config = TrainerConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        device=device,
        checkpoint_dir=str(output_dir),
        save_every=save_every,
        early_stopping_patience=10,
        log_interval=100
    )
    
    trainer = TrajectoryTransformerTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config=trainer_config,
        scheduler=scheduler
    )
    
    # Resume from checkpoint if specified
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
        trainer.load_checkpoint(resume)
    
    # Train
    logger.info("Starting training...")
    logger.info("="*70)
    history = trainer.train(train_loader, val_loader)
    
    # Summary
    logger.info("="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.2f}")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.2f}")
    logger.info(f"Final val loss: {history['val_loss'][-1]:.2f}")
    logger.info(f"Checkpoints saved to: {output_dir}")
    logger.info("="*70)
    
    # Save final summary
    summary = {
        'config': {
            'data_dir': str(data),
            'output_dir': str(output_dir),
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'device': device
        },
        'dataset': {
            'train_sequences': len(train_dataset),
            'val_sequences': len(val_dataset),
            'train_batches': len(train_loader),
            'val_batches': len(val_loader)
        },
        'model': {
            'parameters': sum(p.numel() for p in model.parameters()),
            'config': model_config.__dict__
        },
        'results': {
            'best_val_loss': trainer.best_val_loss,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'epochs_trained': len(history['train_loss'])
        },
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = output_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("\n✅ Training complete! Ready for Stage 5 (Evaluation)")


if __name__ == '__main__':
    main()
