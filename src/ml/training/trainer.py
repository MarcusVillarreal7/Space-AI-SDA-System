"""
Generic Training Loop for ML Models.

This module provides a flexible training framework with:
- Train/validation splits
- Checkpoint saving (best + periodic)
- Learning rate scheduling
- Early stopping
- Metric tracking and logging
- Progress visualization

Author: Space AI Team
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import json
from tqdm import tqdm
from dataclasses import dataclass, asdict
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger("trainer")


@dataclass
class TrainerConfig:
    """Configuration for training."""
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: Optional[float] = 1.0
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5  # Save checkpoint every N epochs
    early_stopping_patience: int = 10  # Stop if no improvement for N epochs
    early_stopping_min_delta: float = 1e-6  # Minimum change to count as improvement
    device: str = "cpu"
    log_interval: int = 10  # Log every N batches
    validation_interval: int = 1  # Validate every N epochs


class Trainer:
    """
    Generic trainer for PyTorch models.
    
    Handles the complete training loop including:
    - Forward/backward passes
    - Optimization
    - Validation
    - Checkpointing
    - Learning rate scheduling
    - Early stopping
    - Metric tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        config: TrainerConfig,
        scheduler: Optional[_LRScheduler] = None,
        metrics: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer instance
            criterion: Loss function
            config: Training configuration
            scheduler: Optional learning rate scheduler
            metrics: Optional dictionary of metric functions
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.scheduler = scheduler
        self.metrics = metrics or {}
        
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # History tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
        }
        
        # Add metric history
        for metric_name in self.metrics.keys():
            self.history[f'train_{metric_name}'] = []
            self.history[f'val_{metric_name}'] = []
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized - Device: {self.device}, Epochs: {config.epochs}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {name: 0.0 for name in self.metrics.keys()}
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = self._move_to_device(batch)
                
                # Forward pass
                self.optimizer.zero_grad()
                loss, outputs, targets = self._forward_pass(batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                for metric_name, metric_fn in self.metrics.items():
                    metric_value = metric_fn(outputs, targets)
                    epoch_metrics[metric_name] += metric_value
                
                # Update progress bar
                if batch_idx % self.config.log_interval == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.6f}',
                        'lr': f'{self._get_lr():.6f}'
                    })
                
                self.global_step += 1
        
        # Average metrics
        avg_loss = epoch_loss / num_batches
        avg_metrics = {name: value / num_batches for name, value in epoch_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        val_metrics = {name: 0.0 for name in self.metrics.keys()}
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._move_to_device(batch)
                
                # Forward pass
                loss, outputs, targets = self._forward_pass(batch)
                
                # Track metrics
                val_loss += loss.item()
                for metric_name, metric_fn in self.metrics.items():
                    metric_value = metric_fn(outputs, targets)
                    val_metrics[metric_name] += metric_value
        
        # Average metrics
        avg_loss = val_loss / num_batches
        avg_metrics = {name: value / num_batches for name, value in val_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            for metric_name, value in train_metrics.items():
                if metric_name != 'loss':
                    self.history[f'train_{metric_name}'].append(value)
            
            # Validation
            if val_loader is not None and epoch % self.config.validation_interval == 0:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                for metric_name, value in val_metrics.items():
                    if metric_name != 'loss':
                        self.history[f'val_{metric_name}'].append(value)
                
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"Train Loss: {train_metrics['loss']:.6f}, "
                    f"Val Loss: {val_metrics['loss']:.6f}"
                )
                
                # Check for improvement
                if val_metrics['loss'] < self.best_val_loss - self.config.early_stopping_min_delta:
                    self.best_val_loss = val_metrics['loss']
                    self.epochs_without_improvement = 0
                    self.save_checkpoint('best_model.pt', is_best=True)
                    logger.info(f"New best model! Val Loss: {self.best_val_loss:.6f}")
                else:
                    self.epochs_without_improvement += 1
                
                # Early stopping
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"Train Loss: {train_metrics['loss']:.6f}"
                )
            
            # Learning rate scheduling
            if self.scheduler is not None:
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                if isinstance(self.scheduler, ReduceLROnPlateau) and val_loader is not None:
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Track learning rate
            self.history['learning_rates'].append(self._get_lr())
            
            # Periodic checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch + 1}.pt')
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        # Save training history
        self.save_history()
        
        logger.info("Training complete!")
        return self.history
    
    def _forward_pass(self, batch: Any) -> tuple:
        """
        Perform forward pass (to be overridden for custom logic).
        
        Args:
            batch: Input batch
        
        Returns:
            (loss, outputs, targets)
        """
        # Default implementation for dict-based batches
        if isinstance(batch, dict):
            inputs = batch['input']
            targets = batch['target']
        elif isinstance(batch, (tuple, list)):
            inputs, targets = batch
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
        
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        return loss, outputs, targets
    
    def _move_to_device(self, batch: Any) -> Any:
        """Move batch to device."""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        elif isinstance(batch, (tuple, list)):
            return tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x 
                        for x in batch)
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            return batch
    
    def _get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': asdict(self.config)
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save model config if available
        if hasattr(self.model, 'get_config'):
            checkpoint['model_config'] = self.model.get_config()
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def save_history(self, filename: str = 'training_history.json'):
        """Save training history to JSON."""
        filepath = self.checkpoint_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved: {filepath}")


# Example usage
if __name__ == "__main__":
    print("Testing Trainer...\n")
    
    # Create dummy model and data
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        
        def forward(self, x):
            return self.fc(x)
        
        def get_config(self):
            return {'input_dim': 10, 'output_dim': 2}
    
    class DummyDataset(Dataset):
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            return {
                'input': torch.randn(10),
                'target': torch.randint(0, 2, (1,)).float()
            }
    
    # Setup
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    config = TrainerConfig(epochs=5, batch_size=16, checkpoint_dir='test_checkpoints')
    
    trainer = Trainer(model, optimizer, criterion, config)
    
    # Create dataloaders
    train_dataset = DummyDataset()
    val_dataset = DummyDataset()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    print("\nTraining History:")
    print(f"Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"Val Loss: {history['val_loss'][-1]:.6f}")
    
    print("\n✅ Trainer test passed!")
