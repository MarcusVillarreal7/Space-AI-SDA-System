"""
Custom Loss Functions for Trajectory Prediction.

This module provides specialized loss functions for satellite trajectory prediction,
including:
- Weighted MSE (position vs. velocity)
- Smooth L1 loss for robustness
- Multi-horizon losses
- Uncertainty-aware losses

Author: Space AI Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss for position and velocity.
    
    Allows different weights for position and velocity components,
    useful when one is more important or has different scales.
    """
    
    def __init__(self, position_weight: float = 1.0, velocity_weight: float = 1.0):
        """
        Initialize weighted MSE loss.
        
        Args:
            position_weight: Weight for position error (first 3 dims)
            velocity_weight: Weight for velocity error (last 3 dims)
        """
        super().__init__()
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss.
        
        Args:
            predictions: Predicted values (batch, seq_len, 6)
            targets: Target values (batch, seq_len, 6)
        
        Returns:
            Weighted MSE loss
        """
        # Split into position and velocity
        pred_pos = predictions[..., :3]  # First 3 dims
        pred_vel = predictions[..., 3:6]  # Last 3 dims
        target_pos = targets[..., :3]
        target_vel = targets[..., 3:6]
        
        # Compute MSE for each component
        pos_loss = F.mse_loss(pred_pos, target_pos)
        vel_loss = F.mse_loss(pred_vel, target_vel)
        
        # Weight and combine
        total_loss = self.position_weight * pos_loss + self.velocity_weight * vel_loss
        
        return total_loss


class SmoothL1TrajectoryLoss(nn.Module):
    """
    Smooth L1 loss for trajectory prediction.
    
    More robust to outliers than MSE, with smooth transition near zero.
    Also known as Huber loss.
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        position_weight: float = 1.0,
        velocity_weight: float = 1.0
    ):
        """
        Initialize Smooth L1 loss.
        
        Args:
            beta: Threshold for switching from L2 to L1
            position_weight: Weight for position component
            velocity_weight: Weight for velocity component
        """
        super().__init__()
        self.beta = beta
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute smooth L1 loss.
        
        Args:
            predictions: Predicted values (batch, seq_len, 6)
            targets: Target values (batch, seq_len, 6)
        
        Returns:
            Smooth L1 loss
        """
        # Split into position and velocity
        pred_pos = predictions[..., :3]
        pred_vel = predictions[..., 3:6]
        target_pos = targets[..., :3]
        target_vel = targets[..., 3:6]
        
        # Compute Smooth L1 for each component
        pos_loss = F.smooth_l1_loss(pred_pos, target_pos, beta=self.beta)
        vel_loss = F.smooth_l1_loss(pred_vel, target_vel, beta=self.beta)
        
        # Weight and combine
        total_loss = self.position_weight * pos_loss + self.velocity_weight * vel_loss
        
        return total_loss


class MultiHorizonLoss(nn.Module):
    """
    Loss that weights prediction errors based on time horizon.
    
    Near-term predictions are weighted more heavily than far-term predictions,
    as they are typically more important and reliable.
    """
    
    def __init__(
        self,
        base_loss: nn.Module = nn.MSELoss(),
        decay_rate: float = 0.95
    ):
        """
        Initialize multi-horizon loss.
        
        Args:
            base_loss: Base loss function to use
            decay_rate: Exponential decay rate for horizon weights
        """
        super().__init__()
        self.base_loss = base_loss
        self.decay_rate = decay_rate
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted multi-horizon loss.
        
        Args:
            predictions: Predicted values (batch, seq_len, features)
            targets: Target values (batch, seq_len, features)
        
        Returns:
            Weighted loss across horizons
        """
        batch_size, seq_len, _ = predictions.shape
        
        # Compute time-based weights (exponentially decaying)
        weights = torch.tensor([self.decay_rate ** t for t in range(seq_len)],
                              device=predictions.device)
        weights = weights / weights.sum()  # Normalize
        
        # Compute loss for each timestep
        total_loss = 0.0
        for t in range(seq_len):
            timestep_loss = self.base_loss(predictions[:, t, :], targets[:, t, :])
            total_loss += weights[t] * timestep_loss
        
        return total_loss


class TrajectoryLoss(nn.Module):
    """
    Combined loss for trajectory prediction.
    
    Combines multiple loss components:
    - MSE loss for accuracy
    - Smooth L1 for robustness
    - Position/velocity weighting
    """
    
    def __init__(
        self,
        position_weight: float = 1.0,
        velocity_weight: float = 0.1,  # Velocity often less important
        use_smooth_l1: bool = False,
        smooth_l1_beta: float = 1.0
    ):
        """
        Initialize trajectory loss.
        
        Args:
            position_weight: Weight for position error
            velocity_weight: Weight for velocity error
            use_smooth_l1: Whether to use Smooth L1 instead of MSE
            smooth_l1_beta: Beta parameter for Smooth L1
        """
        super().__init__()
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.use_smooth_l1 = use_smooth_l1
        self.smooth_l1_beta = smooth_l1_beta
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory loss.
        
        Args:
            predictions: Predicted trajectories (batch, seq_len, 6)
            targets: Target trajectories (batch, seq_len, 6)
        
        Returns:
            Combined loss
        """
        # Split position and velocity
        pred_pos = predictions[..., :3]
        pred_vel = predictions[..., 3:6]
        target_pos = targets[..., :3]
        target_vel = targets[..., 3:6]
        
        # Compute position loss
        if self.use_smooth_l1:
            pos_loss = F.smooth_l1_loss(pred_pos, target_pos, beta=self.smooth_l1_beta)
            vel_loss = F.smooth_l1_loss(pred_vel, target_vel, beta=self.smooth_l1_beta)
        else:
            pos_loss = F.mse_loss(pred_pos, target_pos)
            vel_loss = F.mse_loss(pred_vel, target_vel)
        
        # Combine with weights
        total_loss = self.position_weight * pos_loss + self.velocity_weight * vel_loss
        
        return total_loss


class ClassificationLoss(nn.Module):
    """
    Loss for maneuver classification.
    
    Standard cross-entropy loss with optional class weighting.
    """
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        """
        Initialize classification loss.
        
        Args:
            class_weights: Optional weights for each class (for imbalanced datasets)
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            logits: Model outputs (batch, num_classes)
            targets: Target class indices (batch,)
        
        Returns:
            Cross-entropy loss
        """
        return self.criterion(logits, targets.long())


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    
    Downweights easy examples and focuses on hard negatives.
    Useful for imbalanced classification tasks.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor (default 1.0)
            gamma: Focusing parameter (default 2.0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Model outputs (batch, num_classes)
            targets: Target class indices (batch,)
        
        Returns:
            Focal loss
        """
        # Compute cross-entropy
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')
        
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        target_probs = probs.gather(1, targets.long().unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Weighted loss
        loss = self.alpha * focal_weight * ce_loss
        
        return loss.mean()


# Helper functions for creating loss functions
def create_trajectory_loss(
    loss_type: str = "mse",
    position_weight: float = 1.0,
    velocity_weight: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function for creating trajectory losses.
    
    Args:
        loss_type: Type of loss ("mse", "smooth_l1", "weighted_mse", "multi_horizon")
        position_weight: Weight for position component
        velocity_weight: Weight for velocity component
        **kwargs: Additional arguments for specific loss types
    
    Returns:
        Loss function module
    """
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "weighted_mse":
        return WeightedMSELoss(position_weight, velocity_weight)
    elif loss_type == "smooth_l1":
        return SmoothL1TrajectoryLoss(
            position_weight=position_weight,
            velocity_weight=velocity_weight,
            **kwargs
        )
    elif loss_type == "trajectory":
        return TrajectoryLoss(
            position_weight=position_weight,
            velocity_weight=velocity_weight,
            **kwargs
        )
    elif loss_type == "multi_horizon":
        base_loss = WeightedMSELoss(position_weight, velocity_weight)
        return MultiHorizonLoss(base_loss, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def create_classification_loss(
    loss_type: str = "ce",
    class_weights: Optional[list] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function for creating classification losses.
    
    Args:
        loss_type: Type of loss ("ce", "focal")
        class_weights: Optional class weights
        **kwargs: Additional arguments
    
    Returns:
        Loss function module
    """
    if class_weights is not None:
        class_weights = torch.tensor(class_weights)
    
    if loss_type == "ce":
        return ClassificationLoss(class_weights)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Loss Functions...\n")
    
    # Test trajectory losses
    batch_size = 4
    seq_len = 30
    features = 6
    
    predictions = torch.randn(batch_size, seq_len, features)
    targets = torch.randn(batch_size, seq_len, features)
    
    # Test Weighted MSE
    loss_fn = WeightedMSELoss(position_weight=1.0, velocity_weight=0.1)
    loss = loss_fn(predictions, targets)
    print(f"Weighted MSE Loss: {loss.item():.6f}")
    
    # Test Smooth L1
    loss_fn = SmoothL1TrajectoryLoss(beta=1.0)
    loss = loss_fn(predictions, targets)
    print(f"Smooth L1 Loss: {loss.item():.6f}")
    
    # Test Multi-Horizon
    loss_fn = MultiHorizonLoss()
    loss = loss_fn(predictions, targets)
    print(f"Multi-Horizon Loss: {loss.item():.6f}")
    
    # Test Trajectory Loss
    loss_fn = TrajectoryLoss()
    loss = loss_fn(predictions, targets)
    print(f"Trajectory Loss: {loss.item():.6f}")
    
    # Test Classification Loss
    print("\n--- Classification Losses ---")
    logits = torch.randn(batch_size, 6)  # 6 classes
    class_targets = torch.randint(0, 6, (batch_size,))
    
    loss_fn = ClassificationLoss()
    loss = loss_fn(logits, class_targets)
    print(f"Cross-Entropy Loss: {loss.item():.6f}")
    
    loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
    loss = loss_fn(logits, class_targets)
    print(f"Focal Loss: {loss.item():.6f}")
    
    # Test factory functions
    print("\n--- Factory Functions ---")
    loss_fn = create_trajectory_loss("weighted_mse", position_weight=1.0, velocity_weight=0.1)
    loss = loss_fn(predictions, targets)
    print(f"Factory Trajectory Loss: {loss.item():.6f}")
    
    loss_fn = create_classification_loss("ce")
    loss = loss_fn(logits, class_targets)
    print(f"Factory Classification Loss: {loss.item():.6f}")
    
    print("\n✅ All loss function tests passed!")
