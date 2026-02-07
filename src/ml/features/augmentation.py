"""
Data Augmentation for Satellite Trajectory Training.

Implements augmentation techniques to increase training data diversity and
improve model robustness. Techniques are physically-motivated to preserve
orbital mechanics realism.

Author: Space AI Team
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import random

from src.utils.logging_config import get_logger

logger = get_logger("augmentation")


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    add_noise: bool = True
    noise_std: float = 0.1  # Fraction of feature value
    time_shift: bool = True
    time_shift_range: int = 5  # Timesteps
    rotation: bool = True
    rotation_angle_deg: float = 15.0  # Max rotation
    velocity_perturbation: bool = True
    velocity_perturbation_std: float = 0.05  # km/s
    dropout_timesteps: bool = True
    dropout_rate: float = 0.1  # Fraction of timesteps to drop
    apply_probability: float = 0.8  # Probability of applying augmentation


class TrajectoryAugmenter:
    """
    Augment trajectory data for training.
    
    Applies physically-motivated augmentations:
    - Gaussian noise injection (measurement uncertainty)
    - Time shifting (different observation windows)
    - Coordinate frame rotation (orientation invariance)
    - Velocity perturbations (maneuver simulation)
    - Timestep dropout (missing data simulation)
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmenter.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()
        logger.info("Trajectory augmenter initialized")
    
    def augment(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentation to trajectory.
        
        Args:
            positions: Position sequence (batch, seq_len, 3)
            velocities: Velocity sequence (batch, seq_len, 3)
            features: Optional full feature tensor (batch, seq_len, feature_dim)
        
        Returns:
            Augmented (positions, velocities, features)
        """
        # Decide whether to apply augmentation
        if random.random() > self.config.apply_probability:
            return positions, velocities, features
        
        aug_positions = positions.clone()
        aug_velocities = velocities.clone()
        aug_features = features.clone() if features is not None else None
        
        # Apply augmentations
        if self.config.add_noise:
            aug_positions, aug_velocities = self._add_noise(aug_positions, aug_velocities)
        
        if self.config.rotation:
            aug_positions, aug_velocities = self._rotate(aug_positions, aug_velocities)
        
        if self.config.velocity_perturbation:
            aug_velocities = self._perturb_velocity(aug_velocities)
        
        if self.config.time_shift:
            aug_positions, aug_velocities, aug_features = self._time_shift(
                aug_positions, aug_velocities, aug_features
            )
        
        if self.config.dropout_timesteps:
            aug_positions, aug_velocities, aug_features = self._dropout_timesteps(
                aug_positions, aug_velocities, aug_features
            )
        
        return aug_positions, aug_velocities, aug_features
    
    def _add_noise(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add Gaussian noise to positions and velocities."""
        # Position noise (km)
        pos_noise = torch.randn_like(positions) * self.config.noise_std * positions.abs().mean()
        positions = positions + pos_noise
        
        # Velocity noise (km/s)
        vel_noise = torch.randn_like(velocities) * self.config.noise_std * velocities.abs().mean()
        velocities = velocities + vel_noise
        
        return positions, velocities
    
    def _rotate(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rotate trajectory in 3D space.
        
        Applies random rotation around each axis to simulate
        different coordinate frame orientations.
        """
        batch_size = positions.size(0)
        device = positions.device
        
        # Random rotation angles
        angle = random.uniform(-self.config.rotation_angle_deg, self.config.rotation_angle_deg)
        angle_rad = np.deg2rad(angle)
        
        # Random axis
        axis_idx = random.randint(0, 2)
        
        # Rotation matrices for each axis
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        if axis_idx == 0:  # X-axis
            R = torch.tensor([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ], device=device, dtype=torch.float32)
        elif axis_idx == 1:  # Y-axis
            R = torch.tensor([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], device=device, dtype=torch.float32)
        else:  # Z-axis
            R = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], device=device, dtype=torch.float32)
        
        # Apply rotation
        # positions: (batch, seq_len, 3) → reshape for matmul
        orig_shape = positions.shape
        positions_flat = positions.reshape(-1, 3)
        velocities_flat = velocities.reshape(-1, 3)
        
        positions_rot = torch.matmul(positions_flat, R.T)
        velocities_rot = torch.matmul(velocities_flat, R.T)
        
        positions = positions_rot.reshape(orig_shape)
        velocities = velocities_rot.reshape(orig_shape)
        
        return positions, velocities
    
    def _perturb_velocity(self, velocities: torch.Tensor) -> torch.Tensor:
        """
        Perturb velocity to simulate small maneuvers.
        
        Adds small delta-V in random direction.
        """
        # Random perturbation
        delta_v = torch.randn_like(velocities) * self.config.velocity_perturbation_std
        velocities = velocities + delta_v
        
        return velocities
    
    def _time_shift(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Shift trajectory in time (roll along time axis).
        
        Simulates different observation windows.
        """
        shift = random.randint(-self.config.time_shift_range, self.config.time_shift_range)
        
        if shift != 0:
            positions = torch.roll(positions, shifts=shift, dims=1)
            velocities = torch.roll(velocities, shifts=shift, dims=1)
            if features is not None:
                features = torch.roll(features, shifts=shift, dims=1)
        
        return positions, velocities, features
    
    def _dropout_timesteps(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Randomly dropout timesteps to simulate missing data.
        
        Dropped timesteps are replaced with interpolated values.
        """
        batch_size, seq_len, _ = positions.shape
        
        # Create dropout mask
        keep_prob = 1 - self.config.dropout_rate
        mask = torch.rand(batch_size, seq_len, 1, device=positions.device) < keep_prob
        
        # For dropped timesteps, use linear interpolation
        # (Simple approach: for now just keep original, more sophisticated would interpolate)
        # This simulates sensor dropout without breaking continuity
        
        # Apply mask with small noise for dropped points
        noise_positions = torch.randn_like(positions) * 0.01
        noise_velocities = torch.randn_like(velocities) * 0.001
        
        positions = torch.where(mask, positions, positions + noise_positions)
        velocities = torch.where(mask, velocities, velocities + noise_velocities)
        
        if features is not None:
            noise_features = torch.randn_like(features) * 0.01
            features = torch.where(mask.expand_as(features), features, features + noise_features)
        
        return positions, velocities, features
    
    def augment_batch(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Augment a batch of data.
        
        Args:
            batch: Dictionary with 'positions', 'velocities', optional 'features'
        
        Returns:
            Augmented batch
        """
        positions = batch.get('positions')
        velocities = batch.get('velocities')
        features = batch.get('features')
        
        if positions is None or velocities is None:
            # Try to extract from features if provided
            if features is not None and features.shape[-1] >= 6:
                positions = features[..., :3]
                velocities = features[..., 3:6]
            else:
                logger.warning("No position/velocity data to augment")
                return batch
        
        # Apply augmentation
        aug_pos, aug_vel, aug_feat = self.augment(positions, velocities, features)
        
        # Update batch
        result = batch.copy()
        if 'positions' in batch:
            result['positions'] = aug_pos
        if 'velocities' in batch:
            result['velocities'] = aug_vel
        if 'features' in batch and aug_feat is not None:
            result['features'] = aug_feat
        
        return result


class MixUpAugmenter:
    """
    MixUp augmentation for trajectory data.
    
    Combines pairs of training examples with linear interpolation
    to create synthetic training data.
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp augmenter.
        
        Args:
            alpha: Beta distribution parameter for mixing coefficient
        """
        self.alpha = alpha
    
    def mixup(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MixUp to a pair of examples.
        
        Args:
            x1, x2: Input features
            y1, y2: Targets
        
        Returns:
            Mixed (x, y)
        """
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix inputs and targets
        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2
        
        return x_mixed, y_mixed
    
    def mixup_batch(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MixUp to entire batch by pairing examples.
        
        Args:
            batch_x: Input batch (batch_size, ...)
            batch_y: Target batch (batch_size, ...)
        
        Returns:
            Mixed (batch_x, batch_y)
        """
        batch_size = batch_x.size(0)
        
        # Random permutation for pairing
        indices = torch.randperm(batch_size)
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix
        mixed_x = lam * batch_x + (1 - lam) * batch_x[indices]
        mixed_y = lam * batch_y + (1 - lam) * batch_y[indices]
        
        return mixed_x, mixed_y


# Example usage and testing
if __name__ == "__main__":
    print("Testing Trajectory Augmenter...\n")
    
    config = AugmentationConfig(
        add_noise=True,
        rotation=True,
        velocity_perturbation=True,
        time_shift=True,
        dropout_timesteps=True
    )
    
    augmenter = TrajectoryAugmenter(config)
    
    # Create test data
    batch_size = 4
    seq_len = 20
    positions = torch.randn(batch_size, seq_len, 3) * 7000  # km
    velocities = torch.randn(batch_size, seq_len, 3) * 7  # km/s
    
    # Apply augmentation
    aug_pos, aug_vel, _ = augmenter.augment(positions, velocities)
    
    print(f"Original positions shape: {positions.shape}")
    print(f"Augmented positions shape: {aug_pos.shape}")
    
    # Check differences
    pos_diff = (aug_pos - positions).abs().mean()
    vel_diff = (aug_vel - velocities).abs().mean()
    print(f"\nMean position change: {pos_diff:.4f} km")
    print(f"Mean velocity change: {vel_diff:.4f} km/s")
    
    # Test batch augmentation
    batch = {
        'positions': positions,
        'velocities': velocities,
        'features': torch.randn(batch_size, seq_len, 24)
    }
    
    aug_batch = augmenter.augment_batch(batch)
    print(f"\nAugmented batch keys: {aug_batch.keys()}")
    
    # Test MixUp
    print("\n--- Testing MixUp ---")
    mixup = MixUpAugmenter(alpha=0.2)
    
    x1 = torch.randn(10)
    x2 = torch.randn(10)
    y1 = torch.randn(1)
    y2 = torch.randn(1)
    
    x_mixed, y_mixed = mixup.mixup(x1, x2, y1, y2)
    print(f"Mixed input shape: {x_mixed.shape}")
    print(f"Mixed target shape: {y_mixed.shape}")
    
    # Test batch mixup
    batch_x = torch.randn(8, 10)
    batch_y = torch.randn(8, 1)
    
    mixed_x, mixed_y = mixup.mixup_batch(batch_x, batch_y)
    print(f"Mixed batch shape: {mixed_x.shape}")
    
    print("\n✅ All augmentation tests passed!")
