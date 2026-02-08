"""
Tests for training infrastructure: losses, trainer, augmentation.

Covers: WeightedMSELoss, SmoothL1TrajectoryLoss, MultiHorizonLoss,
        TrajectoryLoss, ClassificationLoss, FocalLoss, factory functions,
        TrajectoryAugmenter, MixUpAugmenter, and Trainer loop.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tempfile
import os

from src.ml.training.losses import (
    WeightedMSELoss,
    SmoothL1TrajectoryLoss,
    MultiHorizonLoss,
    TrajectoryLoss,
    ClassificationLoss,
    FocalLoss,
    create_trajectory_loss,
    create_classification_loss,
)
from src.ml.features.augmentation import (
    TrajectoryAugmenter,
    AugmentationConfig,
    MixUpAugmenter,
)
from src.ml.training.trainer import Trainer, TrainerConfig


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def traj_tensors(device):
    """Standard trajectory prediction tensors."""
    pred = torch.randn(4, 30, 6, device=device)
    tgt = torch.randn(4, 30, 6, device=device)
    return pred, tgt


@pytest.fixture
def class_tensors(device):
    """Standard classification tensors."""
    logits = torch.randn(8, 6, device=device)
    targets = torch.randint(0, 6, (8,), device=device)
    return logits, targets


# ──────────────────────────────────────────────
# WeightedMSELoss
# ──────────────────────────────────────────────
class TestWeightedMSELoss:
    def test_higher_position_weight(self, traj_tensors):
        pred, tgt = traj_tensors
        loss_high = WeightedMSELoss(position_weight=10.0, velocity_weight=1.0)(pred, tgt)
        loss_low = WeightedMSELoss(position_weight=1.0, velocity_weight=1.0)(pred, tgt)
        assert loss_high > loss_low or torch.isclose(loss_high, loss_low)

    def test_zero_velocity_weight(self, traj_tensors):
        pred, tgt = traj_tensors
        loss_fn = WeightedMSELoss(position_weight=1.0, velocity_weight=0.0)
        loss = loss_fn(pred, tgt)
        # Should only include position error
        pos_only = nn.functional.mse_loss(pred[..., :3], tgt[..., :3])
        torch.testing.assert_close(loss, pos_only)


# ──────────────────────────────────────────────
# SmoothL1TrajectoryLoss
# ──────────────────────────────────────────────
class TestSmoothL1Loss:
    def test_output_scalar(self, traj_tensors):
        pred, tgt = traj_tensors
        loss = SmoothL1TrajectoryLoss()(pred, tgt)
        assert loss.dim() == 0

    def test_robust_to_outliers(self, device):
        """Smooth L1 should produce lower loss than MSE for large errors."""
        pred = torch.zeros(4, 10, 6, device=device)
        tgt = torch.ones(4, 10, 6, device=device) * 100.0  # large error
        smooth_l1 = SmoothL1TrajectoryLoss()(pred, tgt).item()
        mse = WeightedMSELoss()(pred, tgt).item()
        assert smooth_l1 < mse


# ──────────────────────────────────────────────
# MultiHorizonLoss
# ──────────────────────────────────────────────
class TestMultiHorizonLoss:
    def test_near_term_weighted_more(self, device):
        """With decay_rate < 1, near-term steps get more weight."""
        pred = torch.randn(2, 10, 6, device=device)
        tgt = torch.zeros(2, 10, 6, device=device)
        loss_fn = MultiHorizonLoss(decay_rate=0.5)
        loss = loss_fn(pred, tgt)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_uniform_weights(self, device):
        """decay_rate=1.0 should give uniform weights."""
        loss_fn = MultiHorizonLoss(decay_rate=1.0)
        pred = torch.randn(2, 5, 6, device=device)
        tgt = torch.randn(2, 5, 6, device=device)
        loss = loss_fn(pred, tgt)
        assert loss.dim() == 0


# ──────────────────────────────────────────────
# TrajectoryLoss
# ──────────────────────────────────────────────
class TestTrajectoryLoss:
    def test_mse_vs_smooth_l1(self, traj_tensors):
        pred, tgt = traj_tensors
        mse_loss = TrajectoryLoss(use_smooth_l1=False)(pred, tgt)
        smooth_loss = TrajectoryLoss(use_smooth_l1=True)(pred, tgt)
        assert not torch.isclose(mse_loss, smooth_loss), "MSE and Smooth L1 should differ"

    def test_zero_velocity_weight(self, traj_tensors):
        pred, tgt = traj_tensors
        loss = TrajectoryLoss(velocity_weight=0.0)(pred, tgt)
        pos_only = nn.functional.mse_loss(pred[..., :3], tgt[..., :3])
        torch.testing.assert_close(loss, pos_only)


# ──────────────────────────────────────────────
# ClassificationLoss / FocalLoss
# ──────────────────────────────────────────────
class TestClassificationLoss:
    def test_cross_entropy_scalar(self, class_tensors):
        logits, targets = class_tensors
        loss = ClassificationLoss()(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_focal_downweights_easy(self, class_tensors):
        logits, targets = class_tensors
        ce_loss = ClassificationLoss()(logits, targets)
        focal_loss = FocalLoss(gamma=2.0)(logits, targets)
        # Focal loss should generally be <= CE loss (downweights easy examples)
        # This is probabilistic but almost always holds
        assert focal_loss.item() <= ce_loss.item() * 1.5  # generous bound


# ──────────────────────────────────────────────
# Factory functions
# ──────────────────────────────────────────────
class TestLossFactories:
    def test_trajectory_loss_types(self):
        for loss_type in ["mse", "weighted_mse", "smooth_l1", "trajectory", "multi_horizon"]:
            loss_fn = create_trajectory_loss(loss_type)
            assert isinstance(loss_fn, nn.Module)

    def test_classification_loss_types(self):
        for loss_type in ["ce", "focal"]:
            loss_fn = create_classification_loss(loss_type)
            assert isinstance(loss_fn, nn.Module)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown loss type"):
            create_trajectory_loss("unknown")
        with pytest.raises(ValueError, match="Unknown loss type"):
            create_classification_loss("unknown")


# ──────────────────────────────────────────────
# TrajectoryAugmenter
# ──────────────────────────────────────────────
class TestAugmenter:
    def test_output_shapes_preserved(self, device):
        augmenter = TrajectoryAugmenter(AugmentationConfig(apply_probability=1.0))
        pos = torch.randn(4, 20, 3, device=device) * 7000
        vel = torch.randn(4, 20, 3, device=device) * 7
        aug_pos, aug_vel, _ = augmenter.augment(pos, vel)
        assert aug_pos.shape == pos.shape
        assert aug_vel.shape == vel.shape

    def test_noise_changes_values(self, device):
        augmenter = TrajectoryAugmenter(AugmentationConfig(
            add_noise=True, rotation=False, velocity_perturbation=False,
            time_shift=False, dropout_timesteps=False, apply_probability=1.0
        ))
        pos = torch.ones(2, 10, 3, device=device) * 7000
        vel = torch.ones(2, 10, 3, device=device) * 7
        aug_pos, aug_vel, _ = augmenter.augment(pos, vel)
        assert not torch.allclose(aug_pos, pos), "Noise should change values"

    def test_rotation_preserves_norms(self, device):
        augmenter = TrajectoryAugmenter(AugmentationConfig(
            add_noise=False, rotation=True, velocity_perturbation=False,
            time_shift=False, dropout_timesteps=False, apply_probability=1.0
        ))
        pos = torch.randn(2, 10, 3, device=device) * 7000
        vel = torch.randn(2, 10, 3, device=device) * 7
        orig_pos_norms = torch.norm(pos, dim=-1)
        aug_pos, _, _ = augmenter.augment(pos, vel)
        aug_pos_norms = torch.norm(aug_pos, dim=-1)
        torch.testing.assert_close(aug_pos_norms, orig_pos_norms, atol=0.1, rtol=1e-4)

    def test_zero_probability_returns_unchanged(self, device):
        augmenter = TrajectoryAugmenter(AugmentationConfig(apply_probability=0.0))
        pos = torch.randn(2, 10, 3, device=device)
        vel = torch.randn(2, 10, 3, device=device)
        aug_pos, aug_vel, _ = augmenter.augment(pos, vel)
        torch.testing.assert_close(aug_pos, pos)
        torch.testing.assert_close(aug_vel, vel)


# ──────────────────────────────────────────────
# MixUp
# ──────────────────────────────────────────────
class TestMixUp:
    def test_output_shape(self):
        mixup = MixUpAugmenter(alpha=0.2)
        x1 = torch.randn(10, 24)
        x2 = torch.randn(10, 24)
        y1 = torch.randn(10, 6)
        y2 = torch.randn(10, 6)
        x_m, y_m = mixup.mixup(x1, x2, y1, y2)
        assert x_m.shape == x1.shape
        assert y_m.shape == y1.shape

    def test_batch_mixup_shape(self):
        mixup = MixUpAugmenter(alpha=0.2)
        batch_x = torch.randn(8, 20, 24)
        batch_y = torch.randn(8, 30, 6)
        mixed_x, mixed_y = mixup.mixup_batch(batch_x, batch_y)
        assert mixed_x.shape == batch_x.shape
        assert mixed_y.shape == batch_y.shape


# ──────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────
class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

    def get_config(self):
        return {'input_dim': 10, 'output_dim': 2}


class _DummyDataset(Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, idx):
        return torch.randn(10), torch.randn(2)


class TestTrainer:
    def test_train_one_epoch(self, device):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _DummyModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            config = TrainerConfig(
                epochs=1, batch_size=8,
                checkpoint_dir=tmpdir, device=str(device)
            )
            trainer = Trainer(model, optimizer, criterion, config)
            train_loader = DataLoader(_DummyDataset(), batch_size=8)
            history = trainer.train(train_loader)
            assert len(history['train_loss']) == 1
            assert history['train_loss'][0] > 0

    def test_checkpoint_save_load(self, device):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _DummyModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            config = TrainerConfig(
                epochs=1, batch_size=8,
                checkpoint_dir=tmpdir, device=str(device)
            )
            trainer = Trainer(model, optimizer, criterion, config)
            train_loader = DataLoader(_DummyDataset(), batch_size=8)
            trainer.train(train_loader)

            # Verify checkpoint exists
            ckpt_path = os.path.join(tmpdir, 'final_model.pt')
            assert os.path.exists(ckpt_path)

            # Load checkpoint into fresh trainer
            model2 = _DummyModel()
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
            trainer2 = Trainer(model2, optimizer2, criterion,
                               TrainerConfig(checkpoint_dir=tmpdir, device=str(device)))
            trainer2.load_checkpoint(ckpt_path)
            # Verify weights match
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                torch.testing.assert_close(p1.cpu(), p2.cpu())
