"""
Tests for TrajectorySequenceBuilder.

Covers: SequenceConfig, sliding window generation, padding modes,
        normalization, and edge cases.
"""

import pytest
import numpy as np
import torch

from src.ml.features.sequence_builder import (
    SequenceConfig,
    TrajectorySequenceBuilder,
)


# ──────────────────────────────────────────────
# SequenceConfig
# ──────────────────────────────────────────────
class TestSequenceConfig:
    def test_default_values(self):
        cfg = SequenceConfig()
        assert cfg.history_length == 20
        assert cfg.prediction_horizon == 30
        assert cfg.stride == 5

    def test_custom_values(self):
        cfg = SequenceConfig(history_length=10, prediction_horizon=5, stride=2)
        assert cfg.history_length == 10
        assert cfg.prediction_horizon == 5
        assert cfg.stride == 2


# ──────────────────────────────────────────────
# Sliding Window
# ──────────────────────────────────────────────
class TestSlidingWindow:
    def test_correct_num_sequences(self):
        cfg = SequenceConfig(history_length=10, prediction_horizon=5, stride=5,
                             normalize=False)
        builder = TrajectorySequenceBuilder(cfg)
        T, D = 100, 8
        features = np.random.randn(T, D)
        result = builder.build_sequences(features)
        # Number of windows: floor((T - hist - pred) / stride) + 1
        expected_n = (T - 10 - 5) // 5 + 1  # = 18
        assert result['history'].shape[0] == expected_n

    def test_history_shape(self):
        cfg = SequenceConfig(history_length=20, prediction_horizon=30, stride=5,
                             normalize=False)
        builder = TrajectorySequenceBuilder(cfg)
        T, D = 200, 12
        features = np.random.randn(T, D)
        result = builder.build_sequences(features)
        N = result['history'].shape[0]
        assert result['history'].shape == (N, 20, D)

    def test_target_shape(self):
        cfg = SequenceConfig(history_length=20, prediction_horizon=30, stride=5,
                             normalize=False)
        builder = TrajectorySequenceBuilder(cfg)
        T, D = 200, 12
        features = np.random.randn(T, D)
        result = builder.build_sequences(features)
        N = result['history'].shape[0]
        assert result['target'].shape == (N, 30, D)

    def test_no_data_leakage(self):
        """History and target should not overlap — target starts after history ends."""
        cfg = SequenceConfig(history_length=10, prediction_horizon=5, stride=10,
                             normalize=False)
        builder = TrajectorySequenceBuilder(cfg)
        T, D = 100, 4
        # Use sequential values so we can verify ranges
        features = np.arange(T * D, dtype=float).reshape(T, D)
        result = builder.build_sequences(features)
        # First window: history=[0..9], target=[10..14]
        hist_last = result['history'][0, -1, 0].item()  # row 9, col 0 → 9*4=36
        tgt_first = result['target'][0, 0, 0].item()    # row 10, col 0 → 10*4=40
        assert tgt_first > hist_last, "Target should come after history (no leakage)"

    def test_stride_controls_step(self):
        cfg1 = SequenceConfig(history_length=10, prediction_horizon=5, stride=1,
                              normalize=False)
        cfg2 = SequenceConfig(history_length=10, prediction_horizon=5, stride=10,
                              normalize=False)
        builder1 = TrajectorySequenceBuilder(cfg1)
        builder2 = TrajectorySequenceBuilder(cfg2)
        features = np.random.randn(100, 4)
        n1 = builder1.build_sequences(features)['history'].shape[0]
        n2 = builder2.build_sequences(features)['history'].shape[0]
        assert n1 > n2, "Smaller stride should produce more sequences"


# ──────────────────────────────────────────────
# Padding
# ──────────────────────────────────────────────
class TestPadding:
    def test_zero_padding_short_sequence(self):
        """Short sequence padded with zeros."""
        cfg = SequenceConfig(history_length=20, prediction_horizon=30, stride=5,
                             padding='zero', normalize=False)
        builder = TrajectorySequenceBuilder(cfg)
        features = np.ones((5, 4))  # Only 5 timesteps, need 50
        result = builder.build_sequences(features)
        assert result['history'].shape == (1, 20, 4)
        # Padded region should be zeros
        assert result['history'][0, 5:, :].sum().item() == 0.0

    def test_edge_padding_short_sequence(self):
        """Short sequence padded with edge (last) value."""
        cfg = SequenceConfig(history_length=20, prediction_horizon=30, stride=5,
                             padding='edge', normalize=False)
        builder = TrajectorySequenceBuilder(cfg)
        features = np.ones((5, 4)) * 3.0
        result = builder.build_sequences(features)
        assert result['history'].shape == (1, 20, 4)
        # Edge padding repeats last value
        assert result['history'][0, 10, 0].item() == 3.0

    def test_none_padding_returns_empty(self):
        """Padding='none' with insufficient data returns empty tensors."""
        cfg = SequenceConfig(history_length=20, prediction_horizon=30, stride=5,
                             padding='none', normalize=False)
        builder = TrajectorySequenceBuilder(cfg)
        features = np.ones((5, 4))
        result = builder.build_sequences(features)
        assert result['history'].shape[0] == 0

    def test_mask_marks_valid_timesteps(self):
        """Mask should be 1 for valid timesteps, 0 for padded."""
        cfg = SequenceConfig(history_length=20, prediction_horizon=30, stride=5,
                             padding='zero', normalize=False)
        builder = TrajectorySequenceBuilder(cfg)
        features = np.ones((8, 4))
        result = builder.build_sequences(features)
        mask = result['mask'][0]
        # First 8 should be valid (1), rest padded (0)
        assert mask[:8].sum().item() == 8.0
        assert mask[8:].sum().item() == 0.0


# ──────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────
class TestNormalization:
    def test_standard_scaler_fit_transform(self):
        cfg = SequenceConfig(history_length=10, prediction_horizon=5, stride=5,
                             normalize=True, normalization_method='standard')
        builder = TrajectorySequenceBuilder(cfg)
        features = np.random.randn(100, 4) * 10 + 50
        result = builder.build_sequences(features)
        assert builder.scaler is not None
        assert result['history'].shape[1] == 10

    def test_subsequent_transform_uses_fitted_scaler(self):
        cfg = SequenceConfig(history_length=10, prediction_horizon=5, stride=5,
                             normalize=True, normalization_method='standard')
        builder = TrajectorySequenceBuilder(cfg)
        features1 = np.random.randn(100, 4) * 10 + 50
        builder.build_sequences(features1)
        # Second call should use existing scaler (transform, not fit_transform)
        features2 = np.random.randn(100, 4) * 10 + 50
        result2 = builder.build_sequences(features2)
        assert result2['history'].shape[1] == 10

    def test_minmax_scaler(self):
        cfg = SequenceConfig(history_length=10, prediction_horizon=5, stride=5,
                             normalize=True, normalization_method='minmax')
        builder = TrajectorySequenceBuilder(cfg)
        features = np.random.randn(100, 4) * 10 + 50
        result = builder.build_sequences(features)
        assert builder.scaler is not None
        assert result['history'].shape[1] == 10
