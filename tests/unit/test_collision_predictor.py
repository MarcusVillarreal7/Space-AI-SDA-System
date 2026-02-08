"""
Tests for CollisionPredictor model.

Covers: RelativeTrajectoryEncoder, CollisionPredictor forward/predict,
        batch collision matrix, config/factory, and risk levels.
"""

import pytest
import torch
import numpy as np

from src.ml.models.collision_predictor import (
    RelativeTrajectoryEncoder,
    CollisionPredictor,
    CollisionPredictorConfig,
    get_risk_level_name,
    RISK_LEVELS,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def default_config():
    return CollisionPredictorConfig()


# ──────────────────────────────────────────────
# RelativeTrajectoryEncoder
# ──────────────────────────────────────────────
class TestRelativeTrajectoryEncoder:
    def test_output_shape(self, device):
        encoder = RelativeTrajectoryEncoder(feature_dim=24).to(device)
        obj1 = torch.randn(4, 24, device=device)
        obj2 = torch.randn(4, 24, device=device)
        out = encoder(obj1, obj2)
        # 24 + 24 + 10 (rel_pos3 + rel_vel3 + dist1 + speed1 + closing1 + ttca1)
        assert out.shape == (4, 58)

    def test_relative_position_velocity(self, device):
        encoder = RelativeTrajectoryEncoder().to(device)
        obj1 = torch.zeros(1, 24, device=device)
        obj1[0, :3] = torch.tensor([100.0, 0.0, 0.0])  # pos1
        obj1[0, 3:6] = torch.tensor([0.0, 1.0, 0.0])   # vel1
        obj2 = torch.zeros(1, 24, device=device)
        obj2[0, :3] = torch.tensor([200.0, 0.0, 0.0])  # pos2
        obj2[0, 3:6] = torch.tensor([0.0, -1.0, 0.0])  # vel2
        out = encoder(obj1, obj2)
        # rel_pos = [100, 0, 0], starts at index 48
        assert out.shape == (1, 58)
        # rel_pos is in the combined output after obj1(24) + obj2(24) = index 48
        rel_pos = out[0, 48:51]
        torch.testing.assert_close(rel_pos, torch.tensor([100.0, 0.0, 0.0], device=device))

    def test_closing_rate_sign(self, device):
        """Negative closing rate means approaching."""
        encoder = RelativeTrajectoryEncoder().to(device)
        # Objects approaching each other along x-axis
        obj1 = torch.zeros(1, 24, device=device)
        obj1[0, :3] = torch.tensor([0.0, 0.0, 0.0])
        obj1[0, 3:6] = torch.tensor([1.0, 0.0, 0.0])   # moving right
        obj2 = torch.zeros(1, 24, device=device)
        obj2[0, :3] = torch.tensor([100.0, 0.0, 0.0])
        obj2[0, 3:6] = torch.tensor([-1.0, 0.0, 0.0])  # moving left
        out = encoder(obj1, obj2)
        # closing_rate index: 48+3+3+1+1 = 56
        closing_rate = out[0, 56].item()
        assert closing_rate < 0, "Approaching objects should have negative closing rate"

    def test_ttca_clamped(self, device):
        encoder = RelativeTrajectoryEncoder().to(device)
        obj1 = torch.randn(4, 24, device=device)
        obj2 = torch.randn(4, 24, device=device)
        out = encoder(obj1, obj2)
        # TTCA at index 57
        ttca = out[:, 57]
        assert (ttca >= 0).all(), "TTCA should be >= 0"
        assert (ttca <= 86400).all(), "TTCA should be <= 86400"


# ──────────────────────────────────────────────
# CollisionPredictor
# ──────────────────────────────────────────────
class TestCollisionPredictor:
    def test_forward_shape(self, device, default_config):
        model = CollisionPredictor(default_config).to(device)
        obj1 = torch.randn(4, 24, device=device)
        obj2 = torch.randn(4, 24, device=device)
        out = model(obj1, obj2)
        assert out.shape == (4, 3)

    def test_risk_score_bounded(self, device, default_config):
        model = CollisionPredictor(default_config).to(device)
        model.eval()
        obj1 = torch.randn(8, 24, device=device)
        obj2 = torch.randn(8, 24, device=device)
        out = model(obj1, obj2)
        risk = out[:, 0]
        assert (risk >= 0).all() and (risk <= 1).all(), "Risk should be in [0,1] (sigmoid)"

    def test_ttca_and_distance_positive(self, device, default_config):
        model = CollisionPredictor(default_config).to(device)
        model.eval()
        obj1 = torch.randn(8, 24, device=device)
        obj2 = torch.randn(8, 24, device=device)
        out = model(obj1, obj2)
        ttca = out[:, 1]
        miss_dist = out[:, 2]
        assert (ttca > 0).all(), "TTCA should be positive (softplus)"
        assert (miss_dist > 0).all(), "Miss distance should be positive (softplus)"

    def test_predict_collision_risk_keys(self, device, default_config):
        model = CollisionPredictor(default_config).to(device)
        model.eval()
        obj1 = torch.randn(2, 24, device=device)
        obj2 = torch.randn(2, 24, device=device)
        result = model.predict_collision_risk(obj1, obj2)
        expected_keys = {'risk_score', 'time_to_closest_approach',
                         'miss_distance_km', 'is_high_risk', 'risk_level'}
        assert set(result.keys()) == expected_keys

    def test_risk_level_categorization(self, device, default_config):
        model = CollisionPredictor(default_config).to(device)
        scores = torch.tensor([0.1, 0.4, 0.7, 0.9], device=device)
        levels = model._categorize_risk(scores)
        assert levels[0].item() == 0  # Low
        assert levels[1].item() == 1  # Medium (>0.3)
        assert levels[2].item() == 2  # High (>0.6)
        assert levels[3].item() == 3  # Critical (>0.8)


# ──────────────────────────────────────────────
# Batch collision matrix
# ──────────────────────────────────────────────
class TestBatchCollisionMatrix:
    def test_correct_pair_count(self, device, default_config):
        model = CollisionPredictor(default_config).to(device)
        model.eval()
        n = 5
        features = torch.randn(n, 24, device=device)
        result = model.batch_collision_matrix(features, top_k=100)
        expected_pairs = n * (n - 1) // 2
        assert result['num_pairs_analyzed'] == expected_pairs

    def test_sorted_by_risk_descending(self, device, default_config):
        model = CollisionPredictor(default_config).to(device)
        model.eval()
        features = torch.randn(4, 24, device=device)
        result = model.batch_collision_matrix(features, top_k=100)
        risks = [p['risk_score'] for p in result['all_pairs']]
        assert risks == sorted(risks, reverse=True)

    def test_top_k_truncation(self, device, default_config):
        model = CollisionPredictor(default_config).to(device)
        model.eval()
        features = torch.randn(6, 24, device=device)
        result = model.batch_collision_matrix(features, top_k=3)
        assert len(result['top_k_risks']) == 3


# ──────────────────────────────────────────────
# Config & Factory
# ──────────────────────────────────────────────
class TestConfigAndFactory:
    def test_default_config(self):
        cfg = CollisionPredictorConfig()
        assert cfg.input_dim == 48
        assert cfg.hidden_dims == [256, 128, 64]
        assert cfg.output_dim == 3

    def test_get_config_from_config(self):
        cfg = CollisionPredictorConfig(dropout=0.5)
        model = CollisionPredictor(cfg)
        config_dict = model.get_config()
        model2 = CollisionPredictor.from_config(config_dict)
        assert model2.config.dropout == 0.5

    def test_risk_level_names(self):
        assert get_risk_level_name(0) == "Low"
        assert get_risk_level_name(1) == "Medium"
        assert get_risk_level_name(2) == "High"
        assert get_risk_level_name(3) == "Critical"
        assert get_risk_level_name(99) == "Unknown"
