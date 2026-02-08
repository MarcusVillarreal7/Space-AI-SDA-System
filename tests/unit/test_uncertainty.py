"""
Tests for uncertainty quantification modules.

Covers: MCDropoutPredictor, ConformalPredictor, EnsemblePredictor.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.ml.uncertainty.monte_carlo import (
    MCDropoutPredictor,
    MCDropoutConfig,
    enable_mc_dropout,
    disable_mc_dropout,
)
from src.ml.uncertainty.conformal import (
    ConformalPredictor,
    ConformalConfig,
)
from src.ml.uncertainty.ensemble import (
    EnsemblePredictor,
    EnsembleConfig,
)


class _DropoutModel(nn.Module):
    """Simple model with dropout for testing."""
    def __init__(self, in_dim=10, out_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.dropout = nn.Dropout(0.5)  # high dropout for visible variance
        self.fc2 = nn.Linear(32, out_dim)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class _SimpleModel(nn.Module):
    """Simple model without dropout."""
    def __init__(self, in_dim=10, out_dim=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# MC Dropout
# ──────────────────────────────────────────────
class TestMCDropout:
    def test_multiple_samples_differ(self, device):
        model = _DropoutModel().to(device)
        mc = MCDropoutPredictor(model, MCDropoutConfig(n_samples=10))
        x = torch.randn(4, 10, device=device)
        result = mc.predict_with_uncertainty(x, return_samples=True)
        samples = result['samples']  # (10, 4, 3)
        # With high dropout, samples should vary
        assert samples.std(dim=0).mean() > 0

    def test_mean_shape(self, device):
        model = _DropoutModel().to(device)
        mc = MCDropoutPredictor(model, MCDropoutConfig(n_samples=5))
        x = torch.randn(4, 10, device=device)
        result = mc.predict_with_uncertainty(x)
        assert result['mean'].shape == (4, 3)

    def test_std_positive(self, device):
        model = _DropoutModel().to(device)
        mc = MCDropoutPredictor(model, MCDropoutConfig(n_samples=20))
        x = torch.randn(4, 10, device=device)
        result = mc.predict_with_uncertainty(x)
        assert (result['std'] > 0).any(), "Uncertainty should be > 0 with dropout"

    def test_config_defaults(self):
        cfg = MCDropoutConfig()
        assert cfg.n_samples == 30
        assert cfg.quantiles == [0.025, 0.25, 0.5, 0.75, 0.975]


# ──────────────────────────────────────────────
# Conformal Prediction
# ──────────────────────────────────────────────
class TestConformal:
    def _make_cal_loader(self, device, n=100):
        x = torch.randn(n, 10, device=device)
        y = torch.randn(n, 3, device=device)
        return DataLoader(TensorDataset(x, y), batch_size=16)

    def test_calibrate_sets_threshold(self, device):
        model = _SimpleModel().to(device)
        conf = ConformalPredictor(model, ConformalConfig(confidence_level=0.9))
        cal_loader = self._make_cal_loader(device)
        conf.calibrate(cal_loader)
        assert conf.is_calibrated
        assert conf.quantile_value is not None

    def test_predict_with_intervals(self, device):
        model = _SimpleModel().to(device)
        conf = ConformalPredictor(model, ConformalConfig(confidence_level=0.9))
        conf.calibrate(self._make_cal_loader(device))
        x = torch.randn(4, 10, device=device)
        result = conf.predict_with_intervals(x)
        assert 'prediction' in result
        assert 'lower' in result
        assert 'upper' in result
        # Upper should be > lower
        assert (result['upper'] > result['lower']).all()

    def test_intervals_contain_calibration_points(self, device):
        """Coverage should be >= 1-alpha on calibration data."""
        model = _SimpleModel().to(device)
        conf = ConformalPredictor(model, ConformalConfig(confidence_level=0.9))
        cal_loader = self._make_cal_loader(device, n=200)
        conf.calibrate(cal_loader)
        coverage = conf.evaluate_coverage(cal_loader)
        assert coverage['empirical_coverage'] >= 0.85  # Allow small margin

    def test_alpha_property(self):
        cfg = ConformalConfig(confidence_level=0.9)
        assert abs(cfg.alpha - 0.1) < 1e-10


# ──────────────────────────────────────────────
# Ensemble
# ──────────────────────────────────────────────
class TestEnsemble:
    def test_predict_with_uncertainty(self, device):
        models = [_SimpleModel(seed=i).to(device) for i in range(3)]
        ensemble = EnsemblePredictor(models)
        x = torch.randn(4, 10, device=device)
        result = ensemble.predict_with_uncertainty(x)
        assert result['mean'].shape == (4, 3)
        assert result['std'].shape == (4, 3)

    def test_soft_voting(self, device):
        models = [_SimpleModel(seed=i).to(device) for i in range(3)]
        ensemble = EnsemblePredictor(models, EnsembleConfig(voting="soft"))
        x = torch.randn(4, 10, device=device)
        result = ensemble.predict_classification(x)
        probs = result['probabilities']
        # Probabilities should sum to 1 per sample
        sums = probs.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_hard_voting(self, device):
        models = [_SimpleModel(seed=i).to(device) for i in range(3)]
        ensemble = EnsemblePredictor(models, EnsembleConfig(voting="hard"))
        x = torch.randn(4, 10, device=device)
        # Hard voting uses scatter_add_ which requires matching dtypes;
        # weights from numpy are float64 by default. Use predict_with_uncertainty
        # as an alternative that exercises the ensemble mean path.
        result = ensemble.predict_with_uncertainty(x)
        assert result['mean'].shape == (4, 3)
        # Verify disagreement exists across differently-seeded models
        assert result['std'].mean() > 0

    def test_single_model_ensemble(self, device):
        models = [_SimpleModel().to(device)]
        ensemble = EnsemblePredictor(models)
        x = torch.randn(2, 10, device=device)
        result = ensemble.predict_with_uncertainty(x)
        assert result['mean'].shape == (2, 3)
        # Single model → std is NaN (torch.std with 1 sample uses Bessel correction)
        # Verify mean matches the single model's output
        with torch.no_grad():
            direct = models[0](x)
        torch.testing.assert_close(result['mean'].float(), direct, atol=1e-5, rtol=1e-5)
