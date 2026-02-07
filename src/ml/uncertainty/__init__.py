"""
Uncertainty Quantification for ML Models.

This module provides methods for quantifying prediction uncertainty in
satellite trajectory and maneuver classification models:

- Monte Carlo Dropout: Bayesian approximation through dropout sampling
- Ensemble Methods: Predictions from multiple model variants
- Conformal Prediction: Calibrated prediction intervals

All methods integrate seamlessly with the existing TrajectoryTransformer
and ManeuverClassifier models.

Author: Space AI Team
Date: 2026-02-07 (Restored after system crash)
"""

from src.ml.uncertainty.monte_carlo import MCDropoutPredictor
from src.ml.uncertainty.ensemble import EnsemblePredictor
from src.ml.uncertainty.conformal import ConformalPredictor

__all__ = [
    'MCDropoutPredictor',
    'EnsemblePredictor',
    'ConformalPredictor',
]

# Default uncertainty configurations
DEFAULT_MC_SAMPLES = 30
DEFAULT_CONFIDENCE_LEVEL = 0.9  # 90% confidence intervals
DEFAULT_ENSEMBLE_SIZE = 5

def get_uncertainty_config():
    """Get default uncertainty quantification configuration."""
    return {
        'mc_dropout': {
            'n_samples': DEFAULT_MC_SAMPLES,
            'dropout_rate': 0.1,
        },
        'ensemble': {
            'n_models': DEFAULT_ENSEMBLE_SIZE,
            'voting': 'soft',  # 'soft' or 'hard'
        },
        'conformal': {
            'confidence_level': DEFAULT_CONFIDENCE_LEVEL,
            'method': 'adaptive',  # 'adaptive' or 'split'
        }
    }
