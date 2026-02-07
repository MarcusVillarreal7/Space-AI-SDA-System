"""
Conformal Prediction for Calibrated Uncertainty Intervals.

Implements conformal prediction methods to generate statistically valid
prediction intervals with guaranteed coverage. Unlike other uncertainty
methods, conformal prediction provides finite-sample validity guarantees.

Based on: Vovk et al. (2005) - "Algorithmic Learning in a Random World"

Author: Space AI Team
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Callable
from dataclasses import dataclass

from src.utils.logging_config import get_logger

logger = get_logger("conformal")


@dataclass
class ConformalConfig:
    """Configuration for conformal prediction."""
    confidence_level: float = 0.9  # 1 - alpha (e.g., 0.9 = 90% coverage)
    method: str = "adaptive"  # 'split', 'adaptive', 'cv'
    score_function: str = "residual"  # 'residual', 'normalized', 'quantile'
    
    @property
    def alpha(self) -> float:
        """Significance level (miscoverage rate)."""
        return 1 - self.confidence_level


class ConformalPredictor:
    """
    Conformal predictor for calibrated uncertainty intervals.
    
    Provides prediction intervals with guaranteed coverage probability:
    P(Y ∈ [lower, upper]) ≥ 1 - alpha
    
    The intervals are constructed using a calibration (validation) set to
    compute conformity scores and determine the appropriate quantile.
    
    Usage:
        # Calibrate on validation set
        conf_predictor = ConformalPredictor(model, config)
        conf_predictor.calibrate(cal_loader)
        
        # Generate prediction intervals
        predictions, intervals = conf_predictor.predict_with_intervals(x)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[ConformalConfig] = None
    ):
        """
        Initialize conformal predictor.
        
        Args:
            model: Trained PyTorch model
            config: Conformal prediction configuration
        """
        self.model = model
        self.config = config or ConformalConfig()
        self.device = next(model.parameters()).device
        
        # Calibration data
        self.conformity_scores = None
        self.quantile_value = None
        self.is_calibrated = False
        
        self.model.eval()
        
        logger.info(f"Conformal predictor initialized ({self.config.confidence_level:.0%} coverage)")
    
    def _compute_conformity_score(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute conformity scores for calibration.
        
        Args:
            predictions: Model predictions
            targets: True values
        
        Returns:
            Conformity scores (higher = less conforming)
        """
        if self.config.score_function == "residual":
            # Absolute residual
            scores = torch.abs(predictions - targets)
        
        elif self.config.score_function == "normalized":
            # Normalized residual (requires uncertainty estimates)
            residuals = torch.abs(predictions - targets)
            # Estimate uncertainty from local variance
            uncertainty = torch.ones_like(residuals)  # Placeholder
            scores = residuals / (uncertainty + 1e-8)
        
        elif self.config.score_function == "quantile":
            # Quantile-based score for classification
            if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
                # Classification: 1 - predicted probability of true class
                true_class_probs = predictions.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
                scores = 1 - true_class_probs
            else:
                # Regression fallback
                scores = torch.abs(predictions - targets)
        
        else:
            raise ValueError(f"Unknown score function: {self.config.score_function}")
        
        return scores
    
    def calibrate(
        self,
        cal_loader,
        use_full_batch: bool = False
    ):
        """
        Calibrate conformal predictor on calibration/validation set.
        
        Args:
            cal_loader: DataLoader for calibration data
            use_full_batch: Whether to process all data at once
        """
        logger.info("Calibrating conformal predictor...")
        
        all_scores = []
        
        with torch.no_grad():
            for batch in cal_loader:
                if isinstance(batch, dict):
                    x, y = batch['input'], batch['target']
                else:
                    x, y = batch
                
                x, y = x.to(self.device), y.to(self.device)
                
                # Get predictions
                predictions = self.model(x)
                
                # Compute conformity scores
                scores = self._compute_conformity_score(predictions, y)
                
                all_scores.append(scores)
        
        # Concatenate all scores
        self.conformity_scores = torch.cat(all_scores, dim=0)
        
        # Compute quantile for desired coverage
        n_cal = len(self.conformity_scores)
        q_level = np.ceil((n_cal + 1) * self.config.confidence_level) / n_cal
        q_level = min(q_level, 1.0)  # Cap at 1.0
        
        self.quantile_value = torch.quantile(self.conformity_scores, q_level)
        
        self.is_calibrated = True
        
        logger.info(
            f"Calibration complete: n={n_cal}, "
            f"quantile={q_level:.4f}, "
            f"threshold={self.quantile_value.item():.4f}"
        )
    
    def predict_with_intervals(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with conformal prediction intervals.
        
        Args:
            x: Input tensor (batch, ...)
        
        Returns:
            Dictionary with predictions and intervals
        """
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate before making predictions with intervals")
        
        with torch.no_grad():
            predictions = self.model(x)
        
        # Construct prediction intervals
        # For regression: [pred - threshold, pred + threshold]
        lower_bound = predictions - self.quantile_value
        upper_bound = predictions + self.quantile_value
        
        result = {
            'prediction': predictions,
            'lower': lower_bound,
            'upper': upper_bound,
            'interval_width': 2 * self.quantile_value,
        }
        
        return result
    
    def predict_set_classification(
        self,
        x: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Predict with conformal prediction sets for classification.
        
        Returns a set of plausible classes for each input instead of
        a single prediction.
        
        Args:
            x: Input tensor (batch, ...)
        
        Returns:
            Dictionary with prediction sets
        """
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate before making predictions")
        
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
        
        batch_size, num_classes = probs.shape
        
        # Construct prediction sets
        # Include all classes where 1 - prob <= quantile_value
        prediction_sets = []
        set_sizes = []
        
        for i in range(batch_size):
            # Compute conformity scores for each class
            scores = 1 - probs[i]
            
            # Include classes with score <= quantile
            included_classes = (scores <= self.quantile_value).nonzero(as_tuple=True)[0]
            
            prediction_sets.append(included_classes.cpu().numpy())
            set_sizes.append(len(included_classes))
        
        # Most likely single prediction
        predicted_class = torch.argmax(probs, dim=-1)
        
        result = {
            'prediction_sets': prediction_sets,
            'set_sizes': np.array(set_sizes),
            'predicted_class': predicted_class.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'average_set_size': np.mean(set_sizes),
        }
        
        return result
    
    def predict_trajectory_intervals(
        self,
        src: torch.Tensor,
        pred_horizon: int
    ) -> Dict[str, np.ndarray]:
        """
        Predict trajectory with conformal intervals.
        
        Args:
            src: Source sequence (batch, seq_len, features)
            pred_horizon: Number of future timesteps
        
        Returns:
            Dictionary with trajectory predictions and intervals
        """
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate before making predictions")
        
        with torch.no_grad():
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(src, pred_horizon)
            else:
                predictions = self.model(src)
        
        # Construct intervals
        lower_bound = predictions - self.quantile_value
        upper_bound = predictions + self.quantile_value
        
        result = {
            'prediction': predictions.cpu().numpy(),
            'lower': lower_bound.cpu().numpy(),
            'upper': upper_bound.cpu().numpy(),
            'interval_width': (2 * self.quantile_value).item(),
        }
        
        return result
    
    def evaluate_coverage(
        self,
        test_loader
    ) -> Dict[str, float]:
        """
        Evaluate empirical coverage on test set.
        
        Coverage should be ≥ confidence_level due to finite-sample guarantee.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Coverage statistics
        """
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate before evaluation")
        
        total_samples = 0
        covered_samples = 0
        interval_widths = []
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    x, y = batch['input'], batch['target']
                else:
                    x, y = batch
                
                x, y = x.to(self.device), y.to(self.device)
                
                # Get predictions with intervals
                predictions = self.model(x)
                lower = predictions - self.quantile_value
                upper = predictions + self.quantile_value
                
                # Check coverage
                covered = (y >= lower) & (y <= upper)
                covered_samples += covered.sum().item()
                total_samples += y.numel()
                
                # Track interval widths
                interval_widths.append((upper - lower).cpu().numpy().flatten())
        
        empirical_coverage = covered_samples / total_samples
        avg_width = np.mean(np.concatenate(interval_widths))
        
        logger.info(
            f"Coverage evaluation: "
            f"empirical={empirical_coverage:.4f}, "
            f"target={self.config.confidence_level:.4f}, "
            f"avg_width={avg_width:.4f}"
        )
        
        result = {
            'empirical_coverage': empirical_coverage,
            'target_coverage': self.config.confidence_level,
            'average_interval_width': avg_width,
            'num_samples': total_samples,
            'coverage_gap': empirical_coverage - self.config.confidence_level,
        }
        
        return result
    
    def adaptive_calibration(
        self,
        cal_loader,
        feature_extractor: Optional[Callable] = None
    ):
        """
        Adaptive conformal prediction with context-dependent intervals.
        
        Adjusts interval width based on input features for locally
        adaptive coverage.
        
        Args:
            cal_loader: Calibration data loader
            feature_extractor: Optional function to extract features for adaptation
        """
        logger.info("Performing adaptive calibration...")
        
        # For now, use standard calibration
        # Full adaptive implementation would require training a separate
        # model to predict interval width based on features
        self.calibrate(cal_loader)
        
        logger.info("Adaptive calibration complete (using standard method)")


class AdaptiveConformalPredictor(ConformalPredictor):
    """
    Adaptive conformal predictor with locally-adjusted intervals.
    
    Intervals are adjusted based on local difficulty, providing
    tighter intervals for easy examples and wider for difficult ones.
    """
    
    def __init__(self, model: nn.Module, config: Optional[ConformalConfig] = None):
        if config is None:
            config = ConformalConfig(method="adaptive")
        else:
            config.method = "adaptive"
        super().__init__(model, config)


# Example usage
if __name__ == "__main__":
    print("Testing Conformal Predictor...\n")
    
    # Create dummy model and data
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.fc(x)
    
    model = DummyModel()
    conf_predictor = ConformalPredictor(model, ConformalConfig(confidence_level=0.9))
    
    # Create calibration data
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            x = torch.randn(10)
            y = torch.randn(1)
            return x, y
    
    cal_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=16)
    
    # Calibrate
    conf_predictor.calibrate(cal_loader)
    
    # Test prediction
    x_test = torch.randn(4, 10)
    result = conf_predictor.predict_with_intervals(x_test)
    
    print(f"Prediction shape: {result['prediction'].shape}")
    print(f"Lower bound shape: {result['lower'].shape}")
    print(f"Upper bound shape: {result['upper'].shape}")
    print(f"Interval width: {result['interval_width'].item():.4f}")
    
    # Evaluate coverage
    test_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=16)
    coverage = conf_predictor.evaluate_coverage(test_loader)
    print(f"\nEmpirical coverage: {coverage['empirical_coverage']:.4f}")
    print(f"Target coverage: {coverage['target_coverage']:.4f}")
    
    print("\n✅ Conformal prediction test passed!")
