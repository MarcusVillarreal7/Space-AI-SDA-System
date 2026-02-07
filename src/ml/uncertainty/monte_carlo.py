"""
Monte Carlo Dropout for Uncertainty Quantification.

Implements Bayesian approximation through dropout sampling to estimate
prediction uncertainty. Uses multiple forward passes with dropout enabled
to generate prediction distributions.

Based on: Gal & Ghahramani (2016) - "Dropout as a Bayesian Approximation"

Author: Space AI Team
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from src.utils.logging_config import get_logger

logger = get_logger("mc_dropout")


@dataclass
class MCDropoutConfig:
    """Configuration for Monte Carlo Dropout."""
    n_samples: int = 30  # Number of forward passes
    dropout_rate: float = 0.1  # Dropout probability
    quantiles: List[float] = None  # Prediction quantiles to compute
    
    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]  # 95% CI + quartiles


class MCDropoutPredictor:
    """
    Monte Carlo Dropout predictor for uncertainty quantification.
    
    Wraps any PyTorch model with dropout layers and performs multiple
    stochastic forward passes to estimate prediction uncertainty.
    
    Usage:
        mc_predictor = MCDropoutPredictor(model, n_samples=30)
        predictions, uncertainty = mc_predictor.predict_with_uncertainty(x)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[MCDropoutConfig] = None
    ):
        """
        Initialize MC Dropout predictor.
        
        Args:
            model: PyTorch model with dropout layers
            config: MC Dropout configuration
        """
        self.model = model
        self.config = config or MCDropoutConfig()
        self.device = next(model.parameters()).device
        
        # Verify model has dropout layers
        self.has_dropout = self._check_dropout_layers()
        if not self.has_dropout:
            logger.warning("Model has no dropout layers - uncertainty may be underestimated")
        
        logger.info(f"MC Dropout initialized with {self.config.n_samples} samples")
    
    def _check_dropout_layers(self) -> bool:
        """Check if model contains dropout layers."""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                return True
        return False
    
    def _enable_dropout(self):
        """Enable dropout in eval mode for MC sampling."""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_samples: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty quantification using MC Dropout.
        
        Args:
            x: Input tensor (batch, ...)
            return_samples: Whether to return all MC samples
        
        Returns:
            Dictionary containing:
                - 'mean': Mean prediction (batch, ...)
                - 'std': Standard deviation (batch, ...)
                - 'quantiles': Prediction quantiles (batch, n_quantiles, ...)
                - 'samples': All MC samples if return_samples=True (n_samples, batch, ...)
        """
        # Set model to eval mode but keep dropout enabled
        self.model.eval()
        self._enable_dropout()
        
        # Collect predictions from multiple forward passes
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.config.n_samples):
                output = self.model(x)
                predictions.append(output)
        
        # Stack predictions (n_samples, batch, ...)
        predictions = torch.stack(predictions, dim=0)
        
        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Compute quantiles
        quantiles = torch.quantile(
            predictions,
            torch.tensor(self.config.quantiles, device=self.device),
            dim=0
        )  # (n_quantiles, batch, ...)
        
        result = {
            'mean': mean_pred,
            'std': std_pred,
            'quantiles': quantiles,
            'epistemic_uncertainty': std_pred,  # Alias for clarity
        }
        
        if return_samples:
            result['samples'] = predictions
        
        return result
    
    def predict_trajectory_with_uncertainty(
        self,
        src: torch.Tensor,
        pred_horizon: int
    ) -> Dict[str, np.ndarray]:
        """
        Predict trajectory with uncertainty (for TrajectoryTransformer).
        
        Args:
            src: Source sequence (batch, seq_len, features)
            pred_horizon: Number of future timesteps
        
        Returns:
            Dictionary with predictions and uncertainty bounds
        """
        # Enable MC sampling
        self.model.eval()
        self._enable_dropout()
        
        # Collect trajectory predictions
        trajectories = []
        
        with torch.no_grad():
            for _ in range(self.config.n_samples):
                if hasattr(self.model, 'predict'):
                    traj = self.model.predict(src, pred_horizon)
                else:
                    # Fallback for generic models
                    traj = self.model(src)
                trajectories.append(traj)
        
        # Stack (n_samples, batch, horizon, features)
        trajectories = torch.stack(trajectories, dim=0)
        
        # Compute statistics
        mean_traj = trajectories.mean(dim=0)
        std_traj = trajectories.std(dim=0)
        
        # Compute prediction intervals (95% CI)
        lower_bound = torch.quantile(trajectories, 0.025, dim=0)
        upper_bound = torch.quantile(trajectories, 0.975, dim=0)
        
        # Convert to numpy
        result = {
            'mean': mean_traj.cpu().numpy(),
            'std': std_traj.cpu().numpy(),
            'lower_95': lower_bound.cpu().numpy(),
            'upper_95': upper_bound.cpu().numpy(),
            'samples': trajectories.cpu().numpy() if self.config.n_samples <= 50 else None
        }
        
        return result
    
    def predict_classification_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Predict classification with uncertainty (for ManeuverClassifier).
        
        Args:
            x: Input features (batch, seq_len, features)
        
        Returns:
            Dictionary with class probabilities and uncertainty metrics
        """
        # Enable MC sampling
        self.model.eval()
        self._enable_dropout()
        
        # Collect class probability predictions
        prob_samples = []
        
        with torch.no_grad():
            for _ in range(self.config.n_samples):
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1)
                prob_samples.append(probs)
        
        # Stack (n_samples, batch, num_classes)
        prob_samples = torch.stack(prob_samples, dim=0)
        
        # Mean probabilities
        mean_probs = prob_samples.mean(dim=0)
        
        # Predicted class (most frequent)
        predicted_classes = torch.argmax(prob_samples, dim=-1)  # (n_samples, batch)
        # Mode across samples
        final_predictions = torch.mode(predicted_classes, dim=0).values
        
        # Uncertainty metrics
        # 1. Predictive entropy (uncertainty in predictions)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
        
        # 2. Mutual information (epistemic uncertainty)
        # H(y|x) - E[H(y|x,θ)]
        expected_entropy = torch.mean(
            -torch.sum(prob_samples * torch.log(prob_samples + 1e-10), dim=-1),
            dim=0
        )
        mutual_information = entropy - expected_entropy
        
        # 3. Variation ratio (fraction of disagreement)
        mode_count = torch.sum(predicted_classes == final_predictions.unsqueeze(0), dim=0).float()
        variation_ratio = 1 - (mode_count / self.config.n_samples)
        
        result = {
            'probabilities': mean_probs.cpu().numpy(),
            'predicted_class': final_predictions.cpu().numpy(),
            'entropy': entropy.cpu().numpy(),
            'mutual_information': mutual_information.cpu().numpy(),
            'variation_ratio': variation_ratio.cpu().numpy(),
            'confidence': mean_probs.max(dim=-1).values.cpu().numpy(),
        }
        
        return result
    
    def calibrate(self, val_loader, num_bins: int = 15) -> Dict[str, float]:
        """
        Calibrate uncertainty estimates on validation set.
        
        Args:
            val_loader: Validation data loader
            num_bins: Number of bins for calibration
        
        Returns:
            Calibration metrics (ECE, MCE)
        """
        all_confidences = []
        all_accuracies = []
        
        for batch in val_loader:
            if isinstance(batch, dict):
                x, y = batch['input'], batch['target']
            else:
                x, y = batch
            
            x, y = x.to(self.device), y.to(self.device)
            
            # Get predictions with uncertainty
            result = self.predict_classification_with_uncertainty(x)
            confidences = result['confidence']
            predictions = result['predicted_class']
            
            # Compute accuracy
            if len(y.shape) > 1:
                y = y.argmax(dim=-1)
            accuracies = (predictions == y.cpu().numpy()).astype(float)
            
            all_confidences.extend(confidences)
            all_accuracies.extend(accuracies)
        
        all_confidences = np.array(all_confidences)
        all_accuracies = np.array(all_accuracies)
        
        # Compute Expected Calibration Error (ECE)
        bins = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(all_confidences, bins) - 1
        
        ece = 0.0
        mce = 0.0
        
        for i in range(num_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_conf = all_confidences[mask].mean()
                bin_acc = all_accuracies[mask].mean()
                bin_size = mask.sum() / len(all_confidences)
                
                calibration_error = abs(bin_conf - bin_acc)
                ece += bin_size * calibration_error
                mce = max(mce, calibration_error)
        
        logger.info(f"Calibration: ECE={ece:.4f}, MCE={mce:.4f}")
        
        return {
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce,
            'num_samples': len(all_confidences)
        }


# Convenience functions
def enable_mc_dropout(model: nn.Module):
    """Enable MC Dropout mode for a model (eval + dropout enabled)."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def disable_mc_dropout(model: nn.Module):
    """Disable MC Dropout mode (full eval)."""
    model.eval()


# Example usage
if __name__ == "__main__":
    print("Testing MC Dropout Predictor...\n")
    
    # Create dummy model with dropout
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 50)
            self.dropout = nn.Dropout(0.1)
            self.fc2 = nn.Linear(50, 3)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)
    
    model = DummyModel()
    mc_predictor = MCDropoutPredictor(model, MCDropoutConfig(n_samples=10))
    
    # Test prediction
    x = torch.randn(4, 10)
    result = mc_predictor.predict_with_uncertainty(x)
    
    print(f"Mean shape: {result['mean'].shape}")
    print(f"Std shape: {result['std'].shape}")
    print(f"Quantiles shape: {result['quantiles'].shape}")
    print(f"Mean prediction: {result['mean'][0]}")
    print(f"Uncertainty (std): {result['std'][0]}")
    
    print("\n✅ MC Dropout test passed!")
