"""
Ensemble Methods for Uncertainty Quantification.

Implements ensemble prediction using multiple independently trained models.
Combines predictions through voting (classification) or averaging (regression)
to improve robustness and estimate uncertainty.

Author: Space AI Team
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
from dataclasses import dataclass

from src.utils.logging_config import get_logger

logger = get_logger("ensemble")


@dataclass
class EnsembleConfig:
    """Configuration for ensemble prediction."""
    voting: str = "soft"  # 'soft' (average probs) or 'hard' (majority vote)
    weights: Optional[List[float]] = None  # Optional model weights
    bootstrap: bool = False  # Whether models were trained on bootstrap samples
    

class EnsemblePredictor:
    """
    Ensemble predictor for uncertainty quantification.
    
    Combines predictions from multiple independently trained models to:
    1. Improve prediction accuracy
    2. Quantify model uncertainty (epistemic)
    3. Increase robustness to individual model failures
    
    Usage:
        models = [model1, model2, model3]
        ensemble = EnsemblePredictor(models)
        predictions, uncertainty = ensemble.predict_with_uncertainty(x)
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        config: Optional[EnsembleConfig] = None
    ):
        """
        Initialize ensemble predictor.
        
        Args:
            models: List of trained PyTorch models
            config: Ensemble configuration
        """
        if len(models) == 0:
            raise ValueError("Ensemble requires at least one model")
        
        self.models = models
        self.config = config or EnsembleConfig()
        self.n_models = len(models)
        self.device = next(models[0].parameters()).device
        
        # Setup weights
        if self.config.weights is None:
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            assert len(self.config.weights) == self.n_models
            self.weights = np.array(self.config.weights)
            self.weights /= self.weights.sum()  # Normalize
        
        # Set all models to eval mode
        for model in self.models:
            model.eval()
            model.to(self.device)
        
        logger.info(f"Ensemble initialized with {self.n_models} models, {self.config.voting} voting")
    
    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: List[str],
        model_class,
        model_config: Optional[Dict] = None
    ) -> 'EnsemblePredictor':
        """
        Load ensemble from checkpoint files.
        
        Args:
            checkpoint_paths: List of paths to model checkpoints
            model_class: Model class to instantiate
            model_config: Model configuration dictionary
        
        Returns:
            EnsemblePredictor instance
        """
        models = []
        
        for path in checkpoint_paths:
            # Create model
            if model_config is not None:
                if hasattr(model_class, 'from_config'):
                    model = model_class.from_config(model_config)
                else:
                    model = model_class(**model_config)
            else:
                model = model_class()
            
            # Load checkpoint
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            
            models.append(model)
            logger.info(f"Loaded model from {path}")
        
        return cls(models)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty quantification using ensemble.
        
        Args:
            x: Input tensor (batch, ...)
            return_individual: Whether to return individual model predictions
        
        Returns:
            Dictionary containing:
                - 'mean': Mean prediction across ensemble
                - 'std': Standard deviation (model disagreement)
                - 'min': Minimum prediction
                - 'max': Maximum prediction
                - 'individual': Individual predictions if return_individual=True
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                predictions.append(output)
        
        # Stack predictions (n_models, batch, ...)
        predictions = torch.stack(predictions, dim=0)
        
        # Weighted mean
        weights_tensor = torch.tensor(self.weights, device=self.device).view(-1, *([1] * (predictions.ndim - 1)))
        weighted_mean = (predictions * weights_tensor).sum(dim=0)
        
        # Statistics
        std = predictions.std(dim=0)
        min_pred = predictions.min(dim=0).values
        max_pred = predictions.max(dim=0).values
        
        result = {
            'mean': weighted_mean,
            'std': std,
            'min': min_pred,
            'max': max_pred,
            'epistemic_uncertainty': std,
        }
        
        if return_individual:
            result['individual'] = predictions
        
        return result
    
    def predict_classification(
        self,
        x: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Ensemble classification with uncertainty.
        
        Args:
            x: Input tensor (batch, ...)
        
        Returns:
            Dictionary with class predictions and uncertainty
        """
        logits_list = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                logits_list.append(logits)
        
        # Stack (n_models, batch, num_classes)
        logits = torch.stack(logits_list, dim=0)
        probs = torch.softmax(logits, dim=-1)
        
        if self.config.voting == "soft":
            # Average probabilities (weighted)
            weights_tensor = torch.tensor(self.weights, device=self.device).view(-1, 1, 1)
            mean_probs = (probs * weights_tensor).sum(dim=0)
            predicted_class = torch.argmax(mean_probs, dim=-1)
            confidence = mean_probs.max(dim=-1).values
        
        else:  # hard voting
            # Majority vote on class predictions
            predicted_classes = torch.argmax(logits, dim=-1)  # (n_models, batch)
            # Count votes for each class
            batch_size = predicted_classes.shape[1]
            num_classes = logits.shape[-1]
            
            votes = torch.zeros(batch_size, num_classes, device=self.device)
            for i in range(self.n_models):
                votes.scatter_add_(1, predicted_classes[i:i+1].T, torch.tensor([[self.weights[i]]], device=self.device))
            
            predicted_class = torch.argmax(votes, dim=-1)
            confidence = votes.max(dim=-1).values / self.weights.sum()
            
            # Use soft probs for mean_probs
            weights_tensor = torch.tensor(self.weights, device=self.device).view(-1, 1, 1)
            mean_probs = (probs * weights_tensor).sum(dim=0)
        
        # Uncertainty metrics
        # Entropy of mean prediction
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
        
        # Variance across models (model disagreement)
        prob_std = probs.std(dim=0).mean(dim=-1)
        
        result = {
            'probabilities': mean_probs.cpu().numpy(),
            'predicted_class': predicted_class.cpu().numpy(),
            'confidence': confidence.cpu().numpy(),
            'entropy': entropy.cpu().numpy(),
            'model_disagreement': prob_std.cpu().numpy(),
        }
        
        return result
    
    def predict_trajectory_ensemble(
        self,
        src: torch.Tensor,
        pred_horizon: int
    ) -> Dict[str, np.ndarray]:
        """
        Ensemble trajectory prediction.
        
        Args:
            src: Source sequence (batch, seq_len, features)
            pred_horizon: Number of future timesteps
        
        Returns:
            Dictionary with ensemble trajectories and uncertainty
        """
        trajectories = []
        
        with torch.no_grad():
            for model in self.models:
                if hasattr(model, 'predict'):
                    traj = model.predict(src, pred_horizon)
                else:
                    traj = model(src)
                trajectories.append(traj)
        
        # Stack (n_models, batch, horizon, features)
        trajectories = torch.stack(trajectories, dim=0)
        
        # Weighted mean
        weights_tensor = torch.tensor(self.weights, device=self.device).view(-1, 1, 1, 1)
        mean_traj = (trajectories * weights_tensor).sum(dim=0)
        
        # Statistics
        std_traj = trajectories.std(dim=0)
        min_traj = trajectories.min(dim=0).values
        max_traj = trajectories.max(dim=0).values
        
        result = {
            'mean': mean_traj.cpu().numpy(),
            'std': std_traj.cpu().numpy(),
            'min': min_traj.cpu().numpy(),
            'max': max_traj.cpu().numpy(),
        }
        
        return result
    
    def evaluate_diversity(self, val_loader) -> Dict[str, float]:
        """
        Evaluate ensemble diversity on validation set.
        
        Measures disagreement between ensemble members.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Diversity metrics
        """
        all_disagreements = []
        all_accuracies = []
        
        for batch in val_loader:
            if isinstance(batch, dict):
                x, y = batch['input'], batch['target']
            else:
                x, y = batch
            
            x, y = x.to(self.device), y.to(self.device)
            
            # Get predictions from all models
            predictions = []
            with torch.no_grad():
                for model in self.models:
                    logits = model(x)
                    pred_class = torch.argmax(logits, dim=-1)
                    predictions.append(pred_class)
            
            predictions = torch.stack(predictions, dim=0)  # (n_models, batch)
            
            # Compute pairwise disagreement
            for i in range(self.n_models):
                for j in range(i + 1, self.n_models):
                    disagreement = (predictions[i] != predictions[j]).float().mean().item()
                    all_disagreements.append(disagreement)
            
            # Ensemble accuracy
            majority_vote = torch.mode(predictions, dim=0).values
            if len(y.shape) > 1:
                y = y.argmax(dim=-1)
            accuracy = (majority_vote == y).float().mean().item()
            all_accuracies.append(accuracy)
        
        avg_disagreement = np.mean(all_disagreements)
        avg_accuracy = np.mean(all_accuracies)
        
        logger.info(f"Ensemble diversity: {avg_disagreement:.4f}, Accuracy: {avg_accuracy:.4f}")
        
        return {
            'average_disagreement': avg_disagreement,
            'ensemble_accuracy': avg_accuracy,
            'num_models': self.n_models
        }


class BootstrapEnsemble(EnsemblePredictor):
    """
    Bootstrap ensemble trained on resampled datasets.
    
    Each model is trained on a different bootstrap sample of the training data,
    which increases diversity and improves uncertainty estimates.
    """
    
    def __init__(self, models: List[nn.Module], config: Optional[EnsembleConfig] = None):
        if config is None:
            config = EnsembleConfig(bootstrap=True)
        else:
            config.bootstrap = True
        super().__init__(models, config)


# Example usage
if __name__ == "__main__":
    print("Testing Ensemble Predictor...\n")
    
    # Create dummy models
    class DummyModel(nn.Module):
        def __init__(self, seed=0):
            super().__init__()
            torch.manual_seed(seed)
            self.fc = nn.Linear(10, 3)
        
        def forward(self, x):
            return self.fc(x)
    
    models = [DummyModel(seed=i) for i in range(5)]
    ensemble = EnsemblePredictor(models)
    
    # Test prediction
    x = torch.randn(4, 10)
    result = ensemble.predict_with_uncertainty(x)
    
    print(f"Mean shape: {result['mean'].shape}")
    print(f"Std shape: {result['std'].shape}")
    print(f"Model disagreement: {result['std'].mean().item():.4f}")
    
    # Test classification
    result_class = ensemble.predict_classification(x)
    print(f"\nPredicted classes: {result_class['predicted_class']}")
    print(f"Confidence: {result_class['confidence']}")
    print(f"Model disagreement: {result_class['model_disagreement']}")
    
    print("\n✅ Ensemble test passed!")
