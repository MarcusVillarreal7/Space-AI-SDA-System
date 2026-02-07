"""
Collision Risk Prediction Model.

Predicts the probability of close approach or collision between satellite pairs
based on their relative trajectories, orbital characteristics, and behavioral patterns.

This model complements the tracking engine by providing forward-looking
collision risk assessment for conjunction analysis.

Author: Space AI Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from src.utils.logging_config import get_logger

logger = get_logger("collision_predictor")


@dataclass
class CollisionPredictorConfig:
    """Configuration for collision predictor model."""
    input_dim: int = 48  # Features for 2 objects (24 each)
    hidden_dims: List[int] = None  # Hidden layer dimensions
    output_dim: int = 3  # Outputs: [risk_score, time_to_closest, miss_distance]
    dropout: float = 0.2
    use_attention: bool = True  # Use attention over relative trajectory
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


class RelativeTrajectoryEncoder(nn.Module):
    """
    Encode relative trajectory between two objects.
    
    Takes absolute trajectories of two satellites and computes
    relative features for collision analysis.
    """
    
    def __init__(self, feature_dim: int = 24):
        super().__init__()
        self.feature_dim = feature_dim
    
    def forward(
        self,
        obj1_features: torch.Tensor,
        obj2_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relative features.
        
        Args:
            obj1_features: Object 1 features (batch, feature_dim)
            obj2_features: Object 2 features (batch, feature_dim)
        
        Returns:
            Relative features (batch, combined_dim)
        """
        # Extract position and velocity (first 6 dims)
        pos1, vel1 = obj1_features[..., :3], obj1_features[..., 3:6]
        pos2, vel2 = obj2_features[..., :3], obj2_features[..., 3:6]
        
        # Compute relative quantities
        rel_pos = pos2 - pos1
        rel_vel = vel2 - vel1
        rel_distance = torch.norm(rel_pos, dim=-1, keepdim=True)
        rel_speed = torch.norm(rel_vel, dim=-1, keepdim=True)
        
        # Closing rate (negative = approaching)
        closing_rate = torch.sum(rel_pos * rel_vel, dim=-1, keepdim=True) / (rel_distance + 1e-8)
        
        # Time to closest approach (estimate)
        ttca = -closing_rate / (rel_speed**2 + 1e-8)
        ttca = torch.clamp(ttca, 0, 86400)  # Clamp to 0-24 hours
        
        # Concatenate: absolute features + relative features
        relative_features = torch.cat([
            rel_pos, rel_vel, rel_distance, rel_speed, closing_rate, ttca
        ], dim=-1)
        
        combined = torch.cat([obj1_features, obj2_features, relative_features], dim=-1)
        
        return combined


class CollisionPredictor(nn.Module):
    """
    Neural network for collision risk prediction.
    
    Predicts:
    1. Collision risk score (0-1)
    2. Time to closest approach (seconds)
    3. Estimated miss distance (km)
    
    Architecture:
        Input: Concatenated features from 2 objects + relative features
        Hidden layers: Multi-layer perceptron with dropout
        Output: 3 values (risk, time, distance)
    """
    
    def __init__(self, config: Optional[CollisionPredictorConfig] = None):
        super().__init__()
        
        if config is None:
            config = CollisionPredictorConfig()
        
        self.config = config
        
        # Relative trajectory encoder
        self.rel_encoder = RelativeTrajectoryEncoder(feature_dim=24)
        
        # Calculate actual input dimension after concatenation
        # obj1 (24) + obj2 (24) + relative (10) = 58
        actual_input_dim = 24 + 24 + 10
        
        # Build network
        layers = []
        in_dim = actual_input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, config.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        obj1_features: torch.Tensor,
        obj2_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for collision prediction.
        
        Args:
            obj1_features: Object 1 features (batch, 24)
            obj2_features: Object 2 features (batch, 24)
        
        Returns:
            Predictions (batch, 3) = [risk_score, ttca, miss_distance]
        """
        # Encode relative trajectory
        combined = self.rel_encoder(obj1_features, obj2_features)
        
        # Pass through network
        output = self.network(combined)
        
        # Apply activations to outputs
        # Risk score: sigmoid to [0, 1]
        risk_score = torch.sigmoid(output[..., 0:1])
        
        # Time to closest approach: softplus to ensure positive
        ttca = F.softplus(output[..., 1:2])
        
        # Miss distance: softplus to ensure positive
        miss_distance = F.softplus(output[..., 2:3])
        
        result = torch.cat([risk_score, ttca, miss_distance], dim=-1)
        
        return result
    
    def predict_collision_risk(
        self,
        obj1_features: torch.Tensor,
        obj2_features: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Predict collision risk with interpretation.
        
        Args:
            obj1_features: Object 1 features (batch, 24)
            obj2_features: Object 2 features (batch, 24)
            threshold: Risk threshold for collision alert
        
        Returns:
            Dictionary with predictions and risk assessment
        """
        output = self.forward(obj1_features, obj2_features)
        
        risk_score = output[..., 0]
        ttca = output[..., 1]
        miss_distance = output[..., 2]
        
        # Determine risk level
        is_high_risk = risk_score > threshold
        
        result = {
            'risk_score': risk_score,
            'time_to_closest_approach': ttca,
            'miss_distance_km': miss_distance,
            'is_high_risk': is_high_risk,
            'risk_level': self._categorize_risk(risk_score),
        }
        
        return result
    
    def _categorize_risk(self, risk_scores: torch.Tensor) -> torch.Tensor:
        """
        Categorize risk scores into levels.
        
        Args:
            risk_scores: Risk scores (batch,)
        
        Returns:
            Risk levels (batch,) - 0: Low, 1: Medium, 2: High, 3: Critical
        """
        levels = torch.zeros_like(risk_scores, dtype=torch.long)
        levels[risk_scores > 0.3] = 1  # Medium
        levels[risk_scores > 0.6] = 2  # High
        levels[risk_scores > 0.8] = 3  # Critical
        
        return levels
    
    def batch_collision_matrix(
        self,
        all_features: torch.Tensor,
        top_k: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Compute collision risk matrix for all object pairs.
        
        Args:
            all_features: Features for all objects (n_objects, 24)
            top_k: Return top K highest risk pairs
        
        Returns:
            Dictionary with risk matrix and top risky pairs
        """
        n_objects = all_features.size(0)
        
        # Compute pairwise risks (upper triangular only)
        risk_pairs = []
        
        for i in range(n_objects):
            for j in range(i + 1, n_objects):
                obj1 = all_features[i:i+1]
                obj2 = all_features[j:j+1]
                
                output = self.forward(obj1, obj2)
                risk_score = output[0, 0].item()
                ttca = output[0, 1].item()
                miss_dist = output[0, 2].item()
                
                risk_pairs.append({
                    'object1_idx': i,
                    'object2_idx': j,
                    'risk_score': risk_score,
                    'ttca': ttca,
                    'miss_distance': miss_dist
                })
        
        # Sort by risk score
        risk_pairs.sort(key=lambda x: x['risk_score'], reverse=True)
        
        # Get top K
        top_risks = risk_pairs[:top_k]
        
        logger.info(f"Analyzed {len(risk_pairs)} pairs, top risk: {top_risks[0]['risk_score']:.4f}")
        
        return {
            'all_pairs': risk_pairs,
            'top_k_risks': top_risks,
            'num_pairs_analyzed': len(risk_pairs),
            'max_risk': top_risks[0]['risk_score'] if top_risks else 0.0
        }
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            'input_dim': self.config.input_dim,
            'hidden_dims': self.config.hidden_dims,
            'output_dim': self.config.output_dim,
            'dropout': self.config.dropout,
            'use_attention': self.config.use_attention
        }
    
    @classmethod
    def from_config(cls, config_dict: Dict) -> 'CollisionPredictor':
        """Create model from configuration dictionary."""
        config = CollisionPredictorConfig(**config_dict)
        return cls(config)


# Risk level names
RISK_LEVELS = {
    0: "Low",
    1: "Medium",
    2: "High",
    3: "Critical"
}


def get_risk_level_name(level: int) -> str:
    """Get human-readable risk level name."""
    return RISK_LEVELS.get(level, "Unknown")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Collision Predictor...\n")
    
    config = CollisionPredictorConfig(
        input_dim=48,
        hidden_dims=[256, 128, 64],
        output_dim=3,
        dropout=0.2
    )
    
    model = CollisionPredictor(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    obj1_features = torch.randn(batch_size, 24)
    obj2_features = torch.randn(batch_size, 24)
    
    output = model(obj1_features, obj2_features)
    print(f"Output shape: {output.shape}")  # Should be (4, 3)
    print(f"Risk scores: {output[:, 0]}")
    print(f"TTCA: {output[:, 1]}")
    print(f"Miss distance: {output[:, 2]}")
    
    # Test risk prediction
    result = model.predict_collision_risk(obj1_features, obj2_features, threshold=0.5)
    print(f"\nRisk scores: {result['risk_score']}")
    print(f"High risk flags: {result['is_high_risk']}")
    print(f"Risk levels: {result['risk_level']}")
    
    # Test batch collision matrix
    all_features = torch.randn(10, 24)
    matrix_result = model.batch_collision_matrix(all_features, top_k=5)
    print(f"\nAnalyzed {matrix_result['num_pairs_analyzed']} pairs")
    print(f"Top risk: {matrix_result['max_risk']:.4f}")
    
    print("\n✅ Collision predictor test passed!")
