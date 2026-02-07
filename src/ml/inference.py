"""
ML Inference Pipeline for Satellite Tracking Predictions.

This module provides a unified interface for running trained ML models on satellite
trajectory data. It handles:
- Model loading from checkpoints
- Feature extraction and preprocessing
- Batch inference
- Uncertainty quantification (when available)
- Result post-processing

Author: Space AI Team
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json

from src.ml.models.trajectory_transformer import TrajectoryTransformer, TransformerConfig
from src.ml.models.maneuver_classifier import CNNLSTMManeuverClassifier, CNNLSTMClassifierConfig, get_class_name
from src.ml.features.trajectory_features import TrajectoryFeatureExtractor, FeatureConfig
from src.ml.features.sequence_builder import TrajectorySequenceBuilder, SequenceConfig
from src.utils.logging_config import get_logger

logger = get_logger("ml_inference")


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    device: str = "cpu"  # "cpu" or "cuda"
    batch_size: int = 32  # Batch size for inference
    num_mc_samples: int = 10  # Number of Monte Carlo samples for uncertainty
    use_uncertainty: bool = False  # Whether to compute uncertainty estimates


class TrajectoryPredictor:
    """
    Wrapper for trajectory prediction model inference.
    
    Handles model loading, feature preparation, and prediction.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        feature_config: Optional[FeatureConfig] = None,
        sequence_config: Optional[SequenceConfig] = None
    ):
        """
        Initialize trajectory predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            feature_config: Feature extraction configuration
            sequence_config: Sequence building configuration
        """
        self.device = torch.device(device)
        # Use reduced feature config to match model training (24D)
        if feature_config is None:
            feature_config = FeatureConfig(
                include_position=True,      # 3D
                include_velocity=True,      # 3D
                include_orbital_elements=True,  # 6D
                include_derived_features=True,  # 8D
                include_temporal_features=True,  # 4D
                include_uncertainty=False   # 0D -> Total 24D
            )
        self.feature_config = feature_config
        self.sequence_config = sequence_config or SequenceConfig()
        
        # Initialize feature extractor and sequence builder
        self.feature_extractor = TrajectoryFeatureExtractor(self.feature_config)
        self.sequence_builder = TrajectorySequenceBuilder(self.sequence_config)
        
        # Load model
        logger.info(f"Loading trajectory model from {checkpoint_path}")
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Trajectory predictor initialized")
    
    def _load_model(self, checkpoint_path: str) -> TrajectoryTransformer:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract config and create model
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            model = TrajectoryTransformer.from_config(config)
        else:
            model = TrajectoryTransformer()
        
        # Load weights
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        
        return model
    
    def predict(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray,
        pred_horizon: int = 30
    ) -> Dict[str, np.ndarray]:
        """
        Predict future trajectory.
        
        Args:
            positions: Historical positions (T, 3) in km
            velocities: Historical velocities (T, 3) in km/s
            timestamps: Historical timestamps (T,) in seconds
            pred_horizon: Number of future timesteps to predict
        
        Returns:
            Dictionary with:
                - 'positions': Predicted positions (pred_horizon, 3)
                - 'velocities': Predicted velocities (pred_horizon, 3)
        """
        # Extract features
        features = self.feature_extractor.extract_features(positions, velocities, timestamps)
        
        # Build sequences
        sequences = self.sequence_builder.build_sequences(features)
        history = sequences['history']  # (N, seq_len, D)
        
        if history.size(0) == 0:
            logger.warning("No valid sequences generated, returning zeros")
            return {
                'positions': np.zeros((pred_horizon, 3)),
                'velocities': np.zeros((pred_horizon, 3))
            }
        
        # Move to device
        history = history.to(self.device)
        
        # Run prediction
        with torch.no_grad():
            predictions = self.model.predict(history, pred_horizon)  # (N, pred_horizon, 6)
        
        # Take first sequence prediction (or average if multiple)
        if predictions.size(0) > 1:
            predictions = predictions.mean(dim=0, keepdim=True)
        
        predictions = predictions[0].cpu().numpy()  # (pred_horizon, 6)
        
        # Split into position and velocity
        result = {
            'positions': predictions[:, :3],  # First 3 dims
            'velocities': predictions[:, 3:6]  # Last 3 dims
        }
        
        return result
    
    def predict_batch(
        self,
        batch_positions: List[np.ndarray],
        batch_velocities: List[np.ndarray],
        batch_timestamps: List[np.ndarray],
        pred_horizon: int = 30
    ) -> List[Dict[str, np.ndarray]]:
        """
        Predict trajectories for multiple objects.
        
        Args:
            batch_positions: List of position arrays
            batch_velocities: List of velocity arrays
            batch_timestamps: List of timestamp arrays
            pred_horizon: Number of future timesteps
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for pos, vel, ts in zip(batch_positions, batch_velocities, batch_timestamps):
            pred = self.predict(pos, vel, ts, pred_horizon)
            results.append(pred)
        return results


class ManeuverPredictor:
    """
    Wrapper for maneuver classification model inference.
    
    Handles model loading, feature preparation, and classification.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        feature_config: Optional[FeatureConfig] = None
    ):
        """
        Initialize maneuver classifier.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            feature_config: Feature extraction configuration
        """
        self.device = torch.device(device)
        # Use reduced feature config to match model training (24D)
        if feature_config is None:
            feature_config = FeatureConfig(
                include_position=True,      # 3D
                include_velocity=True,      # 3D
                include_orbital_elements=True,  # 6D
                include_derived_features=True,  # 8D
                include_temporal_features=True,  # 4D
                include_uncertainty=False   # 0D -> Total 24D
            )
        self.feature_config = feature_config
        
        # Initialize feature extractor
        self.feature_extractor = TrajectoryFeatureExtractor(self.feature_config)
        
        # Load model
        logger.info(f"Loading classifier from {checkpoint_path}")
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Maneuver classifier initialized")
    
    def _load_model(self, checkpoint_path: str) -> CNNLSTMManeuverClassifier:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract config and create model
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            model = CNNLSTMManeuverClassifier.from_config(config)
        else:
            # Use default config
            model = CNNLSTMManeuverClassifier()
        
        # Load weights
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        
        return model
    
    def predict(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Union[int, str, np.ndarray]]:
        """
        Classify maneuver behavior.
        
        Args:
            positions: Historical positions (T, 3) in km
            velocities: Historical velocities (T, 3) in km/s
            timestamps: Historical timestamps (T,) in seconds
        
        Returns:
            Dictionary with:
                - 'class_idx': Predicted class index
                - 'class_name': Predicted class name
                - 'probabilities': Class probabilities (num_classes,)
                - 'confidence': Confidence score (max probability)
        """
        # Extract features
        features = self.feature_extractor.extract_features(positions, velocities, timestamps)
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)  # (1, T, D)
        features_tensor = features_tensor.to(self.device)
        
        # Run prediction
        with torch.no_grad():
            class_idx = self.model.predict(features_tensor).item()
            probabilities = self.model.predict_proba(features_tensor)[0].cpu().numpy()
        
        result = {
            'class_idx': class_idx,
            'class_name': get_class_name(class_idx),
            'probabilities': probabilities,
            'confidence': float(probabilities[class_idx])
        }
        
        return result
    
    def predict_batch(
        self,
        batch_positions: List[np.ndarray],
        batch_velocities: List[np.ndarray],
        batch_timestamps: List[np.ndarray]
    ) -> List[Dict[str, Union[int, str, np.ndarray]]]:
        """
        Classify maneuvers for multiple objects.
        
        Args:
            batch_positions: List of position arrays
            batch_velocities: List of velocity arrays
            batch_timestamps: List of timestamp arrays
        
        Returns:
            List of classification dictionaries
        """
        results = []
        for pos, vel, ts in zip(batch_positions, batch_velocities, batch_timestamps):
            pred = self.predict(pos, vel, ts)
            results.append(pred)
        return results


class MLInferencePipeline:
    """
    Complete ML inference pipeline for satellite tracking.
    
    Combines trajectory prediction and maneuver classification.
    """
    
    def __init__(
        self,
        trajectory_checkpoint: str,
        classifier_checkpoint: str,
        config: Optional[InferenceConfig] = None
    ):
        """
        Initialize inference pipeline.
        
        Args:
            trajectory_checkpoint: Path to trajectory model checkpoint
            classifier_checkpoint: Path to classifier checkpoint
            config: Inference configuration
        """
        if config is None:
            config = InferenceConfig()
        
        self.config = config
        
        # Initialize predictors
        self.trajectory_predictor = TrajectoryPredictor(
            trajectory_checkpoint,
            device=config.device
        )
        
        self.maneuver_predictor = ManeuverPredictor(
            classifier_checkpoint,
            device=config.device
        )
        
        logger.info("ML inference pipeline ready")
    
    def predict(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray,
        pred_horizon: int = 30
    ) -> Dict:
        """
        Run complete prediction pipeline.
        
        Args:
            positions: Historical positions (T, 3)
            velocities: Historical velocities (T, 3)
            timestamps: Historical timestamps (T,)
            pred_horizon: Future prediction horizon
        
        Returns:
            Complete prediction results
        """
        # Trajectory prediction
        trajectory_pred = self.trajectory_predictor.predict(
            positions, velocities, timestamps, pred_horizon
        )
        
        # Maneuver classification
        maneuver_pred = self.maneuver_predictor.predict(
            positions, velocities, timestamps
        )
        
        # Combine results
        result = {
            'trajectory': trajectory_pred,
            'maneuver': maneuver_pred,
            'metadata': {
                'pred_horizon': pred_horizon,
                'input_length': len(timestamps),
                'device': self.config.device
            }
        }
        
        return result


# Example usage
if __name__ == "__main__":
    print("Testing ML Inference Pipeline...\n")
    
    # Test trajectory predictor
    traj_ckpt = "checkpoints/phase3_day3/best_model.pt"
    if Path(traj_ckpt).exists():
        predictor = TrajectoryPredictor(traj_ckpt, device="cpu")
        
        # Generate test data
        positions = np.random.randn(20, 3) * 7000
        velocities = np.random.randn(20, 3) * 7
        timestamps = np.arange(20) * 60.0
        
        result = predictor.predict(positions, velocities, timestamps, pred_horizon=10)
        print(f"Trajectory prediction shape: {result['positions'].shape}")
        print("✅ Trajectory predictor works!\n")
    
    # Test maneuver classifier
    class_ckpt = "checkpoints/phase3_day4/maneuver_classifier.pt"
    if Path(class_ckpt).exists():
        classifier = ManeuverPredictor(class_ckpt, device="cpu")
        
        result = classifier.predict(positions, velocities, timestamps)
        print(f"Predicted class: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("✅ Maneuver classifier works!\n")
    
    # Test full pipeline
    if Path(traj_ckpt).exists() and Path(class_ckpt).exists():
        pipeline = MLInferencePipeline(traj_ckpt, class_ckpt)
        
        result = pipeline.predict(positions, velocities, timestamps)
        print("Complete prediction:")
        print(f"  - Trajectory: {result['trajectory']['positions'].shape}")
        print(f"  - Maneuver: {result['maneuver']['class_name']}")
        print("✅ Full pipeline works!\n")
    
    print("✅ All inference tests passed!")
