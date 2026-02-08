"""
Anomaly Detection for Satellite Behavior Analysis.

Uses a reconstruction-based autoencoder on 19D behavioral feature vectors
to detect anomalous satellite behavior patterns. Features include maneuver
statistics, timing patterns, orbital characteristics, classification history,
and orbital regime encoding.
"""

from src.ml.anomaly.behavior_features import (
    BehaviorProfile,
    BehaviorFeatureExtractor,
)
from src.ml.anomaly.autoencoder import BehaviorAutoencoder, AutoencoderConfig
from src.ml.anomaly.anomaly_detector import AnomalyDetector, AnomalyResult
from src.ml.anomaly.explainer import AnomalyExplainer

__all__ = [
    "BehaviorProfile",
    "BehaviorFeatureExtractor",
    "BehaviorAutoencoder",
    "AutoencoderConfig",
    "AnomalyDetector",
    "AnomalyResult",
    "AnomalyExplainer",
]
