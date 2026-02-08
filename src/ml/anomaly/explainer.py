"""
Human-Readable Anomaly Explanations.

Translates per-feature reconstruction errors into natural-language
explanations that operators can understand without ML expertise.

Author: Space AI Team
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.ml.anomaly.behavior_features import FEATURE_NAMES, BehaviorProfile


# Readable descriptions for each feature group
_GROUP_DESCRIPTIONS = {
    "maneuver_count": "maneuver frequency",
    "maneuver_rate": "maneuver rate",
    "mean_delta_v": "average thrust magnitude",
    "max_delta_v": "peak thrust magnitude",
    "delta_v_variance": "thrust variability",
    "mean_interval_s": "maneuver timing interval",
    "interval_regularity": "timing regularity",
    "altitude_km": "orbital altitude",
    "eccentricity_proxy": "orbital eccentricity",
    "inclination_proxy": "orbital inclination",
    "speed_km_s": "orbital speed",
    "accel_magnitude": "acceleration",
    "dominant_class_fraction": "behavior consistency",
    "class_entropy": "behavior diversity",
    "unique_classes_norm": "number of distinct behaviors",
    "regime_leo": "LEO regime presence",
    "regime_meo": "MEO regime presence",
    "regime_geo": "GEO regime presence",
    "regime_heo": "HEO regime presence",
}


class AnomalyExplainer:
    """Generates human-readable explanations for anomaly detections."""

    def top_features(
        self, per_feature_error: np.ndarray, n: int = 3
    ) -> List[str]:
        """
        Return the *n* feature names with the highest reconstruction error.
        """
        indices = np.argsort(per_feature_error)[::-1][:n]
        return [FEATURE_NAMES[i] for i in indices]

    def explain(
        self,
        anomaly_score: float,
        threshold: float,
        is_anomaly: bool,
        percentile: float,
        per_feature_error: np.ndarray,
        profile: BehaviorProfile,
    ) -> str:
        """
        Build a multi-sentence explanation of the anomaly result.
        """
        parts: List[str] = []

        # Summary
        status = "ANOMALOUS" if is_anomaly else "NORMAL"
        parts.append(
            f"Object {profile.object_id}: {status} "
            f"(score {anomaly_score:.4f}, threshold {threshold:.4f}, "
            f"percentile {percentile:.1f}%)."
        )

        if is_anomaly:
            # Top contributing features
            top = self.top_features(per_feature_error, n=3)
            readable = [_GROUP_DESCRIPTIONS.get(f, f) for f in top]
            parts.append(
                f"Primary anomaly drivers: {', '.join(readable)}."
            )

            # Contextual detail
            if profile.num_observations > 0:
                parts.append(
                    f"Based on {profile.num_observations} observations "
                    f"over {profile.observation_window_s / 3600:.1f} hours."
                )
        else:
            parts.append("Behavior within expected parameters.")

        return " ".join(parts)
