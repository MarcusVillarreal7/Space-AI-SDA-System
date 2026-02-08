"""
Behavioral Feature Extraction for Anomaly Detection.

Transforms raw satellite track/maneuver data into a fixed-size 19D feature
vector that captures behavioral patterns.  The vector is consumed by the
anomaly autoencoder for reconstruction-based anomaly detection.

Feature groups (19D total):
    Maneuver statistics     (5D): count, rate, mean_dv, max_dv, dv_variance
    Timing patterns         (2D): mean_interval, interval_regularity
    Orbital characteristics (5D): altitude, eccentricity, inclination, speed, accel_mag
    Classification history  (3D): dominant_class_frac, class_entropy, unique_classes_norm
    Regime one-hot          (4D): LEO, MEO, GEO, HEO

Author: Space AI Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.ml.intent.asset_catalog import OrbitalRegime
from src.ml.intent.proximity_context import classify_regime


FEATURE_DIM = 19

FEATURE_NAMES: List[str] = [
    # Maneuver statistics (5)
    "maneuver_count",
    "maneuver_rate",
    "mean_delta_v",
    "max_delta_v",
    "delta_v_variance",
    # Timing patterns (2)
    "mean_interval_s",
    "interval_regularity",
    # Orbital characteristics (5)
    "altitude_km",
    "eccentricity_proxy",
    "inclination_proxy",
    "speed_km_s",
    "accel_magnitude",
    # Classification history (3)
    "dominant_class_fraction",
    "class_entropy",
    "unique_classes_norm",
    # Regime one-hot (4)
    "regime_leo",
    "regime_meo",
    "regime_geo",
    "regime_heo",
]

assert len(FEATURE_NAMES) == FEATURE_DIM


@dataclass
class ManeuverRecord:
    """Lightweight maneuver observation for feature extraction."""
    timestamp: float
    maneuver_class: int          # 0-5
    delta_v_magnitude: float     # km/s


@dataclass
class BehaviorProfile:
    """
    A 19D behavioral feature vector for a single satellite over an
    observation window, plus metadata.
    """
    object_id: str
    features: np.ndarray                         # shape (19,)
    observation_window_s: float = 0.0            # duration of observation
    num_observations: int = 0
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))


class BehaviorFeatureExtractor:
    """
    Extracts a 19D behavioral feature vector from raw satellite observations.

    Usage::

        extractor = BehaviorFeatureExtractor()
        profile = extractor.extract(
            object_id="SAT-42",
            maneuvers=maneuver_list,
            position_km=(6800.0, 0.0, 0.0),
            velocity_km_s=(0.0, 7.5, 0.0),
        )
    """

    EARTH_RADIUS_KM = 6371.0

    def extract(
        self,
        object_id: str,
        maneuvers: List[ManeuverRecord],
        position_km: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        velocity_km_s: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        acceleration_km_s2: Optional[Tuple[float, float, float]] = None,
    ) -> BehaviorProfile:
        """
        Build a BehaviorProfile from maneuver history and current state.

        Args:
            object_id: Satellite identifier.
            maneuvers: Chronological list of ManeuverRecord.
            position_km: Current ECI position.
            velocity_km_s: Current ECI velocity.
            acceleration_km_s2: Current acceleration (optional, defaults to zero).

        Returns:
            BehaviorProfile with 19D feature vector.
        """
        feat = np.zeros(FEATURE_DIM, dtype=np.float32)

        # --- Maneuver statistics (5D) ---
        maneuver_stats = self._maneuver_statistics(maneuvers)
        feat[0:5] = maneuver_stats

        # --- Timing patterns (2D) ---
        timing = self._timing_patterns(maneuvers)
        feat[5:7] = timing

        # --- Orbital characteristics (5D) ---
        orbital = self._orbital_characteristics(
            position_km, velocity_km_s, acceleration_km_s2
        )
        feat[7:12] = orbital

        # --- Classification history (3D) ---
        cls_hist = self._classification_history(maneuvers)
        feat[12:15] = cls_hist

        # --- Regime one-hot (4D) ---
        regime_vec = self._regime_encoding(position_km)
        feat[15:19] = regime_vec

        # Observation window
        if len(maneuvers) >= 2:
            window = maneuvers[-1].timestamp - maneuvers[0].timestamp
        else:
            window = 0.0

        return BehaviorProfile(
            object_id=object_id,
            features=feat,
            observation_window_s=window,
            num_observations=len(maneuvers),
        )

    # ------------------------------------------------------------------
    # Feature group extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _maneuver_statistics(maneuvers: List[ManeuverRecord]) -> np.ndarray:
        """5D: count, rate, mean_dv, max_dv, dv_variance."""
        active = [m for m in maneuvers if m.maneuver_class in {3, 4, 5}]
        count = len(active)

        if len(maneuvers) >= 2:
            window = maneuvers[-1].timestamp - maneuvers[0].timestamp
            rate = count / max(window, 1.0) * 86400.0  # maneuvers per day
        else:
            rate = 0.0

        dvs = np.array([m.delta_v_magnitude for m in active], dtype=np.float32)
        if len(dvs) > 0:
            mean_dv = float(dvs.mean())
            max_dv = float(dvs.max())
            dv_var = float(dvs.var())
        else:
            mean_dv = max_dv = dv_var = 0.0

        return np.array([count, rate, mean_dv, max_dv, dv_var], dtype=np.float32)

    @staticmethod
    def _timing_patterns(maneuvers: List[ManeuverRecord]) -> np.ndarray:
        """2D: mean_interval, interval_regularity (1 = perfectly regular)."""
        active = [m for m in maneuvers if m.maneuver_class in {3, 4, 5}]
        if len(active) < 2:
            return np.zeros(2, dtype=np.float32)

        intervals = np.diff([m.timestamp for m in active]).astype(np.float32)
        mean_int = float(intervals.mean())
        if mean_int > 0:
            regularity = 1.0 - float(intervals.std()) / mean_int
            regularity = max(0.0, min(1.0, regularity))
        else:
            regularity = 0.0

        return np.array([mean_int, regularity], dtype=np.float32)

    def _orbital_characteristics(
        self,
        position_km: Tuple[float, float, float],
        velocity_km_s: Tuple[float, float, float],
        acceleration_km_s2: Optional[Tuple[float, float, float]],
    ) -> np.ndarray:
        """5D: altitude, eccentricity_proxy, inclination_proxy, speed, accel_mag."""
        pos = np.asarray(position_km, dtype=np.float64)
        vel = np.asarray(velocity_km_s, dtype=np.float64)

        r = float(np.linalg.norm(pos))
        altitude = r - self.EARTH_RADIUS_KM

        speed = float(np.linalg.norm(vel))

        # Eccentricity proxy: ratio of radial to total velocity
        if r > 0 and speed > 0:
            radial_vel = float(np.dot(pos, vel)) / r
            ecc_proxy = abs(radial_vel) / speed
        else:
            ecc_proxy = 0.0

        # Inclination proxy: angle of position vector from equatorial plane
        if r > 0:
            incl_proxy = abs(float(pos[2])) / r  # sin(inclination) ≈ z/r
        else:
            incl_proxy = 0.0

        if acceleration_km_s2 is not None:
            accel = float(np.linalg.norm(acceleration_km_s2))
        else:
            accel = 0.0

        return np.array(
            [altitude, ecc_proxy, incl_proxy, speed, accel], dtype=np.float32
        )

    @staticmethod
    def _classification_history(maneuvers: List[ManeuverRecord]) -> np.ndarray:
        """3D: dominant_class_fraction, class_entropy, unique_classes_norm."""
        if not maneuvers:
            return np.zeros(3, dtype=np.float32)

        classes = [m.maneuver_class for m in maneuvers]
        n = len(classes)

        # Dominant class fraction
        from collections import Counter
        counts = Counter(classes)
        dominant_frac = counts.most_common(1)[0][1] / n

        # Shannon entropy (normalized to [0, 1])
        probs = np.array(list(counts.values()), dtype=np.float32) / n
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        max_entropy = math.log(6)  # 6 classes
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Unique classes normalized
        unique_norm = len(counts) / 6.0

        return np.array(
            [dominant_frac, norm_entropy, unique_norm], dtype=np.float32
        )

    @staticmethod
    def _regime_encoding(
        position_km: Tuple[float, float, float],
    ) -> np.ndarray:
        """4D one-hot: LEO, MEO, GEO, HEO."""
        regime = classify_regime(position_km)
        mapping = {
            OrbitalRegime.LEO: 0,
            OrbitalRegime.MEO: 1,
            OrbitalRegime.GEO: 2,
            OrbitalRegime.HEO: 3,
        }
        vec = np.zeros(4, dtype=np.float32)
        idx = mapping.get(regime)
        if idx is not None:
            vec[idx] = 1.0
        return vec
