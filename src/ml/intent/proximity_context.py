"""
Proximity Context Computation for Intent Classification.

Determines how close a maneuvering object is to high-value assets and
whether its trajectory is converging or diverging.  The resulting
ProximityContext feeds directly into the intent classifier's threat
escalation logic.

Author: Space AI Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.ml.intent.asset_catalog import Asset, AssetCatalog, OrbitalRegime


EARTH_RADIUS_KM = 6371.0


@dataclass
class ProximityContext:
    """Result of a proximity analysis between an object and the asset catalog."""

    nearest_asset: Optional[Asset]
    distance_km: float              # Euclidean distance to nearest asset
    closing_rate_km_s: float        # Negative = approaching
    relative_speed_km_s: float = 0.0  # Total relative velocity magnitude
    time_to_closest_approach_s: float = float("inf")  # Estimated TCA (≥0, INF if diverging)
    is_approaching: bool = False    # True if closing_rate < 0
    is_coorbital: bool = False      # True if relative speed < threshold (co-moving)
    regime_match: bool = False      # Object in same orbital regime as asset
    object_regime: OrbitalRegime = OrbitalRegime.UNKNOWN
    assets_in_range: int = 0       # Number of assets within warning radius


def classify_regime(position_km: Tuple[float, float, float]) -> OrbitalRegime:
    """Classify orbital regime from ECI position."""
    r = math.sqrt(sum(c ** 2 for c in position_km))
    alt = r - EARTH_RADIUS_KM
    if alt < 2000:
        return OrbitalRegime.LEO
    elif alt < 35000:
        return OrbitalRegime.MEO
    elif alt < 37000:
        return OrbitalRegime.GEO
    else:
        return OrbitalRegime.HEO


def compute_proximity(
    position_km: Tuple[float, float, float],
    velocity_km_s: Tuple[float, float, float],
    catalog: Optional[AssetCatalog] = None,
    warning_radius_km: float = 500.0,
) -> ProximityContext:
    """
    Compute proximity context between an object and the asset catalog.

    Args:
        position_km:  Object ECI position (x, y, z) in km.
        velocity_km_s: Object ECI velocity (vx, vy, vz) in km/s.
        catalog:      Asset catalog (default catalog if None).
        warning_radius_km: Range threshold for counting nearby assets.

    Returns:
        ProximityContext with distance, closing rate, TCA, etc.
    """
    if catalog is None:
        catalog = AssetCatalog()

    pos = np.asarray(position_km, dtype=np.float64)
    vel = np.asarray(velocity_km_s, dtype=np.float64)
    obj_regime = classify_regime(position_km)

    nearest = catalog.nearest(position_km)
    if nearest is None:
        return ProximityContext(
            nearest_asset=None,
            distance_km=float("inf"),
            closing_rate_km_s=0.0,
            time_to_closest_approach_s=float("inf"),
            is_approaching=False,
            regime_match=False,
            object_regime=obj_regime,
            assets_in_range=0,
        )

    asset_pos = np.asarray(nearest.position_km, dtype=np.float64)
    asset_vel = np.asarray(nearest.velocity_km_s, dtype=np.float64)

    # Relative state
    rel_pos = pos - asset_pos          # position of object relative to asset
    rel_vel = vel - asset_vel          # velocity of object relative to asset

    distance = float(np.linalg.norm(rel_pos))

    # Closing rate = projection of relative velocity onto line-of-sight
    # Negative means approaching
    if distance > 0:
        los = rel_pos / distance       # unit line-of-sight vector
        closing_rate = float(np.dot(rel_vel, los))
    else:
        closing_rate = 0.0

    # Time to closest approach (linear approximation)
    rel_speed_sq = float(np.dot(rel_vel, rel_vel))
    if rel_speed_sq > 1e-12 and closing_rate < 0:
        # TCA ≈ -dot(rel_pos, rel_vel) / dot(rel_vel, rel_vel)
        tca = max(0.0, -float(np.dot(rel_pos, rel_vel)) / rel_speed_sq)
    else:
        tca = float("inf")

    regime_match = obj_regime == nearest.regime
    assets_in_range = len(catalog.within_range(position_km, warning_radius_km))

    # Total relative speed — determines if object is co-orbital (< 1 km/s)
    # or on a crossing/flyby trajectory (> 1 km/s).
    # Co-orbital thresholds: GEO ~0.01 km/s, LEO ~0.5 km/s.
    # A generous 1.0 km/s catches rendezvous approaches (slowing down).
    relative_speed = float(np.linalg.norm(rel_vel))
    co_orbital_threshold = 1.0  # km/s
    is_coorbital = relative_speed < co_orbital_threshold

    return ProximityContext(
        nearest_asset=nearest,
        distance_km=distance,
        closing_rate_km_s=closing_rate,
        relative_speed_km_s=relative_speed,
        time_to_closest_approach_s=tca,
        is_approaching=closing_rate < 0,
        is_coorbital=is_coorbital,
        regime_match=regime_match,
        object_regime=obj_regime,
        assets_in_range=assets_in_range,
    )
