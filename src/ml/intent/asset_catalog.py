"""
High-Value Asset Catalog for Proximity-Based Threat Assessment.

Provides a simulated catalog of protected assets (e.g. ISS, GPS, comms
satellites) used by the intent classifier to determine whether a maneuver
brings an object dangerously close to an asset of interest.

Author: Space AI Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class OrbitalRegime(Enum):
    """Coarse orbital regime classification."""
    LEO = "LEO"          # < 2,000 km altitude
    MEO = "MEO"          # 2,000 - 35,786 km
    GEO = "GEO"          # ~35,786 km (geosynchronous)
    HEO = "HEO"          # Highly elliptical
    UNKNOWN = "UNKNOWN"


@dataclass
class Asset:
    """A protected high-value space asset."""
    asset_id: str
    name: str
    regime: OrbitalRegime
    position_km: tuple[float, float, float]  # ECI (x, y, z)
    velocity_km_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    altitude_km: float = 0.0
    priority: int = 1  # 1 = highest
    description: str = ""

    @property
    def position_magnitude(self) -> float:
        return math.sqrt(sum(c ** 2 for c in self.position_km))


def _default_assets() -> List[Asset]:
    """Simulated catalog of high-value space assets."""
    EARTH_RADIUS = 6371.0
    return [
        Asset(
            asset_id="ISS",
            name="International Space Station",
            regime=OrbitalRegime.LEO,
            position_km=(6771.0, 0.0, 0.0),
            velocity_km_s=(0.0, 7.66, 0.0),
            altitude_km=400.0,
            priority=1,
            description="Crewed orbital laboratory",
        ),
        Asset(
            asset_id="GPS-IIF-01",
            name="GPS IIF Satellite 01",
            regime=OrbitalRegime.MEO,
            position_km=(26560.0, 0.0, 0.0),
            velocity_km_s=(0.0, 3.87, 0.0),
            altitude_km=20200.0,
            priority=2,
            description="Navigation constellation member",
        ),
        Asset(
            asset_id="DSP-23",
            name="Defense Support Program 23",
            regime=OrbitalRegime.GEO,
            position_km=(42164.0, 0.0, 0.0),
            velocity_km_s=(0.0, 3.07, 0.0),
            altitude_km=35786.0,
            priority=1,
            description="Missile early-warning satellite",
        ),
        Asset(
            asset_id="SBIRS-GEO-1",
            name="SBIRS GEO-1",
            regime=OrbitalRegime.GEO,
            position_km=(0.0, 42164.0, 0.0),
            velocity_km_s=(-3.07, 0.0, 0.0),
            altitude_km=35786.0,
            priority=1,
            description="IR surveillance satellite",
        ),
        Asset(
            asset_id="TDRS-13",
            name="TDRS-13",
            regime=OrbitalRegime.GEO,
            position_km=(-42164.0, 0.0, 0.0),
            velocity_km_s=(0.0, -3.07, 0.0),
            altitude_km=35786.0,
            priority=2,
            description="Tracking and data relay satellite",
        ),
        Asset(
            asset_id="WGS-10",
            name="Wideband Global SATCOM 10",
            regime=OrbitalRegime.GEO,
            position_km=(0.0, -42164.0, 0.0),
            velocity_km_s=(3.07, 0.0, 0.0),
            altitude_km=35786.0,
            priority=2,
            description="Military communications satellite",
        ),
    ]


class AssetCatalog:
    """
    Queryable catalog of protected space assets.

    Provides lookups by regime, proximity, and priority.
    """

    def __init__(self, assets: Optional[List[Asset]] = None):
        self.assets = assets if assets is not None else _default_assets()

    def all(self) -> List[Asset]:
        return list(self.assets)

    def by_regime(self, regime: OrbitalRegime) -> List[Asset]:
        return [a for a in self.assets if a.regime == regime]

    def by_priority(self, max_priority: int = 1) -> List[Asset]:
        """Return assets with priority <= max_priority (lower = higher priority)."""
        return [a for a in self.assets if a.priority <= max_priority]

    def nearest(self, position_km: tuple[float, float, float]) -> Optional[Asset]:
        """Return the asset closest to *position_km* (Euclidean in ECI)."""
        if not self.assets:
            return None
        return min(self.assets, key=lambda a: _dist(a.position_km, position_km))

    def within_range(
        self, position_km: tuple[float, float, float], range_km: float
    ) -> List[Asset]:
        """Return assets within *range_km* of *position_km*."""
        return [
            a for a in self.assets if _dist(a.position_km, position_km) <= range_km
        ]


def _dist(a: tuple, b: tuple) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
