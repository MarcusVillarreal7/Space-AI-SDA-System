"""
SpaceCatalog — In-memory data store for satellite positions.

Loads ground_truth.parquet at startup, organizes per-object numpy arrays,
and pre-computes geodetic coordinates (lat/lon/alt) for all 1000 objects
× 1440 timesteps using vectorized ECI-to-geodetic conversion.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# WGS84 ellipsoid parameters
_WGS84_A = 6378.137  # Semi-major axis (km)
_WGS84_F = 1 / 298.257223563
_WGS84_E2 = 2 * _WGS84_F - _WGS84_F**2


def _compute_gmst(dt: pd.Timestamp) -> float:
    """
    Compute Greenwich Mean Sidereal Time in radians.

    Uses the IAU 1982 approximation based on Julian centuries from J2000.0.
    """
    # Julian date
    jd = dt.to_julian_date()
    # Julian centuries from J2000.0
    T = (jd - 2451545.0) / 36525.0
    # GMST in degrees (IAU 1982)
    gmst_deg = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + \
               0.000387933 * T**2 - T**3 / 38710000.0
    return np.radians(gmst_deg % 360.0)


def _classify_regime(alt_km: float) -> str:
    """Classify orbital regime by altitude."""
    if alt_km < 2000:
        return "LEO"
    elif alt_km < 35000:
        return "MEO"
    elif alt_km < 36500:
        return "GEO"
    else:
        return "HEO"


def _vectorized_eci_to_geodetic(
    positions: np.ndarray, gmst_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized ECI to geodetic conversion for all objects at all timesteps.

    Args:
        positions: (N_objects, N_timesteps, 3) ECI positions in km
        gmst_values: (N_timesteps,) GMST in radians

    Returns:
        lat, lon, alt arrays each (N_objects, N_timesteps) in degrees, degrees, km
    """
    n_objects, n_timesteps, _ = positions.shape

    # Broadcast GMST: (1, T) for rotation
    cos_g = np.cos(gmst_values)[np.newaxis, :]  # (1, T)
    sin_g = np.sin(gmst_values)[np.newaxis, :]  # (1, T)

    # ECI to ECEF rotation (vectorized)
    x_eci = positions[:, :, 0]  # (N, T)
    y_eci = positions[:, :, 1]
    z_eci = positions[:, :, 2]

    x_ecef = x_eci * cos_g + y_eci * sin_g
    y_ecef = -x_eci * sin_g + y_eci * cos_g
    z_ecef = z_eci

    # ECEF to geodetic (iterative, vectorized)
    lon = np.arctan2(y_ecef, x_ecef)
    p = np.sqrt(x_ecef**2 + y_ecef**2)
    lat = np.arctan2(z_ecef, p * (1 - _WGS84_E2))

    for _ in range(5):
        N = _WGS84_A / np.sqrt(1 - _WGS84_E2 * np.sin(lat)**2)
        lat = np.arctan2(z_ecef + _WGS84_E2 * N * np.sin(lat), p)

    N = _WGS84_A / np.sqrt(1 - _WGS84_E2 * np.sin(lat)**2)
    # Handle polar singularity
    cos_lat = np.cos(lat)
    alt = np.where(
        np.abs(cos_lat) > 1e-10,
        p / cos_lat - N,
        np.abs(z_ecef) - N * (1 - _WGS84_E2),
    )

    return np.degrees(lat), np.degrees(lon), alt


class SpaceCatalog:
    """
    In-memory catalog of all tracked space objects.

    Holds positions, velocities, and pre-computed geodetic coordinates
    for fast lookup by timestep index.
    """

    def __init__(self):
        self.n_objects: int = 0
        self.n_timesteps: int = 0
        self.object_ids: np.ndarray = np.array([], dtype=int)
        self.object_names: list[str] = []
        self.object_types: list[str] = []
        self.regimes: list[str] = []

        # (N_objects, N_timesteps, 3)
        self.positions: Optional[np.ndarray] = None
        self.velocities: Optional[np.ndarray] = None

        # (N_objects, N_timesteps) — pre-computed geodetic
        self.latitudes: Optional[np.ndarray] = None
        self.longitudes: Optional[np.ndarray] = None
        self.altitudes: Optional[np.ndarray] = None

        # Per-object scalars at reference timestep
        self.ref_altitudes: np.ndarray = np.array([])
        self.ref_speeds: np.ndarray = np.array([])

        # Time axis
        self.timestamps: list[pd.Timestamp] = []
        self.time_isos: list[str] = []

        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, parquet_path: str | Path) -> None:
        """Load ground truth parquet and pre-compute all derived data."""
        t0 = time.perf_counter()
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Data file not found: {parquet_path}")

        logger.info("Loading ground truth from %s", parquet_path)
        df = pd.read_parquet(parquet_path)

        # Sort by object_id and time for consistent ordering
        df = df.sort_values(["object_id", "time"]).reset_index(drop=True)

        # Extract unique objects
        unique_objects = df.groupby("object_id").first().reset_index()
        self.object_ids = unique_objects["object_id"].values.astype(int)
        self.object_names = unique_objects["object_name"].tolist()
        self.n_objects = len(self.object_ids)

        # Time axis from first object
        first_obj = df[df["object_id"] == self.object_ids[0]]
        self.timestamps = first_obj["time"].tolist()
        self.time_isos = [t.isoformat() for t in self.timestamps]
        self.n_timesteps = len(self.timestamps)

        logger.info("Catalog: %d objects × %d timesteps", self.n_objects, self.n_timesteps)

        # Reshape into (N_objects, N_timesteps, 3) arrays
        pos_cols = ["position_x", "position_y", "position_z"]
        vel_cols = ["velocity_x", "velocity_y", "velocity_z"]

        self.positions = df[pos_cols].values.reshape(self.n_objects, self.n_timesteps, 3)
        self.velocities = df[vel_cols].values.reshape(self.n_objects, self.n_timesteps, 3)

        # Reference altitude/speed from midpoint
        mid = self.n_timesteps // 2
        self.ref_altitudes = df["altitude_km"].values.reshape(
            self.n_objects, self.n_timesteps
        )[:, mid].copy()
        self.ref_speeds = df["speed_km_s"].values.reshape(
            self.n_objects, self.n_timesteps
        )[:, mid].copy()

        # Classify regimes
        self.regimes = [_classify_regime(alt) for alt in self.ref_altitudes]

        # Object types (from parquet column if present, else default to PAYLOAD)
        if "object_type" in df.columns:
            type_series = df.groupby("object_id")["object_type"].first()
            self.object_types = [
                type_series.get(oid, "PAYLOAD") for oid in self.object_ids
            ]
            logger.info("Object types loaded from parquet: %s",
                        {t: self.object_types.count(t) for t in set(self.object_types)})
        else:
            self.object_types = ["PAYLOAD"] * self.n_objects

        # Pre-compute geodetic coordinates
        logger.info("Pre-computing geodetic coordinates (vectorized)...")
        t_geo = time.perf_counter()
        gmst_values = np.array([_compute_gmst(t) for t in self.timestamps])
        self.latitudes, self.longitudes, self.altitudes = _vectorized_eci_to_geodetic(
            self.positions, gmst_values
        )
        logger.info("Geodetic conversion done in %.2fs", time.perf_counter() - t_geo)

        self._loaded = True
        elapsed = time.perf_counter() - t0
        logger.info("SpaceCatalog loaded in %.2fs (%.1f MB positions)",
                     elapsed, self.positions.nbytes / 1e6)

    def add_object(
        self,
        object_id: int,
        name: str,
        object_type: str,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: list[pd.Timestamp],
    ) -> None:
        """Add a single object to the live catalog.

        Args:
            object_id: Unique ID for the new object.
            name: Display name.
            object_type: PAYLOAD, DEBRIS, or ROCKET_BODY.
            positions: (N_timesteps, 3) ECI positions in km.
            velocities: (N_timesteps, 3) ECI velocities in km/s.
            timestamps: Must match self.timestamps in length.
        """
        if len(timestamps) != self.n_timesteps:
            raise ValueError(
                f"Timestep count mismatch: got {len(timestamps)}, expected {self.n_timesteps}"
            )

        # Expand ID array
        self.object_ids = np.append(self.object_ids, object_id)
        self.object_names.append(name)
        self.object_types.append(object_type)

        # Reshape to (1, T, 3) for concatenation
        pos_3d = positions[np.newaxis, :, :]  # (1, T, 3)
        vel_3d = velocities[np.newaxis, :, :]

        self.positions = np.concatenate([self.positions, pos_3d], axis=0)
        self.velocities = np.concatenate([self.velocities, vel_3d], axis=0)

        # Compute geodetic coords for the new object
        gmst_values = np.array([_compute_gmst(t) for t in self.timestamps])
        lat, lon, alt = _vectorized_eci_to_geodetic(pos_3d, gmst_values)
        self.latitudes = np.concatenate([self.latitudes, lat], axis=0)
        self.longitudes = np.concatenate([self.longitudes, lon], axis=0)
        self.altitudes = np.concatenate([self.altitudes, alt], axis=0)

        # Reference values at midpoint
        mid = self.n_timesteps // 2
        ref_alt = float(alt[0, mid])
        ref_speed = float(np.linalg.norm(velocities[mid]))
        self.ref_altitudes = np.append(self.ref_altitudes, ref_alt)
        self.ref_speeds = np.append(self.ref_speeds, ref_speed)

        self.regimes.append(_classify_regime(ref_alt))
        self.n_objects += 1

        logger.info("Added object %d (%s) — catalog now has %d objects",
                     object_id, name, self.n_objects)

    def get_all_positions_at_timestep(self, timestep: int) -> list[dict]:
        """Get lat/lon/alt for all objects at a given timestep index."""
        if not self._loaded:
            return []
        ts = max(0, min(timestep, self.n_timesteps - 1))
        return [
            {
                "id": int(self.object_ids[i]),
                "name": self.object_names[i],
                "object_type": self.object_types[i] if self.object_types else "PAYLOAD",
                "lat": float(self.latitudes[i, ts]),
                "lon": float(self.longitudes[i, ts]),
                "alt_km": float(self.altitudes[i, ts]),
            }
            for i in range(self.n_objects)
        ]

    def get_object_index(self, object_id: int) -> Optional[int]:
        """Get array index for a given object_id."""
        matches = np.where(self.object_ids == object_id)[0]
        return int(matches[0]) if len(matches) > 0 else None

    def get_object_trajectory(
        self, object_id: int, start: int = 0, end: Optional[int] = None
    ) -> list[dict]:
        """Get full or partial trajectory for a single object."""
        idx = self.get_object_index(object_id)
        if idx is None:
            return []
        end = end or self.n_timesteps
        start = max(0, start)
        end = min(end, self.n_timesteps)

        trajectory = []
        for ts in range(start, end):
            trajectory.append({
                "timestep": ts,
                "time_iso": self.time_isos[ts],
                "lat": float(self.latitudes[idx, ts]),
                "lon": float(self.longitudes[idx, ts]),
                "alt_km": float(self.altitudes[idx, ts]),
                "position_x": float(self.positions[idx, ts, 0]),
                "position_y": float(self.positions[idx, ts, 1]),
                "position_z": float(self.positions[idx, ts, 2]),
                "velocity_x": float(self.velocities[idx, ts, 0]),
                "velocity_y": float(self.velocities[idx, ts, 1]),
                "velocity_z": float(self.velocities[idx, ts, 2]),
            })
        return trajectory

    def get_object_summary(self, object_id: int) -> Optional[dict]:
        """Get summary info for a single object."""
        idx = self.get_object_index(object_id)
        if idx is None:
            return None
        return {
            "id": int(self.object_ids[idx]),
            "name": self.object_names[idx],
            "object_type": self.object_types[idx] if self.object_types else "PAYLOAD",
            "regime": self.regimes[idx],
            "altitude_km": float(self.ref_altitudes[idx]),
            "speed_km_s": float(self.ref_speeds[idx]),
        }

    def get_all_summaries(self) -> list[dict]:
        """Get summary info for all objects."""
        return [
            {
                "id": int(self.object_ids[i]),
                "name": self.object_names[i],
                "object_type": self.object_types[i] if self.object_types else "PAYLOAD",
                "regime": self.regimes[i],
                "altitude_km": float(self.ref_altitudes[i]),
                "speed_km_s": float(self.ref_speeds[i]),
            }
            for i in range(self.n_objects)
        ]

    def get_positions_and_velocities(
        self, object_id: int, start: int = 0, end: Optional[int] = None
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get raw position/velocity arrays for threat assessment.

        Returns:
            (positions, velocities, timestamps_seconds) or None
        """
        idx = self.get_object_index(object_id)
        if idx is None:
            return None
        end = end or self.n_timesteps
        positions = self.positions[idx, start:end]  # (T, 3)
        velocities = self.velocities[idx, start:end]  # (T, 3)
        # Timestamps as seconds from first observation
        ts = np.array([
            (self.timestamps[i] - self.timestamps[start]).total_seconds()
            for i in range(start, end)
        ])
        return positions, velocities, ts
