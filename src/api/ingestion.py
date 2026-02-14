"""
IngestionService — Accepts TLE data, propagates via SGP4, and adds to the live catalog.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _classify_type_from_name(name: str) -> str:
    """Classify object type from its TLE name."""
    upper = name.upper()
    if " DEB" in upper or upper.startswith("DEB "):
        return "DEBRIS"
    if "R/B" in upper or " RB" in upper:
        return "ROCKET_BODY"
    return "PAYLOAD"


class IngestionService:
    """Ingests TLE data into the running SpaceCatalog."""

    def __init__(self, catalog):
        self._catalog = catalog
        self._ingested_count = 0
        self._next_id: int | None = None

    def _get_next_id(self) -> int:
        """Allocate the next available object ID."""
        if self._next_id is None:
            self._next_id = int(self._catalog.object_ids.max()) + 1
        obj_id = self._next_id
        self._next_id += 1
        return obj_id

    def ingest_tle(self, name: str, tle_line1: str, tle_line2: str) -> dict:
        """Propagate a TLE and add the resulting trajectory to the catalog.

        Returns:
            Dict with object_id, name, object_type, timesteps, regime, altitude_km.
        """
        from src.simulation.tle_loader import TLE
        from src.simulation.orbital_mechanics import SGP4Propagator

        t0 = time.perf_counter()

        tle = TLE.from_lines(name, tle_line1, tle_line2)
        propagator = SGP4Propagator(tle)

        # Propagate to match catalog time axis
        catalog_timestamps = self._catalog.timestamps
        n_timesteps = len(catalog_timestamps)

        positions = np.empty((n_timesteps, 3), dtype=np.float64)
        velocities = np.empty((n_timesteps, 3), dtype=np.float64)

        for i, ts in enumerate(catalog_timestamps):
            dt = ts.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            state = propagator.propagate(dt)
            positions[i] = state.position
            velocities[i] = state.velocity

        object_type = _classify_type_from_name(name)
        object_id = self._get_next_id()

        self._catalog.add_object(
            object_id=object_id,
            name=name,
            object_type=object_type,
            positions=positions,
            velocities=velocities,
            timestamps=catalog_timestamps,
        )

        self._ingested_count += 1

        # Log ingestion event
        try:
            from src.api.database import log_ingestion
            log_ingestion(
                object_id=object_id,
                object_name=name,
                object_type=object_type,
                source="manual",
                tle_epoch=str(tle.epoch),
            )
        except Exception:
            logger.debug("Failed to log ingestion for object %d", object_id, exc_info=True)

        # Look up the computed regime and altitude
        idx = self._catalog.get_object_index(object_id)
        regime = self._catalog.regimes[idx] if idx is not None else "LEO"
        altitude_km = float(self._catalog.ref_altitudes[idx]) if idx is not None else 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Ingested TLE '%s' as object %d (%s) in %.0fms",
                     name, object_id, object_type, elapsed_ms)

        return {
            "object_id": object_id,
            "name": name,
            "object_type": object_type,
            "timesteps": n_timesteps,
            "regime": regime,
            "altitude_km": round(altitude_km, 2),
        }

    def get_status(self) -> dict:
        """Return ingestion status summary."""
        return {
            "ingested_count": self._ingested_count,
            "catalog_size": self._catalog.n_objects,
        }
