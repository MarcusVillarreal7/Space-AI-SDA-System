"""
ThreatService — Wraps ThreatAssessmentPipeline for dashboard use.

Provides on-demand assessment, tier summary, rotating batch assessment,
trajectory prediction, and full assess-all with progress tracking.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.api.database import cache_assessment, get_cached_assessment, log_prediction, store_alert

logger = logging.getLogger(__name__)

# WGS84 ellipsoid parameters (for ECI → geodetic conversion)
_WGS84_A = 6378.137
_WGS84_F = 1 / 298.257223563
_WGS84_E2 = 2 * _WGS84_F - _WGS84_F ** 2


def _eci_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert single ECI position (km) to lat/lon/alt (deg, deg, km)."""
    # Approximate: assume GMST=0 (static snapshot)
    lon = math.degrees(math.atan2(y, x))
    p = math.sqrt(x ** 2 + y ** 2)
    lat = math.atan2(z, p * (1 - _WGS84_E2))
    for _ in range(5):
        N = _WGS84_A / math.sqrt(1 - _WGS84_E2 * math.sin(lat) ** 2)
        lat = math.atan2(z + _WGS84_E2 * N * math.sin(lat), p)
    N = _WGS84_A / math.sqrt(1 - _WGS84_E2 * math.sin(lat) ** 2)
    cos_lat = math.cos(lat)
    if abs(cos_lat) > 1e-10:
        alt = p / cos_lat - N
    else:
        alt = abs(z) - N * (1 - _WGS84_E2)
    return math.degrees(lat), lon, alt


class ThreatService:
    """
    High-level threat assessment service for the dashboard.

    Wraps ThreatAssessmentPipeline with caching, alert generation,
    and batch assessment for the alert feed.
    """

    def __init__(self):
        self._pipeline = None
        self._pipeline_loaded = False
        self._trajectory_predictor = None
        self._trajectory_loaded = False
        self._tier_counts: dict[str, int] = {
            "MINIMAL": 0, "LOW": 0, "MODERATE": 0,
            "ELEVATED": 0, "CRITICAL": 0,
        }
        self._assessments: dict[int, dict] = {}  # object_id -> latest assessment
        self._batch_index: int = 0  # For rotating batch assessment
        # Assess-all progress
        self._assess_all_running = False
        self._assess_all_completed = 0
        self._assess_all_total = 0

    def warm_load(self, assessments: dict[int, dict]) -> None:
        """Populate in-memory state from pre-computed assessments (snapshot or DB cache).

        Called on startup so the globe and WebSocket broadcasts show correct threat
        tiers immediately without needing to run the ML pipeline.
        """
        for object_id, result in assessments.items():
            self._assessments[object_id] = result
            tier = result.get("threat_tier", "MINIMAL")
            if tier in self._tier_counts:
                self._tier_counts[tier] += 1
        self._assess_all_completed = len(assessments)
        self._assess_all_total = len(assessments)
        logger.info("Warm-loaded %d pre-computed assessments into ThreatService", len(assessments))

    def _load_pipeline(self) -> None:
        """Lazy-load the threat assessment pipeline."""
        if self._pipeline_loaded:
            return
        try:
            from src.ml.threat_assessment import ThreatAssessmentPipeline
            # CNN_LSTM_ENABLED=false skips the maneuver classifier on CPU-constrained
            # deployments (e.g. Railway). Proximity, intent, and anomaly scoring still run.
            cnn_lstm_enabled = os.environ.get("CNN_LSTM_ENABLED", "true").lower() != "false"
            self._pipeline = ThreatAssessmentPipeline(
                anomaly_checkpoint="checkpoints/phase3_anomaly",
                maneuver_checkpoint=(
                    "checkpoints/phase3_day4/maneuver_classifier.pt"
                    if cnn_lstm_enabled else None
                ),
                device="cpu",
            )
            self._pipeline_loaded = True
            if cnn_lstm_enabled:
                logger.info("ThreatAssessmentPipeline loaded (full — CNN-LSTM enabled)")
            else:
                logger.info("ThreatAssessmentPipeline loaded (fast — CNN-LSTM disabled via CNN_LSTM_ENABLED=false)")
        except Exception:
            logger.exception("Failed to load ThreatAssessmentPipeline — using defaults")
            self._pipeline = None
            self._pipeline_loaded = True

    def assess_object(
        self,
        object_id: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray,
        timestep: int = 0,
        use_cache: bool = True,
        full_positions: Optional[np.ndarray] = None,
        full_velocities: Optional[np.ndarray] = None,
        full_timestamps: Optional[np.ndarray] = None,
        object_type: str = "PAYLOAD",
    ) -> dict:
        """
        Assess a single object, with optional caching.

        Returns a dict matching ThreatAssessmentResponse schema.
        """
        # Check in-memory cache first (populated by warm_load from snapshot/DB)
        if use_cache and object_id in self._assessments:
            return self._assessments[object_id]

        # Check database cache
        if use_cache:
            cached = get_cached_assessment(object_id, timestep)
            if cached:
                return cached

        self._load_pipeline()

        if self._pipeline is None:
            return self._default_assessment(object_id)

        t0 = time.perf_counter()
        logger.info(
            "ASSESS object_id=%d | %d observations, %.0fs window",
            object_id, len(timestamps),
            float(timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0,
        )

        assessment = self._pipeline.assess_by_type(
            object_id=str(object_id),
            positions=positions,
            velocities=velocities,
            timestamps=timestamps,
            object_type=object_type,
            full_positions=full_positions,
            full_velocities=full_velocities,
            full_timestamps=full_timestamps,
        )

        result = {
            "object_id": object_id,
            "threat_score": round(assessment.threat_score.score, 2),
            "threat_tier": assessment.threat_score.tier.value,
            "intent_score": round(assessment.threat_score.intent_score, 2),
            "anomaly_score": round(assessment.threat_score.anomaly_score, 2),
            "proximity_score": round(assessment.threat_score.proximity_score, 2),
            "pattern_score": round(assessment.threat_score.pattern_score, 2),
            "maneuver_class": assessment.maneuver_name,
            "maneuver_confidence": round(assessment.maneuver_confidence, 3),
            "maneuver_probabilities": assessment.maneuver_probabilities,
            "contributing_factors": assessment.threat_score.contributing_factors,
            "explanation": assessment.threat_score.explanation,
            "latency_ms": round(assessment.latency_ms, 2),
        }

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "RESULT object_id=%d | tier=%s score=%.1f | maneuver=%s (%.0f%%) | "
            "intent=%.1f anomaly=%.1f proximity=%.1f pattern=%.1f | %.1fms",
            object_id, result["threat_tier"], result["threat_score"],
            result["maneuver_class"], result["maneuver_confidence"] * 100,
            result["intent_score"], result["anomaly_score"],
            result["proximity_score"], result["pattern_score"],
            elapsed_ms,
        )

        # Cache the result
        cache_assessment(object_id, timestep, result)

        # Log prediction for monitoring/audit
        try:
            # Resolve object name from catalog
            from src.api.main import app_state
            catalog = app_state.get("catalog")
            idx = catalog.get_object_index(object_id) if catalog else None
            obj_name = catalog.object_names[idx] if (catalog and idx is not None) else f"Object-{object_id}"
            obj_type_str = (catalog.object_types[idx]
                            if (catalog and idx is not None and catalog.object_types)
                            else object_type)
            log_prediction(
                object_id=object_id,
                object_name=obj_name,
                object_type=obj_type_str,
                threat_tier=result["threat_tier"],
                threat_score=result["threat_score"],
                maneuver_class=result.get("maneuver_class"),
                anomaly_score=result.get("anomaly_score"),
                proximity_score=result.get("proximity_score"),
                latency_ms=result.get("latency_ms", elapsed_ms),
            )
        except Exception:
            logger.debug("Failed to log prediction for object %d", object_id, exc_info=True)

        # Store in memory
        self._assessments[object_id] = result

        # Sync tier to app_state so WebSocket broadcasts and /api/objects
        # reflect the assessed tier (not just the default MINIMAL)
        self._sync_tier(object_id, result["threat_tier"])

        return result

    def batch_assess(
        self,
        catalog,
        timestep: int,
        batch_size: int = 50,
    ) -> list[dict]:
        """
        Assess a rotating batch of objects (for alert feed).

        Cycles through all objects over multiple calls.
        """
        results = []
        object_ids = catalog.object_ids

        start = self._batch_index
        end = min(start + batch_size, len(object_ids))
        batch_ids = object_ids[start:end]

        # Wrap around
        if end >= len(object_ids):
            self._batch_index = 0
        else:
            self._batch_index = end

        for oid in batch_ids:
            oid = int(oid)
            data = catalog.get_positions_and_velocities(oid)
            if data is None:
                continue
            positions, velocities, ts = data

            # Look up object type
            idx = catalog.get_object_index(oid)
            obj_type = (catalog.object_types[idx]
                        if hasattr(catalog, "object_types") and catalog.object_types and idx is not None
                        else "PAYLOAD")

            # Use a window of recent observations for classification,
            # but pass the full trajectory for proximity scanning.
            window = min(20, len(ts))
            result = self.assess_object(
                oid,
                positions[-window:],
                velocities[-window:],
                ts[-window:],
                timestep=timestep,
                full_positions=positions,
                full_velocities=velocities,
                full_timestamps=ts,
                object_type=obj_type,
            )
            results.append(result)

            # Generate alert for elevated/critical
            tier = result["threat_tier"]
            if tier in ("ELEVATED", "CRITICAL"):
                obj_name = catalog.object_names[catalog.get_object_index(oid)]
                store_alert(
                    object_id=oid,
                    object_name=obj_name,
                    threat_tier=tier,
                    threat_score=result["threat_score"],
                    message=f"{obj_name} assessed as {tier} (score: {result['threat_score']})",
                )

        return results

    def get_tier_summary(self) -> dict[str, int]:
        """Get count of objects by threat tier from cached assessments."""
        counts = {"MINIMAL": 0, "LOW": 0, "MODERATE": 0, "ELEVATED": 0, "CRITICAL": 0}
        for a in self._assessments.values():
            tier = a.get("threat_tier", "MINIMAL")
            counts[tier] = counts.get(tier, 0) + 1

        # Count unassessed objects as MINIMAL
        from src.api.main import app_state
        catalog = app_state.get("catalog")
        if catalog:
            assessed = len(self._assessments)
            counts["MINIMAL"] += catalog.n_objects - assessed

        return counts

    def get_threat_tiers(self) -> dict[int, str]:
        """Get a mapping of object_id -> threat_tier for all assessed objects."""
        return {
            oid: a.get("threat_tier", "MINIMAL")
            for oid, a in self._assessments.items()
        }

    def _load_trajectory_predictor(self) -> None:
        """Lazy-load the TrajectoryTransformer predictor."""
        if self._trajectory_loaded:
            return
        self._trajectory_loaded = True
        ckpt = Path("checkpoints/phase3_parallel/best_model.pt")
        if not ckpt.exists():
            logger.warning("Trajectory checkpoint not found: %s", ckpt)
            return
        try:
            from src.ml.inference import TrajectoryPredictor
            self._trajectory_predictor = TrajectoryPredictor(
                str(ckpt), device="cpu"
            )
            logger.info("TrajectoryPredictor loaded")
        except Exception:
            logger.exception("Failed to load TrajectoryPredictor")

    def predict_trajectory(self, object_id: int, catalog) -> Optional[dict]:
        """
        Predict 30-step future trajectory for an object.

        Returns dict with 'points' list of {step, lat, lon, alt_km, position_x/y/z}.
        """
        self._load_trajectory_predictor()
        if self._trajectory_predictor is None:
            return None

        data = catalog.get_positions_and_velocities(object_id)
        if data is None:
            return None

        positions, velocities, timestamps = data
        window = min(20, len(timestamps))

        t0 = time.perf_counter()
        pred = self._trajectory_predictor.predict(
            positions[-window:], velocities[-window:], timestamps[-window:],
            pred_horizon=30,
        )
        latency = (time.perf_counter() - t0) * 1000.0

        points = []
        for i in range(pred["positions"].shape[0]):
            px, py, pz = float(pred["positions"][i, 0]), float(pred["positions"][i, 1]), float(pred["positions"][i, 2])
            lat, lon, alt = _eci_to_geodetic(px, py, pz)
            points.append({
                "step": i,
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "alt_km": round(alt, 2),
                "position_x": round(px, 3),
                "position_y": round(py, 3),
                "position_z": round(pz, 3),
            })

        idx = catalog.get_object_index(object_id)
        name = catalog.object_names[idx] if idx is not None else f"Object-{object_id}"

        return {
            "object_id": object_id,
            "object_name": name,
            "points": points,
            "model": "TrajectoryTransformer",
            "latency_ms": round(latency, 2),
        }

    async def assess_all(self, catalog, timestep: int) -> None:
        """
        Assess all objects in background batches of 50.
        Updates progress state and syncs tiers live.
        """
        if self._assess_all_running:
            return
        self._assess_all_running = True
        self._assess_all_completed = 0
        self._assess_all_total = catalog.n_objects

        try:
            object_ids = catalog.object_ids
            batch_size = 50
            for start in range(0, len(object_ids), batch_size):
                batch = object_ids[start:start + batch_size]
                for oid in batch:
                    oid = int(oid)
                    data = catalog.get_positions_and_velocities(oid)
                    if data is None:
                        self._assess_all_completed += 1
                        continue
                    positions, velocities, ts = data
                    window = min(20, len(ts))

                    # Look up object type
                    idx = catalog.get_object_index(oid)
                    obj_type = (catalog.object_types[idx]
                                if hasattr(catalog, "object_types") and catalog.object_types and idx is not None
                                else "PAYLOAD")

                    result = self.assess_object(
                        oid,
                        positions[-window:],
                        velocities[-window:],
                        ts[-window:],
                        timestep=timestep,
                        use_cache=False,
                        full_positions=positions,
                        full_velocities=velocities,
                        full_timestamps=ts,
                        object_type=obj_type,
                    )
                    self._assess_all_completed += 1

                    # Generate alert for elevated/critical
                    tier = result["threat_tier"]
                    if tier in ("ELEVATED", "CRITICAL"):
                        idx = catalog.get_object_index(oid)
                        obj_name = catalog.object_names[idx] if idx is not None else f"Object-{oid}"
                        explanation = result.get("explanation", "")
                        msg = f"{obj_name}: {explanation}" if explanation else f"{obj_name} assessed as {tier}"
                        store_alert(
                            object_id=oid,
                            object_name=obj_name,
                            threat_tier=tier,
                            threat_score=result["threat_score"],
                            message=msg,
                        )

                    # Yield to event loop after every object so health checks
                    # and WebSocket broadcasts are never starved during assess-all
                    await asyncio.sleep(0)
        finally:
            self._assess_all_running = False
            logger.info("Assess-all complete: %d/%d objects",
                        self._assess_all_completed, self._assess_all_total)

    def get_assess_all_status(self) -> dict:
        """Get progress of assess-all operation."""
        return {
            "running": self._assess_all_running,
            "completed": self._assess_all_completed,
            "total": self._assess_all_total,
        }

    @staticmethod
    def _sync_tier(object_id: int, tier: str) -> None:
        """Push assessed tier into app_state so broadcasts and REST endpoints see it."""
        try:
            from src.api.main import app_state
            threat_tiers = app_state.get("threat_tiers")
            if threat_tiers is not None:
                threat_tiers[object_id] = tier
        except Exception:
            pass  # Don't break assessment if app_state isn't ready

    @staticmethod
    def _default_assessment(object_id: int) -> dict:
        return {
            "object_id": object_id,
            "threat_score": 0.0,
            "threat_tier": "MINIMAL",
            "intent_score": 0.0,
            "anomaly_score": 0.0,
            "proximity_score": 0.0,
            "pattern_score": 0.0,
            "maneuver_class": "Normal",
            "maneuver_confidence": 0.5,
            "maneuver_probabilities": None,
            "contributing_factors": [],
            "explanation": "No assessment pipeline available",
            "latency_ms": 0.0,
        }
