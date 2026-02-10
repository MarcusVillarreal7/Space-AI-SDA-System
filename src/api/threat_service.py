"""
ThreatService — Wraps ThreatAssessmentPipeline for dashboard use.

Provides on-demand assessment, tier summary, and rotating batch assessment.
Caches results to avoid redundant computation.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from src.api.database import cache_assessment, get_cached_assessment, store_alert

logger = logging.getLogger(__name__)


class ThreatService:
    """
    High-level threat assessment service for the dashboard.

    Wraps ThreatAssessmentPipeline with caching, alert generation,
    and batch assessment for the alert feed.
    """

    def __init__(self):
        self._pipeline = None
        self._pipeline_loaded = False
        self._tier_counts: dict[str, int] = {
            "MINIMAL": 0, "LOW": 0, "MODERATE": 0,
            "ELEVATED": 0, "CRITICAL": 0,
        }
        self._assessments: dict[int, dict] = {}  # object_id -> latest assessment
        self._batch_index: int = 0  # For rotating batch assessment

    def _load_pipeline(self) -> None:
        """Lazy-load the threat assessment pipeline."""
        if self._pipeline_loaded:
            return
        try:
            from src.ml.threat_assessment import ThreatAssessmentPipeline
            self._pipeline = ThreatAssessmentPipeline(
                anomaly_checkpoint="checkpoints/phase3_anomaly",
                device="cpu",  # CPU for dashboard responsiveness
            )
            self._pipeline_loaded = True
            logger.info("ThreatAssessmentPipeline loaded")
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
    ) -> dict:
        """
        Assess a single object, with optional caching.

        Returns a dict matching ThreatAssessmentResponse schema.
        """
        # Check cache
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

        assessment = self._pipeline.assess(
            object_id=str(object_id),
            positions=positions,
            velocities=velocities,
            timestamps=timestamps,
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

            # Use a window of recent observations for assessment
            window = min(20, len(ts))
            result = self.assess_object(
                oid,
                positions[-window:],
                velocities[-window:],
                ts[-window:],
                timestep=timestep,
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
            "contributing_factors": [],
            "explanation": "No assessment pipeline available",
            "latency_ms": 0.0,
        }
