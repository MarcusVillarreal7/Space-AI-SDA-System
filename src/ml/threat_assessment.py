"""
End-to-End Threat Assessment Pipeline.

Wires together all Phase 3 ML modules into a single entry point:

    Raw Track Data → Trajectory Prediction → Maneuver Classification
                   → Intent Classification → Anomaly Detection
                   → Threat Scoring → ThreatAssessment

This is the top-level API that Phase 4 (Dashboard) will consume.

Author: Space AI Team
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.ml.anomaly.anomaly_detector import AnomalyDetector, AnomalyResult
from src.ml.anomaly.behavior_features import (
    BehaviorFeatureExtractor,
    BehaviorProfile,
    ManeuverRecord,
)
from src.ml.intent.intent_classifier import (
    IntentCategory,
    IntentClassifier,
    IntentResult,
    ThreatLevel,
)
from src.ml.intent.threat_escalation import ManeuverEvent
from src.ml.threat.threat_scorer import ThreatScore, ThreatScorer, ThreatTier


# -----------------------------------------------------------------------
# Result dataclass
# -----------------------------------------------------------------------

@dataclass
class ThreatAssessment:
    """Complete assessment for a single space object."""
    object_id: str
    # Maneuver classification
    maneuver_class: int
    maneuver_name: str
    maneuver_confidence: float
    maneuver_probabilities: Optional[List[float]] = None
    # Intent
    intent_result: Optional[IntentResult] = None
    # Anomaly
    anomaly_result: Optional[AnomalyResult] = None
    # Threat score
    threat_score: Optional[ThreatScore] = None
    # Timing
    latency_ms: float = 0.0
    # Raw data summary
    num_observations: int = 0
    observation_window_s: float = 0.0


# -----------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------

class ThreatAssessmentPipeline:
    """
    End-to-end pipeline: raw track data → threat assessment.

    Modules:
        1. ManeuverClassifier (CNN-LSTM) — class 0-5 + confidence
        2. IntentClassifier — intent + threat level + proximity
        3. AnomalyDetector — behavioral anomaly score
        4. ThreatScorer — composite 0-100 score

    The pipeline does NOT include trajectory prediction (Transformer) because
    that operates on feature sequences, while this pipeline operates on raw
    position/velocity time series. The Transformer predictions feed into
    downstream consumers (dashboard visualization). This pipeline focuses on
    the classification → scoring chain.

    Usage::

        pipeline = ThreatAssessmentPipeline(
            anomaly_checkpoint="checkpoints/phase3_anomaly",
        )
        assessment = pipeline.assess(
            object_id="SAT-42",
            positions=positions_array,
            velocities=velocities_array,
            timestamps=timestamps_array,
        )
        print(assessment.threat_score.score)  # 0-100
        print(assessment.threat_score.tier)   # MINIMAL / LOW / ...
    """

    # Delta-V classification thresholds (matches training script)
    # Applied AFTER subtracting expected gravitational acceleration
    _DV_THRESHOLDS = [0.005, 0.01, 0.02, 0.1, 1.0]

    # Earth gravitational parameter (km³/s²)
    _MU_EARTH = 398600.4418

    def __init__(
        self,
        anomaly_checkpoint: Optional[str] = None,
        maneuver_checkpoint: Optional[str] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        threat_scorer: Optional[ThreatScorer] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Neural maneuver classifier (CNN-LSTM, 719K params)
        self.maneuver_predictor = None
        maneuver_ckpt = maneuver_checkpoint or "checkpoints/phase3_day4/maneuver_classifier.pt"
        if Path(maneuver_ckpt).exists():
            try:
                from src.ml.inference import ManeuverPredictor
                self.maneuver_predictor = ManeuverPredictor(
                    maneuver_ckpt, device=self.device
                )
            except Exception:
                pass  # Fall back to heuristic

        # Intent classifier (rule-based, no checkpoint needed)
        self.intent_classifier = intent_classifier or IntentClassifier()

        # Anomaly detector (needs trained checkpoint)
        if anomaly_detector is not None:
            self.anomaly_detector = anomaly_detector
        elif anomaly_checkpoint and Path(anomaly_checkpoint).exists():
            self.anomaly_detector = AnomalyDetector.load(
                anomaly_checkpoint, device=self.device
            )
        else:
            self.anomaly_detector = None

        # Threat scorer (rule-based, no checkpoint needed)
        self.threat_scorer = threat_scorer or ThreatScorer()

        # Feature extractor for anomaly detection
        self.behavior_extractor = BehaviorFeatureExtractor()

    def assess(
        self,
        object_id: str,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray,
        maneuver_class_override: Optional[int] = None,
        confidence_override: Optional[float] = None,
        full_positions: Optional[np.ndarray] = None,
        full_velocities: Optional[np.ndarray] = None,
        full_timestamps: Optional[np.ndarray] = None,
    ) -> ThreatAssessment:
        """
        Run full threat assessment for a single object.

        Args:
            object_id: Object identifier.
            positions: Position time series (T, 3) in km — used for
                CNN-LSTM classification and anomaly detection.
            velocities: Velocity time series (T, 3) in km/s.
            timestamps: Timestamps (T,) in seconds from epoch.
            maneuver_class_override: Override maneuver class (skip heuristic).
            confidence_override: Override confidence (skip heuristic).
            full_positions: Optional full trajectory (T_full, 3) for proximity
                scanning and maneuver history derivation.
            full_velocities: Matching velocities for *full_positions*.
            full_timestamps: Matching timestamps for *full_positions*.

        Returns:
            ThreatAssessment with all module outputs.
        """
        t0 = time.perf_counter()
        n = len(timestamps)

        # Step 1: Derive maneuver events from velocity changes.
        # Use the full trajectory when available so that pattern detection
        # (SHADOWING, PHASING, EVASION) can see hours of maneuver history
        # instead of only the 20-step classification window (~20 min).
        if (full_positions is not None and full_velocities is not None
                and full_timestamps is not None):
            maneuver_records, maneuver_events = self._derive_maneuvers(
                full_positions, full_velocities, full_timestamps
            )
        else:
            maneuver_records, maneuver_events = self._derive_maneuvers(
                positions, velocities, timestamps
            )

        # Step 2: Classify current maneuver (neural, heuristic, or override)
        maneuver_probs = None
        if maneuver_class_override is not None:
            maneuver_class = maneuver_class_override
            confidence = confidence_override or 0.9
        elif self.maneuver_predictor is not None and n >= 5:
            try:
                pred = self.maneuver_predictor.predict(
                    positions, velocities, timestamps
                )
                maneuver_class = pred["class_idx"]
                confidence = pred["confidence"]
                maneuver_probs = pred["probabilities"].tolist()
            except Exception:
                maneuver_class, confidence = self._classify_current_maneuver(
                    maneuver_records
                )
        else:
            maneuver_class, confidence = self._classify_current_maneuver(
                maneuver_records
            )

        maneuver_names = {
            0: "Normal", 1: "Drift/Decay", 2: "Station-keeping",
            3: "Minor Maneuver", 4: "Major Maneuver", 5: "Deorbit",
        }

        # Step 3: Find closest approach to any asset across the trajectory.
        # When full_positions is provided, scan the entire trajectory for
        # proximity events; otherwise scan only the classification window.
        prox_pos = full_positions if full_positions is not None else positions
        prox_vel = full_velocities if full_velocities is not None else velocities
        # Only use propagated asset positions when we have the full trajectory.
        # Short classification windows would give wrong results because the
        # asset orbits far from its catalog position in that time.
        prox_ts = full_timestamps if full_timestamps is not None else None
        current_pos, current_vel = self._find_closest_approach(
            prox_pos, prox_vel, timestamps=prox_ts
        )

        # Step 4: Intent classification
        intent_result = self.intent_classifier.classify(
            maneuver_class=maneuver_class,
            confidence=confidence,
            position_km=current_pos,
            velocity_km_s=current_vel,
            maneuver_history=maneuver_events,
        )

        # Step 5: Anomaly detection
        anomaly_result = None
        if self.anomaly_detector is not None:
            profile = self.behavior_extractor.extract(
                object_id=object_id,
                maneuvers=maneuver_records,
                position_km=current_pos,
                velocity_km_s=current_vel,
            )
            anomaly_result = self.anomaly_detector.detect(profile)

        # Step 6: Threat scoring
        threat_score = self.threat_scorer.score(
            object_id=object_id,
            intent_result=intent_result,
            anomaly_result=anomaly_result,
            maneuver_class=maneuver_class,
            maneuver_confidence=confidence,
        )

        latency = (time.perf_counter() - t0) * 1000.0

        # Observation window
        if n >= 2:
            window = float(timestamps[-1] - timestamps[0])
        else:
            window = 0.0

        return ThreatAssessment(
            object_id=object_id,
            maneuver_class=maneuver_class,
            maneuver_name=maneuver_names.get(maneuver_class, "Unknown"),
            maneuver_confidence=confidence,
            maneuver_probabilities=maneuver_probs,
            intent_result=intent_result,
            anomaly_result=anomaly_result,
            threat_score=threat_score,
            latency_ms=latency,
            num_observations=n,
            observation_window_s=window,
        )

    def assess_by_type(
        self,
        object_id: str,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray,
        object_type: str = "PAYLOAD",
        **kwargs,
    ) -> ThreatAssessment:
        """
        Type-aware assessment dispatcher.

        Routes PAYLOAD objects to the full 6-step pipeline (assess()),
        and DEBRIS / ROCKET_BODY to a collision-only path that skips
        maneuver classification, intent analysis, and anomaly detection.
        """
        if object_type in ("DEBRIS", "ROCKET_BODY"):
            return self._assess_passive_object(
                object_id=object_id,
                positions=positions,
                velocities=velocities,
                timestamps=timestamps,
                object_type=object_type,
                full_positions=kwargs.get("full_positions"),
                full_velocities=kwargs.get("full_velocities"),
                full_timestamps=kwargs.get("full_timestamps"),
            )
        return self.assess(
            object_id=object_id,
            positions=positions,
            velocities=velocities,
            timestamps=timestamps,
            **kwargs,
        )

    def _assess_passive_object(
        self,
        object_id: str,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray,
        object_type: str = "DEBRIS",
        full_positions: Optional[np.ndarray] = None,
        full_velocities: Optional[np.ndarray] = None,
        full_timestamps: Optional[np.ndarray] = None,
    ) -> ThreatAssessment:
        """
        Collision-only assessment for passive objects (debris / rocket bodies).

        Skips maneuver classification, intent analysis, and anomaly detection.
        Only computes proximity to high-value assets and assigns a threat score
        based on collision geometry. Rocket bodies get a +5 breakup risk bonus.
        """
        t0 = time.perf_counter()
        n = len(timestamps)

        # Proximity scan (full trajectory if available)
        prox_pos = full_positions if full_positions is not None else positions
        prox_vel = full_velocities if full_velocities is not None else velocities
        prox_ts = full_timestamps if full_timestamps is not None else None
        current_pos, current_vel = self._find_closest_approach(
            prox_pos, prox_vel, timestamps=prox_ts
        )

        # Compute proximity using the same proximity context + threat scorer logic
        proximity_score = 0.0
        nearest_asset = "none"
        nearest_dist = float("inf")
        try:
            from src.ml.intent.proximity_context import compute_proximity
            prox = compute_proximity(
                current_pos, current_vel,
                catalog=self.intent_classifier.catalog if self.intent_classifier else None,
                warning_radius_km=self.intent_classifier.warning_radius_km if self.intent_classifier else 500.0,
            )
            if prox and prox.nearest_asset is not None:
                nearest_asset = prox.nearest_asset.name
                nearest_dist = prox.distance_km
                # Use same scoring as ThreatScorer._compute_proximity_score
                proximity_score = self.threat_scorer._compute_proximity_score(prox, [])
        except Exception:
            pass

        # Breakup risk bonus for rocket bodies (pressurized tanks)
        breakup_bonus = 5.0 if object_type == "ROCKET_BODY" else 0.0

        # Threat score: proximity-only (capped at 100)
        raw_score = proximity_score + breakup_bonus
        score = min(100.0, max(0.0, raw_score))

        # Tier from score
        if score >= 80:
            tier = ThreatTier.CRITICAL
        elif score >= 60:
            tier = ThreatTier.ELEVATED
        elif score >= 40:
            tier = ThreatTier.MODERATE
        elif score >= 20:
            tier = ThreatTier.LOW
        else:
            tier = ThreatTier.MINIMAL

        factors = []
        if proximity_score > 0:
            factors.append(f"Proximity to {nearest_asset}: {nearest_dist:.0f} km")
        if breakup_bonus > 0:
            factors.append("Rocket body breakup risk (+5)")

        # Build contextual explanation
        explanation = self._build_passive_explanation(
            object_type, nearest_asset, nearest_dist, proximity_score,
            breakup_bonus, score, tier,
        )

        threat_score = ThreatScore(
            object_id=object_id,
            score=round(score, 2),
            tier=tier,
            intent_score=0.0,
            anomaly_score=0.0,
            proximity_score=round(proximity_score, 2),
            pattern_score=0.0,
            contributing_factors=factors,
            explanation=explanation,
        )

        latency = (time.perf_counter() - t0) * 1000.0

        return ThreatAssessment(
            object_id=object_id,
            maneuver_class=0,
            maneuver_name="Normal",
            maneuver_confidence=1.0,
            maneuver_probabilities=None,
            intent_result=None,
            anomaly_result=None,
            threat_score=threat_score,
            latency_ms=latency,
            num_observations=n,
            observation_window_s=float(timestamps[-1] - timestamps[0]) if n >= 2 else 0.0,
        )

    @staticmethod
    def _build_passive_explanation(
        object_type: str,
        nearest_asset: str,
        nearest_dist: float,
        proximity_score: float,
        breakup_bonus: float,
        score: float,
        tier: "ThreatTier",
    ) -> str:
        """Build a contextual explanation for debris/rocket body assessment."""
        type_label = "Debris" if object_type == "DEBRIS" else "Rocket body"
        parts = []

        # What is this object?
        if object_type == "DEBRIS":
            parts.append(
                f"This is tracked orbital debris — a non-maneuverable fragment "
                f"that poses a passive collision risk. Intent and maneuver analysis "
                f"are not applicable (debris cannot change its orbit)."
            )
        else:
            parts.append(
                f"This is a spent rocket body — a large, non-maneuverable object. "
                f"Rocket bodies carry additional risk because pressurized fuel tanks "
                f"can fragment unpredictably, generating new debris fields."
            )

        # Proximity context
        if nearest_dist < float("inf"):
            if nearest_dist < 100:
                parts.append(
                    f"CLOSE APPROACH: Currently {nearest_dist:.0f} km from "
                    f"{nearest_asset}. At this range, even small tracking "
                    f"uncertainties could mask a collision trajectory."
                )
            elif nearest_dist < 500:
                parts.append(
                    f"Within warning radius of {nearest_asset} at "
                    f"{nearest_dist:.0f} km. Orbital mechanics could bring this "
                    f"object closer on subsequent passes."
                )
            elif nearest_dist < 2000:
                parts.append(
                    f"Nearest protected asset is {nearest_asset} at "
                    f"{nearest_dist:.0f} km — outside the 500 km warning radius "
                    f"but within monitoring range."
                )
            else:
                parts.append(
                    f"Nearest protected asset is {nearest_asset} at "
                    f"{nearest_dist:.0f} km — well outside the 500 km warning "
                    f"radius. No conjunction risk at this time."
                )
        else:
            parts.append("No protected assets in the same orbital regime.")

        # Tier context
        tier_val = tier.value if hasattr(tier, "value") else str(tier)
        if tier_val == "MINIMAL":
            parts.append(
                f"Threat score: {score:.0f}/100 ({tier_val}). No action required."
            )
        elif tier_val == "LOW":
            parts.append(
                f"Threat score: {score:.0f}/100 ({tier_val}). Logged for periodic review."
            )
        elif tier_val == "MODERATE":
            parts.append(
                f"Threat score: {score:.0f}/100 ({tier_val}). Active monitoring recommended "
                f"— track this object through its next orbital pass."
            )
        elif tier_val in ("ELEVATED", "CRITICAL"):
            parts.append(
                f"Threat score: {score:.0f}/100 ({tier_val}). Collision avoidance "
                f"maneuver may be required for the threatened asset."
            )

        return "\n".join(parts)

    def assess_batch(
        self,
        objects: List[Dict],
    ) -> List[ThreatAssessment]:
        """
        Assess multiple objects.

        Args:
            objects: List of dicts with keys:
                object_id, positions, velocities, timestamps

        Returns:
            List of ThreatAssessment.
        """
        return [
            self.assess(
                object_id=obj["object_id"],
                positions=obj["positions"],
                velocities=obj["velocities"],
                timestamps=obj["timestamps"],
            )
            for obj in objects
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_closest_approach(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        sample_stride: int = 10,
    ) -> Tuple[tuple, tuple]:
        """
        Find the trajectory timestep with the closest approach to any
        **same-regime** high-value asset, then return a (position,
        velocity) pair suitable for ``compute_proximity()``.

        When *timestamps* are provided, asset positions are propagated
        forward using a circular-orbit (Rodrigues rotation) approximation
        so that LEO assets (ISS, 90-min period) are compared at their
        correct positions.  The returned position/velocity are "warped"
        so that ``compute_proximity()`` — which uses the static asset
        catalog — produces the correct relative state (distance, closing
        rate, TCA).

        Only overrides the default (last timestep) when a genuine close
        approach is detected (min distance < warning radius).
        """
        import math as _math

        n = len(positions)
        last_pos = tuple(positions[-1].tolist())
        last_vel = tuple(velocities[-1].tolist())
        catalog = self.intent_classifier.catalog

        if n <= 1 or catalog is None or not catalog.all():
            return last_pos, last_vel

        from src.ml.intent.proximity_context import classify_regime
        mid = n // 2
        obj_regime = classify_regime(tuple(positions[mid].tolist()))

        same_regime_assets = catalog.by_regime(obj_regime)
        if not same_regime_assets:
            return last_pos, last_vel

        warning_radius = self.intent_classifier.warning_radius_km

        # Sample indices
        indices = np.arange(0, n, sample_stride)
        if indices[-1] != n - 1:
            indices = np.append(indices, n - 1)
        sampled_pos = positions[indices]   # (S, 3)

        best_dist = float("inf")
        best_tidx = -1
        best_asset_idx = -1
        best_prop_pos = None
        best_prop_vel = None

        use_propagation = (timestamps is not None and len(timestamps) > 1)

        if use_propagation:
            sampled_times = timestamps[indices]
            t0 = float(timestamps[0])

            for ai, asset in enumerate(same_regime_assets):
                a_pos = np.asarray(asset.position_km, dtype=np.float64)
                a_vel = np.asarray(asset.velocity_km_s, dtype=np.float64)
                r_mag = np.linalg.norm(a_pos)
                v_mag = np.linalg.norm(a_vel)
                if r_mag < 100.0 or v_mag < 0.01:
                    continue

                # Angular velocity and orbital angular momentum
                omega = v_mag / r_mag
                h = np.cross(a_pos, a_vel)
                h_norm = np.linalg.norm(h)
                if h_norm < 1e-10:
                    continue
                h_hat = h / h_norm

                # Vectorized Rodrigues rotation for all sampled times
                angles = omega * (sampled_times - t0)     # (S,)
                cos_a = np.cos(angles)[:, np.newaxis]     # (S,1)
                sin_a = np.sin(angles)[:, np.newaxis]     # (S,1)

                cross_hp = np.cross(h_hat, a_pos)         # (3,)
                dot_hp = np.dot(h_hat, a_pos)             # scalar
                cross_hv = np.cross(h_hat, a_vel)         # (3,)
                dot_hv = np.dot(h_hat, a_vel)             # scalar

                prop_pos = (a_pos * cos_a
                            + cross_hp * sin_a
                            + h_hat * dot_hp * (1.0 - cos_a))   # (S,3)
                prop_vel = (a_vel * cos_a
                            + cross_hv * sin_a
                            + h_hat * dot_hv * (1.0 - cos_a))   # (S,3)

                dists = np.linalg.norm(sampled_pos - prop_pos, axis=1)
                min_idx = int(np.argmin(dists))
                min_d = float(dists[min_idx])

                if min_d < best_dist:
                    best_dist = min_d
                    best_tidx = int(indices[min_idx])
                    best_asset_idx = ai
                    best_prop_pos = prop_pos[min_idx].copy()
                    best_prop_vel = prop_vel[min_idx].copy()
        else:
            # Static comparison (no timestamps available)
            asset_positions = np.array(
                [a.position_km for a in same_regime_assets], dtype=np.float64
            )
            diff = sampled_pos[:, np.newaxis, :] - asset_positions[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            best_dist = float(dists.min())
            if best_dist < warning_radius:
                flat_idx = int(np.argmin(dists))
                si = flat_idx // dists.shape[1]
                best_tidx = int(indices[si])
                return (tuple(positions[best_tidx].tolist()),
                        tuple(velocities[best_tidx].tolist()))
            return last_pos, last_vel

        if best_dist < warning_radius and best_tidx >= 0:
            # "Warp" position/velocity so that compute_proximity()
            # (which uses static catalog positions) produces the correct
            # relative state (distance, closing rate, TCA).
            asset = same_regime_assets[best_asset_idx]
            static_pos = np.asarray(asset.position_km, dtype=np.float64)
            static_vel = np.asarray(asset.velocity_km_s, dtype=np.float64)

            rel_pos = positions[best_tidx] - best_prop_pos
            rel_vel = velocities[best_tidx] - best_prop_vel

            synth_pos = static_pos + rel_pos
            synth_vel = static_vel + rel_vel
            return (tuple(synth_pos.tolist()), tuple(synth_vel.tolist()))

        return last_pos, last_vel

    def _derive_maneuvers(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray,
    ) -> Tuple[List[ManeuverRecord], List[ManeuverEvent]]:
        """
        Derive maneuver events from velocity changes.

        Subtracts expected gravitational acceleration so that natural orbital
        velocity direction changes (which can be 0.4+ km/s per 60s step for
        LEO) are not misclassified as maneuvers. Only the residual delta-V
        (actual thrust) is classified.
        """
        if len(timestamps) < 2:
            return [], []

        # Compute raw velocity differences
        dv_raw = np.diff(velocities, axis=0)

        # Subtract expected gravitational contribution using an adaptive
        # method: compute residuals via both forward Euler and trapezoidal
        # gravity subtraction, then pick whichever gives a smaller residual
        # at each timestep.
        #
        # Why adaptive: the scenario injector propagates trajectories using
        # forward Euler integration.  Trapezoidal gravity subtraction on
        # Euler-integrated data introduces ~16 m/s residuals (class 2),
        # masking the real maneuver events needed for pattern detection.
        # Conversely, analytical/Kepler trajectories have O(dt²) Euler
        # residuals (~18 m/s for LEO) that trapezoidal reduces to O(dt³).
        # Using the minimum of both methods handles either data source.
        dt = np.diff(timestamps)  # (T-1,)
        r_start = positions[:-1]
        r_end = positions[1:]
        r_mag_start = np.maximum(np.linalg.norm(r_start, axis=1, keepdims=True), 1.0)
        r_mag_end = np.maximum(np.linalg.norm(r_end, axis=1, keepdims=True), 1.0)
        a_start = -self._MU_EARTH * r_start / (r_mag_start ** 3)
        a_end = -self._MU_EARTH * r_end / (r_mag_end ** 3)

        # Forward Euler: dv_gravity ≈ a(r_start) * dt
        dv_euler = a_start * dt[:, np.newaxis]
        # Trapezoidal: dv_gravity ≈ (a(r_start) + a(r_end))/2 * dt
        dv_trap = 0.5 * (a_start + a_end) * dt[:, np.newaxis]

        res_euler = dv_raw - dv_euler
        res_trap = dv_raw - dv_trap
        euler_mag = np.linalg.norm(res_euler, axis=1)
        trap_mag = np.linalg.norm(res_trap, axis=1)

        # Pick whichever method gives the smaller residual at each step
        use_trap = trap_mag < euler_mag
        dv_maneuver = np.where(use_trap[:, np.newaxis], res_trap, res_euler)
        dv_mag = np.minimum(euler_mag, trap_mag)

        maneuver_records = []
        maneuver_events = []
        names = {
            0: "Normal", 1: "Drift/Decay", 2: "Station-keeping",
            3: "Minor Maneuver", 4: "Major Maneuver", 5: "Deorbit",
        }

        for i, mag in enumerate(dv_mag):
            mc = self._classify_delta_v(float(mag))
            t = float(timestamps[i + 1])

            maneuver_records.append(ManeuverRecord(
                timestamp=t,
                maneuver_class=mc,
                delta_v_magnitude=float(mag),
            ))
            maneuver_events.append(ManeuverEvent(
                timestamp=t,
                maneuver_class=mc,
                class_name=names.get(mc, "Unknown"),
                delta_v_magnitude=float(mag),
                position_km=tuple(positions[i + 1].tolist()),
            ))

        return maneuver_records, maneuver_events

    def _classify_delta_v(self, dv: float) -> int:
        """Classify delta-V magnitude into maneuver class 0-5."""
        for cls_idx, threshold in enumerate(self._DV_THRESHOLDS):
            if dv < threshold:
                return cls_idx
        return 5

    @staticmethod
    def _classify_current_maneuver(
        records: List[ManeuverRecord],
    ) -> Tuple[int, float]:
        """
        Classify current maneuver state from recent records.

        Uses a windowed majority vote over the last 10 records.
        """
        if not records:
            return 0, 0.5

        recent = records[-10:]
        classes = [r.maneuver_class for r in recent]

        from collections import Counter
        counts = Counter(classes)
        dominant_class, dominant_count = counts.most_common(1)[0]
        confidence = dominant_count / len(recent)

        return dominant_class, confidence
