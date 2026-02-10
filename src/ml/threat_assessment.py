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
    # Intent
    intent_result: Optional[IntentResult]
    # Anomaly
    anomaly_result: Optional[AnomalyResult]
    # Threat score
    threat_score: ThreatScore
    # Timing
    latency_ms: float
    # Raw data summary
    num_observations: int
    observation_window_s: float


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
        intent_classifier: Optional[IntentClassifier] = None,
        threat_scorer: Optional[ThreatScorer] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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
    ) -> ThreatAssessment:
        """
        Run full threat assessment for a single object.

        Args:
            object_id: Object identifier.
            positions: Position time series (T, 3) in km.
            velocities: Velocity time series (T, 3) in km/s.
            timestamps: Timestamps (T,) in seconds from epoch.
            maneuver_class_override: Override maneuver class (skip heuristic).
            confidence_override: Override confidence (skip heuristic).

        Returns:
            ThreatAssessment with all module outputs.
        """
        t0 = time.perf_counter()
        n = len(timestamps)

        # Step 1: Derive maneuver events from velocity changes
        maneuver_records, maneuver_events = self._derive_maneuvers(
            positions, velocities, timestamps
        )

        # Step 2: Classify current maneuver (heuristic or override)
        if maneuver_class_override is not None:
            maneuver_class = maneuver_class_override
            confidence = confidence_override or 0.9
        else:
            maneuver_class, confidence = self._classify_current_maneuver(
                maneuver_records
            )

        maneuver_names = {
            0: "Normal", 1: "Drift/Decay", 2: "Station-keeping",
            3: "Minor Maneuver", 4: "Major Maneuver", 5: "Deorbit",
        }

        # Step 3: Current state
        current_pos = tuple(positions[-1].tolist())
        current_vel = tuple(velocities[-1].tolist())

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
            intent_result=intent_result,
            anomaly_result=anomaly_result,
            threat_score=threat_score,
            latency_ms=latency,
            num_observations=n,
            observation_window_s=window,
        )

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

        # Subtract expected gravitational contribution at each step
        # a_gravity = -mu * r / |r|^3, so expected dv = a * dt
        dt = np.diff(timestamps)  # (T-1,)
        r = positions[:-1]  # position at start of each interval
        r_mag = np.linalg.norm(r, axis=1, keepdims=True)  # (T-1, 1)
        r_mag = np.maximum(r_mag, 1.0)  # prevent division by zero
        a_gravity = -self._MU_EARTH * r / (r_mag ** 3)  # (T-1, 3)
        dv_gravity = a_gravity * dt[:, np.newaxis]  # (T-1, 3)

        # Residual delta-V = actual maneuver thrust
        dv_maneuver = dv_raw - dv_gravity
        dv_mag = np.linalg.norm(dv_maneuver, axis=1)

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
