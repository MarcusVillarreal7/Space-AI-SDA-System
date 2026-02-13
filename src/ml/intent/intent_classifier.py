"""
Intent Classification for Satellite Maneuvers.

Maps the 6 maneuver classes from the CNN-LSTM classifier to 10 operational
intent categories, incorporating proximity context and behavioral patterns
to determine threat level.

The 6 maneuver classes are:
    0: Normal, 1: Drift/Decay, 2: Station-keeping,
    3: Minor Maneuver, 4: Major Maneuver, 5: Deorbit

Author: Space AI Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.ml.intent.asset_catalog import AssetCatalog
from src.ml.intent.proximity_context import ProximityContext, compute_proximity
from src.ml.intent.threat_escalation import ManeuverEvent, ThreatEscalator


# -----------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------

class IntentCategory(Enum):
    """Operational intent categories derived from maneuver + context."""
    NOMINAL = "NOMINAL"                     # Normal ops / no maneuver
    STATION_KEEPING = "STATION_KEEPING"     # Routine station-keeping
    ORBIT_MAINTENANCE = "ORBIT_MAINTENANCE" # Minor adjustment, no threat context
    COLLISION_AVOIDANCE = "COLLISION_AVOIDANCE"
    DEORBIT = "DEORBIT"                     # End-of-life disposal
    ORBIT_RAISING = "ORBIT_RAISING"         # Significant altitude change
    RENDEZVOUS = "RENDEZVOUS"              # Approaching another object
    SURVEILLANCE = "SURVEILLANCE"           # Shadowing / loitering near asset
    EVASIVE = "EVASIVE"                     # Maneuver suggesting observation avoidance
    UNKNOWN = "UNKNOWN"


class ThreatLevel(Enum):
    """Threat severity (ordinal)."""
    NONE = 0
    LOW = 1
    MODERATE = 2
    ELEVATED = 3
    HIGH = 4


# Ordinal comparison helpers
_THREAT_ORDER = {t: t.value for t in ThreatLevel}


def _max_threat(*levels: ThreatLevel) -> ThreatLevel:
    return max(levels, key=lambda t: _THREAT_ORDER[t])


# -----------------------------------------------------------------------
# Result dataclass
# -----------------------------------------------------------------------

@dataclass
class IntentResult:
    """Output of the intent classifier."""
    intent: IntentCategory
    threat_level: ThreatLevel
    confidence: float                # 0-1, inherited from maneuver classifier
    proximity: Optional[ProximityContext] = None
    escalation_patterns: List[str] = field(default_factory=list)
    explanation: str = ""


# -----------------------------------------------------------------------
# Base mapping: maneuver class → (default intent, default threat)
# -----------------------------------------------------------------------

_BASE_MAPPING: Dict[int, Tuple[IntentCategory, ThreatLevel]] = {
    0: (IntentCategory.NOMINAL, ThreatLevel.NONE),
    1: (IntentCategory.NOMINAL, ThreatLevel.LOW),           # Drift/Decay
    2: (IntentCategory.STATION_KEEPING, ThreatLevel.NONE),
    3: (IntentCategory.ORBIT_MAINTENANCE, ThreatLevel.LOW),
    4: (IntentCategory.ORBIT_RAISING, ThreatLevel.MODERATE),
    5: (IntentCategory.DEORBIT, ThreatLevel.LOW),
}


# -----------------------------------------------------------------------
# Intent Classifier
# -----------------------------------------------------------------------

class IntentClassifier:
    """
    Classifies satellite maneuver intent and threat level.

    Pipeline:
        1. Map maneuver class → base intent + base threat
        2. Compute proximity context to high-value assets
        3. Apply threat escalation rules (proximity, patterns)
        4. Generate human-readable explanation
    """

    def __init__(
        self,
        catalog: Optional[AssetCatalog] = None,
        escalator: Optional[ThreatEscalator] = None,
        warning_radius_km: float = 500.0,
        approach_threshold_km: float = 100.0,
    ):
        self.catalog = catalog or AssetCatalog()
        self.escalator = escalator or ThreatEscalator()
        self.warning_radius_km = warning_radius_km
        self.approach_threshold_km = approach_threshold_km

    def classify(
        self,
        maneuver_class: int,
        confidence: float,
        position_km: Tuple[float, float, float],
        velocity_km_s: Tuple[float, float, float],
        maneuver_history: Optional[List[ManeuverEvent]] = None,
    ) -> IntentResult:
        """
        Classify intent from a maneuver prediction and orbital state.

        Args:
            maneuver_class: Predicted maneuver class index (0-5).
            confidence: Classifier confidence (0-1).
            position_km: Current ECI position.
            velocity_km_s: Current ECI velocity.
            maneuver_history: Recent maneuver events for pattern detection.

        Returns:
            IntentResult with intent, threat level, and explanation.
        """
        # Step 1: Base mapping
        base_intent, base_threat = _BASE_MAPPING.get(
            maneuver_class, (IntentCategory.UNKNOWN, ThreatLevel.LOW)
        )

        # Step 2: Proximity context
        proximity = compute_proximity(
            position_km, velocity_km_s, self.catalog, self.warning_radius_km
        )

        # Step 3: Escalation
        intent, threat, patterns = self._apply_escalation(
            base_intent, base_threat, maneuver_class, proximity,
            maneuver_history or [],
        )

        # Step 4: Confidence adjustment — lower confidence reduces threat
        if confidence < 0.5:
            threat = ThreatLevel(max(0, threat.value - 1))

        # Step 5: Explanation
        explanation = self._build_explanation(
            intent, threat, maneuver_class, confidence, proximity, patterns
        )

        return IntentResult(
            intent=intent,
            threat_level=threat,
            confidence=confidence,
            proximity=proximity,
            escalation_patterns=patterns,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Escalation logic
    # ------------------------------------------------------------------

    def _apply_escalation(
        self,
        intent: IntentCategory,
        threat: ThreatLevel,
        maneuver_class: int,
        proximity: ProximityContext,
        history: List[ManeuverEvent],
    ) -> Tuple[IntentCategory, ThreatLevel, List[str]]:
        """Apply proximity and pattern-based escalation rules."""
        patterns = self.escalator.detect_patterns(history, proximity)

        # Rule 1: Minor/Major maneuver + approaching asset → RENDEZVOUS
        if maneuver_class in {3, 4} and proximity.is_approaching:
            if proximity.distance_km < self.warning_radius_km:
                intent = IntentCategory.RENDEZVOUS
                threat = _max_threat(threat, ThreatLevel.ELEVATED)
            if proximity.distance_km < self.approach_threshold_km:
                threat = ThreatLevel.HIGH

        # Rule 2: Station-keeping near asset while co-orbital → SURVEILLANCE.
        # Require co-orbital relative speed (< 1 km/s) to distinguish active
        # shadowing from flyby conjunctions. A LEO satellite passing the ISS
        # at 4+ km/s relative velocity is a brief flyby, not surveillance.
        if (maneuver_class == 2
                and proximity.closing_rate_km_s < -0.01
                and proximity.distance_km < self.warning_radius_km
                and proximity.regime_match
                and proximity.is_coorbital):
            intent = IntentCategory.SURVEILLANCE
            threat = _max_threat(threat, ThreatLevel.ELEVATED)

        # Rule 3: Phasing pattern → SURVEILLANCE / RENDEZVOUS
        if "PHASING" in patterns:
            if proximity.distance_km < self.warning_radius_km:
                intent = IntentCategory.RENDEZVOUS
            else:
                intent = IntentCategory.SURVEILLANCE
            threat = _max_threat(threat, ThreatLevel.ELEVATED)

        # Rule 4: Shadowing pattern → SURVEILLANCE
        if "SHADOWING" in patterns:
            intent = IntentCategory.SURVEILLANCE
            threat = _max_threat(threat, ThreatLevel.HIGH)

        # Rule 5: Evasion pattern → EVASIVE
        if "EVASION" in patterns:
            intent = IntentCategory.EVASIVE
            threat = _max_threat(threat, ThreatLevel.ELEVATED)

        # Rule 6: Collision avoidance — any maneuver when object is very
        # close and closing fast, regardless of class
        if (
            proximity.is_approaching
            and proximity.distance_km < self.approach_threshold_km
            and proximity.closing_rate_km_s < -0.01
        ):
            intent = IntentCategory.COLLISION_AVOIDANCE
            threat = _max_threat(threat, ThreatLevel.HIGH)

        # Rule 7: CNN-LSTM station-keeping + sustained co-orbital proximity → SHADOWING.
        # The per-timestep heuristic delta-V classifier may produce class 0
        # for GEO station-keeping (residual < 5 m/s), so the ThreatEscalator
        # won't detect SHADOWING via class-2 events.  Use the CNN-LSTM's
        # overall classification combined with trajectory duration.
        # Requires co-orbital relative speed (< 1 km/s) — objects crossing
        # through at high relative velocity are flybys, not shadowing.
        if (
            "SHADOWING" not in patterns
            and maneuver_class == 2
            and proximity.regime_match
            and proximity.is_coorbital
            and proximity.distance_km < self.escalator.shadowing_range_km
            and len(history) >= 2
            and (history[-1].timestamp - history[0].timestamp)
                >= self.escalator.shadowing_duration_s
        ):
            patterns.append("SHADOWING")
            intent = IntentCategory.SURVEILLANCE
            threat = _max_threat(threat, ThreatLevel.HIGH)

        return intent, threat, patterns

    # ------------------------------------------------------------------
    # Explanation generation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_explanation(
        intent: IntentCategory,
        threat: ThreatLevel,
        maneuver_class: int,
        confidence: float,
        proximity: ProximityContext,
        patterns: List[str],
    ) -> str:
        from src.ml.models.maneuver_classifier import CLASS_NAMES

        class_name = CLASS_NAMES.get(maneuver_class, "Unknown")
        parts = [
            f"Maneuver classified as '{class_name}' (confidence {confidence:.0%}).",
            f"Intent: {intent.value}.",
            f"Threat level: {threat.name}.",
        ]

        if proximity.nearest_asset is not None:
            parts.append(
                f"Nearest asset: {proximity.nearest_asset.name} "
                f"at {proximity.distance_km:.1f} km "
                f"({'approaching' if proximity.is_approaching else 'receding'})."
            )
            if proximity.time_to_closest_approach_s < float("inf"):
                tca_min = proximity.time_to_closest_approach_s / 60.0
                parts.append(f"Estimated TCA: {tca_min:.1f} min.")

        if patterns:
            parts.append(f"Detected patterns: {', '.join(patterns)}.")

        return " ".join(parts)
