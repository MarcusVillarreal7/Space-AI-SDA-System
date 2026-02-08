"""
Threat Scoring — Fuses Intent + Anomaly + Proximity into 0-100 Score.

The threat score combines four sub-scores:
    1. Intent score   (0-100): Based on IntentResult.threat_level (0-4 → 0-100)
    2. Anomaly score  (0-100): Based on AnomalyResult.percentile
    3. Proximity score(0-100): Based on distance, closing rate, TCA
    4. Pattern score  (0-100): Based on escalation patterns detected

Each sub-score is multiplied by a configurable weight, then the weighted
sum is clamped to [0, 100].

The system also assigns a threat tier:
    0-19:  MINIMAL   — No action required
    20-39: LOW       — Log for review
    40-59: MODERATE  — Active monitoring
    60-79: ELEVATED  — Alert operations team
    80-100: CRITICAL — Immediate response required

Author: Space AI Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.ml.intent.intent_classifier import IntentCategory, IntentResult, ThreatLevel
from src.ml.intent.proximity_context import ProximityContext
from src.ml.anomaly.anomaly_detector import AnomalyResult


# -----------------------------------------------------------------------
# Threat tier
# -----------------------------------------------------------------------

class ThreatTier(Enum):
    """Operational threat tier derived from the 0-100 score."""
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MODERATE = "MODERATE"
    ELEVATED = "ELEVATED"
    CRITICAL = "CRITICAL"


def _score_to_tier(score: float) -> ThreatTier:
    if score < 20:
        return ThreatTier.MINIMAL
    elif score < 40:
        return ThreatTier.LOW
    elif score < 60:
        return ThreatTier.MODERATE
    elif score < 80:
        return ThreatTier.ELEVATED
    else:
        return ThreatTier.CRITICAL


# -----------------------------------------------------------------------
# Weights
# -----------------------------------------------------------------------

@dataclass
class ScoringWeights:
    """
    Configurable weights for each sub-score component.

    Weights do not need to sum to 1.0 — the final score is clamped to [0, 100].
    Higher weights amplify the influence of that component.
    """
    intent: float = 0.35
    anomaly: float = 0.25
    proximity: float = 0.25
    pattern: float = 0.15

    def total(self) -> float:
        return self.intent + self.anomaly + self.proximity + self.pattern


# -----------------------------------------------------------------------
# Result
# -----------------------------------------------------------------------

@dataclass
class ThreatScore:
    """Output of the threat scoring system."""
    object_id: str
    score: float                      # 0-100 composite threat score
    tier: ThreatTier                  # Operational tier
    # Sub-scores (each 0-100)
    intent_score: float
    anomaly_score: float
    proximity_score: float
    pattern_score: float
    # Source data
    intent_result: Optional[IntentResult] = None
    anomaly_result: Optional[AnomalyResult] = None
    # Explanation
    contributing_factors: List[str] = field(default_factory=list)
    explanation: str = ""


# -----------------------------------------------------------------------
# Intent sub-score mapping
# -----------------------------------------------------------------------

# ThreatLevel (0-4) → base intent score (0-100)
_THREAT_LEVEL_SCORES = {
    ThreatLevel.NONE: 0,
    ThreatLevel.LOW: 15,
    ThreatLevel.MODERATE: 40,
    ThreatLevel.ELEVATED: 70,
    ThreatLevel.HIGH: 100,
}

# Intent category bonus — some intents are inherently more concerning
_INTENT_BONUS = {
    IntentCategory.NOMINAL: 0,
    IntentCategory.STATION_KEEPING: 0,
    IntentCategory.ORBIT_MAINTENANCE: 5,
    IntentCategory.COLLISION_AVOIDANCE: 10,
    IntentCategory.DEORBIT: 5,
    IntentCategory.ORBIT_RAISING: 10,
    IntentCategory.RENDEZVOUS: 20,
    IntentCategory.SURVEILLANCE: 25,
    IntentCategory.EVASIVE: 30,
    IntentCategory.UNKNOWN: 15,
}


# -----------------------------------------------------------------------
# Proximity sub-score functions
# -----------------------------------------------------------------------

def _proximity_distance_score(distance_km: float, warning_radius_km: float) -> float:
    """
    Score based on distance to nearest asset.
    Returns 0-100: 100 when distance=0, 0 when distance >= warning_radius.
    Uses exponential decay for smooth scoring.
    """
    if distance_km <= 0:
        return 100.0
    if distance_km >= warning_radius_km:
        return 0.0
    # Exponential decay: score = 100 * exp(-3 * d / R)
    # At d=0 → 100, at d=R → ~5
    return 100.0 * math.exp(-3.0 * distance_km / warning_radius_km)


def _proximity_closing_score(closing_rate_km_s: float) -> float:
    """
    Score based on closing rate. Only scores when approaching (negative rate).
    Maps -10 km/s → 100, 0 → 0.
    """
    if closing_rate_km_s >= 0:
        return 0.0
    # Approach rate magnitude
    rate = abs(closing_rate_km_s)
    # Linear mapping: 0 → 0, 10 → 100, clamp
    return min(100.0, rate * 10.0)


def _proximity_tca_score(tca_s: float) -> float:
    """
    Score based on time to closest approach.
    Lower TCA → higher score. TCA < 10 min → 100, TCA > 24h → 0.
    """
    if tca_s == float("inf") or tca_s < 0:
        return 0.0
    # Inverse mapping: 600s (10min) → 100, 86400s (24h) → 0
    if tca_s <= 600:
        return 100.0
    if tca_s >= 86400:
        return 0.0
    # Log-scale decay
    return 100.0 * (1.0 - math.log(tca_s / 600.0) / math.log(86400.0 / 600.0))


# -----------------------------------------------------------------------
# Pattern sub-score
# -----------------------------------------------------------------------

_PATTERN_SCORES = {
    "PHASING": 40,
    "SHADOWING": 60,
    "EVASION": 50,
}


# -----------------------------------------------------------------------
# Scorer
# -----------------------------------------------------------------------

class ThreatScorer:
    """
    Produces a unified 0-100 threat score from intent, anomaly, and proximity inputs.

    Usage::

        scorer = ThreatScorer()
        result = scorer.score(
            object_id="SAT-42",
            intent_result=intent_result,
            anomaly_result=anomaly_result,
        )
        print(result.score)  # 0-100
        print(result.tier)   # MINIMAL / LOW / MODERATE / ELEVATED / CRITICAL
    """

    def __init__(
        self,
        weights: ScoringWeights | None = None,
        warning_radius_km: float = 500.0,
    ):
        self.weights = weights or ScoringWeights()
        self.warning_radius_km = warning_radius_km

    def score(
        self,
        object_id: str,
        intent_result: Optional[IntentResult] = None,
        anomaly_result: Optional[AnomalyResult] = None,
    ) -> ThreatScore:
        """
        Compute composite threat score from available inputs.

        Either or both inputs may be provided. Missing inputs contribute 0.
        """
        factors: List[str] = []

        # --- Intent sub-score ---
        intent_sub = self._compute_intent_score(intent_result, factors)

        # --- Anomaly sub-score ---
        anomaly_sub = self._compute_anomaly_score(anomaly_result, factors)

        # --- Proximity sub-score ---
        proximity = intent_result.proximity if intent_result else None
        proximity_sub = self._compute_proximity_score(proximity, factors)

        # --- Pattern sub-score ---
        patterns = intent_result.escalation_patterns if intent_result else []
        pattern_sub = self._compute_pattern_score(patterns, factors)

        # --- Weighted combination ---
        w = self.weights
        raw = (
            w.intent * intent_sub
            + w.anomaly * anomaly_sub
            + w.proximity * proximity_sub
            + w.pattern * pattern_sub
        )
        composite = max(0.0, min(100.0, raw))
        tier = _score_to_tier(composite)

        # --- Explanation ---
        from src.ml.threat.threat_explainer import ThreatExplainer
        explanation = ThreatExplainer.explain(
            object_id=object_id,
            score=composite,
            tier=tier,
            intent_sub=intent_sub,
            anomaly_sub=anomaly_sub,
            proximity_sub=proximity_sub,
            pattern_sub=pattern_sub,
            factors=factors,
            intent_result=intent_result,
            anomaly_result=anomaly_result,
        )

        return ThreatScore(
            object_id=object_id,
            score=composite,
            tier=tier,
            intent_score=intent_sub,
            anomaly_score=anomaly_sub,
            proximity_score=proximity_sub,
            pattern_score=pattern_sub,
            intent_result=intent_result,
            anomaly_result=anomaly_result,
            contributing_factors=factors,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Sub-score computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_intent_score(
        result: Optional[IntentResult], factors: List[str]
    ) -> float:
        if result is None:
            return 0.0

        base = _THREAT_LEVEL_SCORES.get(result.threat_level, 0)
        bonus = _INTENT_BONUS.get(result.intent, 0)
        score = min(100.0, base + bonus)

        if score >= 40:
            factors.append(
                f"Intent: {result.intent.value} at {result.threat_level.name} "
                f"(confidence {result.confidence:.0%})"
            )
        return score

    @staticmethod
    def _compute_anomaly_score(
        result: Optional[AnomalyResult], factors: List[str]
    ) -> float:
        if result is None:
            return 0.0

        # Use percentile directly as the 0-100 score
        score = min(100.0, result.percentile)

        if result.is_anomaly:
            factors.append(
                f"Anomaly: score {result.anomaly_score:.4f} "
                f"(P{result.percentile:.0f}, threshold {result.threshold:.4f})"
            )
        return score

    def _compute_proximity_score(
        self, proximity: Optional[ProximityContext], factors: List[str]
    ) -> float:
        if proximity is None or proximity.nearest_asset is None:
            return 0.0

        dist_score = _proximity_distance_score(
            proximity.distance_km, self.warning_radius_km
        )
        closing_score = _proximity_closing_score(proximity.closing_rate_km_s)
        tca_score = _proximity_tca_score(proximity.time_to_closest_approach_s)

        # Weighted combination of proximity components
        score = 0.5 * dist_score + 0.3 * closing_score + 0.2 * tca_score

        if score >= 20:
            factors.append(
                f"Proximity: {proximity.distance_km:.0f} km from "
                f"{proximity.nearest_asset.name}"
                + (f", closing at {abs(proximity.closing_rate_km_s):.2f} km/s"
                   if proximity.is_approaching else ", receding")
            )
        return score

    @staticmethod
    def _compute_pattern_score(
        patterns: List[str], factors: List[str]
    ) -> float:
        if not patterns:
            return 0.0

        score = 0.0
        for p in patterns:
            ps = _PATTERN_SCORES.get(p, 20)
            score = max(score, ps)

        if score > 0:
            factors.append(f"Patterns: {', '.join(patterns)}")

        return score
