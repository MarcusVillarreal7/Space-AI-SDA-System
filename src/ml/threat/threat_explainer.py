"""
Human-Readable Threat Explanations.

Generates natural-language explanations for threat scores that
operators can understand and act upon.

Author: Space AI Team
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.ml.intent.intent_classifier import IntentResult
    from src.ml.anomaly.anomaly_detector import AnomalyResult
    from src.ml.threat.threat_scorer import ThreatTier


_TIER_ACTIONS = {
    "MINIMAL": "No action required.",
    "LOW": "Log for periodic review.",
    "MODERATE": "Active monitoring recommended.",
    "ELEVATED": "Alert operations team for assessment.",
    "CRITICAL": "Immediate response required — potential threat to assets.",
}

# Maneuver class descriptions for operators
_MANEUVER_CONTEXT = {
    0: "maintaining a stable orbit with no detectable thrust activity",
    1: "experiencing gradual orbital decay, likely due to atmospheric drag",
    2: "performing routine station-keeping burns to maintain its assigned orbit",
    3: "executing a minor orbital adjustment (delta-V < 0.1 km/s)",
    4: "executing a significant orbital maneuver (delta-V > 0.1 km/s) — potential orbit change",
    5: "performing a deorbit burn — controlled descent or disposal maneuver",
}


class ThreatExplainer:
    """Generates human-readable threat explanations."""

    @staticmethod
    def explain(
        object_id: str,
        score: float,
        tier: "ThreatTier",
        intent_sub: float,
        anomaly_sub: float,
        proximity_sub: float,
        pattern_sub: float,
        factors: List[str],
        intent_result: Optional["IntentResult"] = None,
        anomaly_result: Optional["AnomalyResult"] = None,
        maneuver_class: int = 0,
        maneuver_confidence: float = 0.0,
    ) -> str:
        parts: List[str] = []

        # Summary line
        parts.append(
            f"Object {object_id}: Threat score {score:.1f}/100 ({tier.value})."
        )

        # Recommended action
        action = _TIER_ACTIONS.get(tier.value, "")
        if action:
            parts.append(action)

        # Behavioral context — what is this object doing?
        maneuver_desc = _MANEUVER_CONTEXT.get(maneuver_class, "")
        if maneuver_desc:
            conf_pct = int(maneuver_confidence * 100)
            parts.append(
                f"This satellite is currently {maneuver_desc} "
                f"({conf_pct}% confidence)."
            )

        # Why this IS a threat (for MODERATE+)
        if tier.value in ("ELEVATED", "CRITICAL") and intent_result is not None:
            intent_name = getattr(intent_result, "intent", None)
            if intent_name is not None:
                intent_val = intent_name.value if hasattr(intent_name, "value") else str(intent_name)
                reason = _intent_threat_reason(intent_val)
                if reason:
                    parts.append(reason)

        # Why this is NOT a threat (for MINIMAL/LOW with no significant scores)
        if tier.value in ("MINIMAL", "LOW"):
            benign_reasons = []
            if proximity_sub < 10:
                benign_reasons.append("no proximity to protected assets")
            if intent_sub < 15:
                benign_reasons.append("no hostile intent indicators")
            if anomaly_sub < 15:
                benign_reasons.append("behavior within normal parameters")
            if pattern_sub < 10:
                benign_reasons.append("no concerning behavioral patterns detected")
            if benign_reasons:
                parts.append(
                    f"Assessment basis: {'; '.join(benign_reasons)}."
                )

        # Sub-score breakdown (only mention significant contributors)
        breakdown = []
        if intent_sub >= 15:
            breakdown.append(f"intent={intent_sub:.0f}")
        if anomaly_sub >= 15:
            breakdown.append(f"anomaly={anomaly_sub:.0f}")
        if proximity_sub >= 10:
            breakdown.append(f"proximity={proximity_sub:.0f}")
        if pattern_sub >= 10:
            breakdown.append(f"pattern={pattern_sub:.0f}")
        if breakdown:
            parts.append(f"Sub-scores: {', '.join(breakdown)}.")

        # Contributing factors
        if factors:
            for f in factors:
                parts.append(f"- {f}")

        return "\n".join(parts)


def _intent_threat_reason(intent: str) -> str:
    """Return a plain-language explanation of why this intent is threatening."""
    reasons = {
        "RENDEZVOUS": (
            "This object is actively closing distance with a protected asset. "
            "Rendezvous approaches can indicate inspection, servicing, or hostile proximity operations."
        ),
        "SURVEILLANCE": (
            "This object is maintaining a persistent standoff position near a protected asset. "
            "Sustained co-orbital shadowing is a hallmark of intelligence-gathering behavior."
        ),
        "EVASIVE": (
            "This object performed maneuvers consistent with evading observation or tracking. "
            "Evasive behavior following an approach suggests deliberate concealment."
        ),
        "COLLISION_AVOIDANCE": (
            "This object is on a trajectory that could intersect with a protected asset. "
            "The closing geometry and velocity indicate a high-energy encounter."
        ),
    }
    return reasons.get(intent, "")
