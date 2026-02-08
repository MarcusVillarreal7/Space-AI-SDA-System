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

        return " ".join(parts) if len(parts) <= 3 else "\n".join(parts)
