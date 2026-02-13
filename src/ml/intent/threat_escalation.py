"""
Behavioral Pattern Detection and Threat Escalation.

Detects multi-step behavioral patterns (phasing, shadowing, evasion) that
escalate the base intent classification to a higher threat level.

Author: Space AI Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.ml.intent.proximity_context import ProximityContext


@dataclass
class ManeuverEvent:
    """A single maneuver observation used for pattern detection."""
    timestamp: float           # Epoch seconds
    maneuver_class: int        # 0-5 class index
    class_name: str
    delta_v_magnitude: float   # km/s (0 for non-maneuvers)
    position_km: tuple[float, float, float] = (0.0, 0.0, 0.0)


class ThreatEscalator:
    """
    Detects multi-step behavioral patterns that escalate threat level.

    Patterns detected:
      - Phasing: repeated minor maneuvers in the same direction
      - Shadowing: prolonged station-keeping near a high-value asset
      - Evasion: maneuver immediately following a sensor tasking window
    """

    def __init__(
        self,
        phasing_window_s: float = 86400.0,       # 24 h look-back
        phasing_min_count: int = 3,               # min maneuvers for phasing
        shadowing_duration_s: float = 43200.0,    # 12 h of station-keeping
        shadowing_range_km: float = 200.0,        # proximity threshold
    ):
        self.phasing_window_s = phasing_window_s
        self.phasing_min_count = phasing_min_count
        self.shadowing_duration_s = shadowing_duration_s
        self.shadowing_range_km = shadowing_range_km

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_patterns(
        self,
        history: List[ManeuverEvent],
        proximity: Optional[ProximityContext] = None,
    ) -> List[str]:
        """
        Return a list of detected pattern names.

        Args:
            history: Chronological list of recent ManeuverEvents.
            proximity: Current proximity context (may be None).

        Returns:
            List of pattern name strings, e.g. ["PHASING", "SHADOWING"].
        """
        patterns: List[str] = []

        if self.is_phasing_pattern(history):
            patterns.append("PHASING")

        if proximity is not None and self.is_shadowing_pattern(history, proximity):
            patterns.append("SHADOWING")

        if self.is_evasive_pattern(history):
            patterns.append("EVASION")

        return patterns

    # ------------------------------------------------------------------
    # Pattern detectors
    # ------------------------------------------------------------------

    def is_phasing_pattern(self, history: List[ManeuverEvent]) -> bool:
        """
        Detect phasing: ≥N minor/major maneuvers within a time window.

        Phasing orbits use a series of small burns to slowly change
        the phase angle relative to a target.
        """
        if len(history) < self.phasing_min_count:
            return False

        maneuver_classes = {3, 4}  # Minor Maneuver, Major Maneuver
        recent = _recent_events(history, self.phasing_window_s)
        maneuver_events = [e for e in recent if e.maneuver_class in maneuver_classes]

        return len(maneuver_events) >= self.phasing_min_count

    def is_shadowing_pattern(
        self, history: List[ManeuverEvent], proximity: ProximityContext
    ) -> bool:
        """
        Detect shadowing: sustained station-keeping near a high-value asset.
        Requires co-orbital relative speed — high-speed flybys are not shadowing.
        """
        if proximity.distance_km > self.shadowing_range_km:
            return False
        if not proximity.is_coorbital:
            return False

        sk_class = 2  # Station-keeping
        recent = _recent_events(history, self.shadowing_duration_s)
        sk_events = [e for e in recent if e.maneuver_class == sk_class]

        if len(sk_events) < 2:
            return False

        # Check duration span of station-keeping events
        span = sk_events[-1].timestamp - sk_events[0].timestamp
        return span >= self.shadowing_duration_s * 0.5

    def is_evasive_pattern(self, history: List[ManeuverEvent]) -> bool:
        """
        Detect evasion: a sudden significant maneuver after a prolonged
        period of normal behavior, suggesting a response to being observed.

        Scans the full history for any class 3/4/5 event preceded by ≥5
        consecutive normal events.
        """
        if len(history) < 6:
            return False

        for i in range(5, len(history)):
            if history[i].maneuver_class in {3, 4, 5}:
                preceding = history[i - 5:i]
                if all(e.maneuver_class == 0 for e in preceding):
                    return True
        return False


def _recent_events(
    events: List[ManeuverEvent], window_s: float
) -> List[ManeuverEvent]:
    """Filter events within the last *window_s* seconds."""
    if not events:
        return []
    latest = events[-1].timestamp
    cutoff = latest - window_s
    return [e for e in events if e.timestamp >= cutoff]
