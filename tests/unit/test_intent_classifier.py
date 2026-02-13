"""
Unit tests for the Intent Classification module.

Tests:
  - Base maneuver → intent mapping
  - Proximity context computation
  - Threat escalation patterns (phasing, shadowing, evasion)
  - Confidence adjustment
  - Explanation generation
  - Asset catalog queries
  - Edge cases
"""

import math
import pytest

from src.ml.intent.asset_catalog import Asset, AssetCatalog, OrbitalRegime
from src.ml.intent.proximity_context import (
    ProximityContext,
    classify_regime,
    compute_proximity,
)
from src.ml.intent.threat_escalation import ManeuverEvent, ThreatEscalator
from src.ml.intent.intent_classifier import (
    IntentCategory,
    IntentClassifier,
    IntentResult,
    ThreatLevel,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def catalog():
    return AssetCatalog()


@pytest.fixture
def classifier(catalog):
    return IntentClassifier(catalog=catalog, warning_radius_km=500.0)


@pytest.fixture
def escalator():
    return ThreatEscalator(
        phasing_window_s=86400.0,
        phasing_min_count=3,
        shadowing_duration_s=43200.0,
        shadowing_range_km=200.0,
    )


def _make_history(classes, start_t=0.0, dt=3600.0):
    """Helper: build ManeuverEvent list from class indices."""
    names = {0: "Normal", 1: "Drift/Decay", 2: "Station-keeping",
             3: "Minor Maneuver", 4: "Major Maneuver", 5: "Deorbit"}
    return [
        ManeuverEvent(
            timestamp=start_t + i * dt,
            maneuver_class=c,
            class_name=names.get(c, "Unknown"),
            delta_v_magnitude=0.01 if c in {3, 4} else 0.0,
        )
        for i, c in enumerate(classes)
    ]


# -----------------------------------------------------------------------
# Asset Catalog
# -----------------------------------------------------------------------

class TestAssetCatalog:
    def test_default_catalog_non_empty(self, catalog):
        assert len(catalog.all()) >= 4

    def test_by_regime(self, catalog):
        geo = catalog.by_regime(OrbitalRegime.GEO)
        assert len(geo) >= 2
        for a in geo:
            assert a.regime == OrbitalRegime.GEO

    def test_by_priority(self, catalog):
        top = catalog.by_priority(max_priority=1)
        assert all(a.priority <= 1 for a in top)

    def test_nearest(self, catalog):
        # Position near ISS orbit
        nearest = catalog.nearest((6800.0, 0.0, 0.0))
        assert nearest is not None
        assert nearest.asset_id == "ISS"

    def test_within_range(self, catalog):
        # Nothing should be within 1 km of origin
        results = catalog.within_range((0.0, 0.0, 0.0), 1.0)
        assert len(results) == 0

        # Everything should be within 100,000 km
        results = catalog.within_range((0.0, 0.0, 0.0), 100000.0)
        assert len(results) == len(catalog.all())

    def test_empty_catalog(self):
        empty = AssetCatalog(assets=[])
        assert empty.nearest((0.0, 0.0, 0.0)) is None
        assert empty.within_range((0.0, 0.0, 0.0), 1e6) == []


# -----------------------------------------------------------------------
# Proximity Context
# -----------------------------------------------------------------------

class TestProximityContext:
    def test_classify_regime_leo(self):
        assert classify_regime((6771.0, 0.0, 0.0)) == OrbitalRegime.LEO

    def test_classify_regime_meo(self):
        assert classify_regime((26560.0, 0.0, 0.0)) == OrbitalRegime.MEO

    def test_classify_regime_geo(self):
        assert classify_regime((42164.0, 0.0, 0.0)) == OrbitalRegime.GEO

    def test_approaching_object(self, catalog):
        # Object near ISS, moving toward it
        ctx = compute_proximity(
            position_km=(6900.0, 0.0, 0.0),
            velocity_km_s=(-1.0, 0.0, 0.0),  # moving toward ISS at 6771
            catalog=catalog,
        )
        assert ctx.nearest_asset.asset_id == "ISS"
        assert ctx.is_approaching
        assert ctx.closing_rate_km_s < 0
        assert ctx.distance_km == pytest.approx(129.0, abs=1.0)

    def test_receding_object(self, catalog):
        ctx = compute_proximity(
            position_km=(6900.0, 0.0, 0.0),
            velocity_km_s=(1.0, 0.0, 0.0),  # moving away from ISS
            catalog=catalog,
        )
        assert not ctx.is_approaching
        assert ctx.closing_rate_km_s > 0

    def test_tca_finite_when_approaching(self, catalog):
        ctx = compute_proximity(
            position_km=(6900.0, 0.0, 0.0),
            velocity_km_s=(-1.0, 0.0, 0.0),
            catalog=catalog,
        )
        assert ctx.time_to_closest_approach_s < float("inf")
        assert ctx.time_to_closest_approach_s > 0

    def test_tca_infinite_when_receding(self, catalog):
        ctx = compute_proximity(
            position_km=(6900.0, 0.0, 0.0),
            velocity_km_s=(1.0, 0.0, 0.0),
            catalog=catalog,
        )
        assert ctx.time_to_closest_approach_s == float("inf")

    def test_no_catalog(self):
        ctx = compute_proximity(
            position_km=(6900.0, 0.0, 0.0),
            velocity_km_s=(0.0, 7.5, 0.0),
            catalog=AssetCatalog(assets=[]),
        )
        assert ctx.nearest_asset is None
        assert ctx.distance_km == float("inf")


# -----------------------------------------------------------------------
# Threat Escalation
# -----------------------------------------------------------------------

class TestThreatEscalation:
    def test_phasing_detected(self, escalator):
        history = _make_history([3, 3, 3, 3])  # 4 minor maneuvers
        assert escalator.is_phasing_pattern(history)

    def test_phasing_not_detected_too_few(self, escalator):
        history = _make_history([3, 3])  # Only 2
        assert not escalator.is_phasing_pattern(history)

    def test_phasing_ignores_normal(self, escalator):
        history = _make_history([0, 0, 0, 0, 0])  # All normal
        assert not escalator.is_phasing_pattern(history)

    def test_shadowing_detected(self, escalator):
        # Station-keeping events near an asset
        history = _make_history(
            [2, 2, 2, 2, 2, 2, 2],
            dt=7200.0  # 7 events * 2h = 14h span > 12h * 0.5
        )
        asset = Asset(
            asset_id="TEST", name="Test", regime=OrbitalRegime.LEO,
            position_km=(6800.0, 0.0, 0.0), altitude_km=400.0,
        )
        proximity = ProximityContext(
            nearest_asset=asset, distance_km=100.0,  # within 200 km
            closing_rate_km_s=0.0, relative_speed_km_s=0.1,
            time_to_closest_approach_s=float("inf"),
            is_approaching=False, is_coorbital=True, regime_match=True,
            object_regime=OrbitalRegime.LEO, assets_in_range=1,
        )
        assert escalator.is_shadowing_pattern(history, proximity)

    def test_shadowing_not_detected_far(self, escalator):
        history = _make_history([2, 2, 2, 2], dt=7200.0)
        proximity = ProximityContext(
            nearest_asset=None, distance_km=5000.0,
            closing_rate_km_s=0.0, time_to_closest_approach_s=float("inf"),
            is_approaching=False, regime_match=False,
            object_regime=OrbitalRegime.LEO, assets_in_range=0,
        )
        assert not escalator.is_shadowing_pattern(history, proximity)

    def test_evasive_detected(self, escalator):
        # 5 normal then a major maneuver
        history = _make_history([0, 0, 0, 0, 0, 4])
        assert escalator.is_evasive_pattern(history)

    def test_evasive_not_detected_short(self, escalator):
        history = _make_history([0, 4])  # Too short
        assert not escalator.is_evasive_pattern(history)

    def test_detect_patterns_combined(self, escalator):
        # 5 normal + major (evasion) but also enough maneuvers for phasing
        history = _make_history([0, 0, 0, 0, 0, 4, 3, 3, 3])
        patterns = escalator.detect_patterns(history)
        assert "PHASING" in patterns


# -----------------------------------------------------------------------
# Intent Classifier
# -----------------------------------------------------------------------

class TestIntentClassifier:
    def test_normal_class_no_threat(self, classifier):
        result = classifier.classify(
            maneuver_class=0, confidence=0.9,
            position_km=(6800.0, 0.0, 0.0),
            velocity_km_s=(0.0, 7.5, 0.0),
        )
        assert result.intent == IntentCategory.NOMINAL
        assert result.threat_level == ThreatLevel.NONE

    def test_station_keeping_base(self, classifier):
        result = classifier.classify(
            maneuver_class=2, confidence=0.85,
            position_km=(50000.0, 0.0, 0.0),  # Far from any asset
            velocity_km_s=(0.0, 2.5, 0.0),
        )
        assert result.intent == IntentCategory.STATION_KEEPING
        assert result.threat_level == ThreatLevel.NONE

    def test_deorbit(self, classifier):
        result = classifier.classify(
            maneuver_class=5, confidence=0.8,
            position_km=(6800.0, 0.0, 0.0),
            velocity_km_s=(0.0, 7.5, 0.0),
        )
        assert result.intent == IntentCategory.DEORBIT
        assert result.threat_level == ThreatLevel.LOW

    def test_minor_maneuver_approaching_asset_escalates(self, classifier):
        # Near ISS (6771, 0, 0) and approaching
        result = classifier.classify(
            maneuver_class=3, confidence=0.9,
            position_km=(6900.0, 0.0, 0.0),
            velocity_km_s=(-1.0, 0.0, 0.0),
        )
        assert result.intent == IntentCategory.RENDEZVOUS
        assert result.threat_level.value >= ThreatLevel.ELEVATED.value

    def test_major_maneuver_very_close_is_high_threat(self, classifier):
        # Very close to ISS and approaching fast
        classifier_close = IntentClassifier(
            catalog=classifier.catalog,
            approach_threshold_km=200.0,
        )
        result = classifier_close.classify(
            maneuver_class=4, confidence=0.95,
            position_km=(6800.0, 0.0, 0.0),  # ~29 km from ISS
            velocity_km_s=(-0.5, 0.0, 0.0),
        )
        assert result.threat_level == ThreatLevel.HIGH

    def test_low_confidence_reduces_threat(self, classifier):
        result = classifier.classify(
            maneuver_class=4, confidence=0.3,  # Low confidence
            position_km=(50000.0, 0.0, 0.0),
            velocity_km_s=(0.0, 2.5, 0.0),
        )
        # Base threat MODERATE - 1 = LOW due to low confidence
        assert result.threat_level.value < ThreatLevel.MODERATE.value

    def test_phasing_escalation(self, classifier):
        history = _make_history([3, 3, 3, 3])
        result = classifier.classify(
            maneuver_class=3, confidence=0.85,
            position_km=(6900.0, 0.0, 0.0),
            velocity_km_s=(0.0, 7.5, 0.0),
            maneuver_history=history,
        )
        assert "PHASING" in result.escalation_patterns
        assert result.threat_level.value >= ThreatLevel.ELEVATED.value

    def test_explanation_contains_class_name(self, classifier):
        result = classifier.classify(
            maneuver_class=4, confidence=0.9,
            position_km=(6800.0, 0.0, 0.0),
            velocity_km_s=(0.0, 7.5, 0.0),
        )
        assert "Major Maneuver" in result.explanation
        assert "90%" in result.explanation

    def test_explanation_contains_asset(self, classifier):
        result = classifier.classify(
            maneuver_class=3, confidence=0.8,
            position_km=(6900.0, 0.0, 0.0),
            velocity_km_s=(-0.5, 0.0, 0.0),
        )
        assert "International Space Station" in result.explanation

    def test_unknown_maneuver_class(self, classifier):
        result = classifier.classify(
            maneuver_class=99, confidence=0.5,
            position_km=(20000.0, 0.0, 0.0),
            velocity_km_s=(0.0, 4.0, 0.0),
        )
        assert result.intent == IntentCategory.UNKNOWN

    def test_all_base_mappings(self, classifier):
        """Every maneuver class 0-5 should produce a valid IntentResult."""
        for cls_idx in range(6):
            result = classifier.classify(
                maneuver_class=cls_idx, confidence=0.8,
                position_km=(50000.0, 0.0, 0.0),
                velocity_km_s=(0.0, 2.0, 0.0),
            )
            assert isinstance(result, IntentResult)
            assert isinstance(result.intent, IntentCategory)
            assert isinstance(result.threat_level, ThreatLevel)
