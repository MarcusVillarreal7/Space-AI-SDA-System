"""
Unit tests for the Threat Scoring system.

Tests:
  - Sub-score computation (intent, anomaly, proximity, pattern)
  - Weighted combination and clamping
  - Threat tier assignment
  - Explanation generation
  - Edge cases (missing inputs, extremes)
  - Configurable weights
"""

import math
import pytest

from src.ml.intent.intent_classifier import (
    IntentCategory,
    IntentResult,
    ThreatLevel,
)
from src.ml.intent.proximity_context import ProximityContext
from src.ml.intent.asset_catalog import Asset, OrbitalRegime
from src.ml.anomaly.anomaly_detector import AnomalyResult
from src.ml.threat.threat_scorer import (
    ScoringWeights,
    ThreatScore,
    ThreatScorer,
    ThreatTier,
    _score_to_tier,
    _proximity_distance_score,
    _proximity_closing_score,
    _proximity_tca_score,
)
from src.ml.threat.threat_explainer import ThreatExplainer


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def scorer():
    return ThreatScorer()


@pytest.fixture
def iss_asset():
    return Asset(
        asset_id="ISS", name="International Space Station",
        regime=OrbitalRegime.LEO,
        position_km=(6771.0, 0.0, 0.0), altitude_km=400.0,
    )


def _make_proximity(
    asset=None, distance=1000.0, closing_rate=0.0,
    tca=float("inf"), approaching=False, regime_match=False,
):
    return ProximityContext(
        nearest_asset=asset,
        distance_km=distance,
        closing_rate_km_s=closing_rate,
        time_to_closest_approach_s=tca,
        is_approaching=approaching,
        regime_match=regime_match,
        object_regime=OrbitalRegime.LEO,
        assets_in_range=1 if asset else 0,
    )


def _make_intent(
    intent=IntentCategory.NOMINAL,
    threat=ThreatLevel.NONE,
    confidence=0.9,
    proximity=None,
    patterns=None,
):
    return IntentResult(
        intent=intent,
        threat_level=threat,
        confidence=confidence,
        proximity=proximity,
        escalation_patterns=patterns or [],
        explanation="Test intent",
    )


def _make_anomaly(
    object_id="SAT-01",
    score=0.01,
    threshold=0.13,
    is_anomaly=False,
    percentile=30.0,
):
    return AnomalyResult(
        object_id=object_id,
        anomaly_score=score,
        threshold=threshold,
        is_anomaly=is_anomaly,
        percentile=percentile,
        top_features=["maneuver_rate"],
        explanation="Test anomaly",
    )


# -----------------------------------------------------------------------
# Threat Tier Assignment
# -----------------------------------------------------------------------

class TestThreatTier:
    def test_minimal(self):
        assert _score_to_tier(0.0) == ThreatTier.MINIMAL
        assert _score_to_tier(19.9) == ThreatTier.MINIMAL

    def test_low(self):
        assert _score_to_tier(20.0) == ThreatTier.LOW
        assert _score_to_tier(39.9) == ThreatTier.LOW

    def test_moderate(self):
        assert _score_to_tier(40.0) == ThreatTier.MODERATE
        assert _score_to_tier(59.9) == ThreatTier.MODERATE

    def test_elevated(self):
        assert _score_to_tier(60.0) == ThreatTier.ELEVATED
        assert _score_to_tier(79.9) == ThreatTier.ELEVATED

    def test_critical(self):
        assert _score_to_tier(80.0) == ThreatTier.CRITICAL
        assert _score_to_tier(100.0) == ThreatTier.CRITICAL


# -----------------------------------------------------------------------
# Proximity Sub-score Functions
# -----------------------------------------------------------------------

class TestProximityFunctions:
    def test_distance_score_at_zero(self):
        assert _proximity_distance_score(0.0, 500.0) == 100.0

    def test_distance_score_at_warning_radius(self):
        score = _proximity_distance_score(500.0, 500.0)
        assert score < 10.0  # Should be very low at boundary

    def test_distance_score_beyond_warning(self):
        assert _proximity_distance_score(1000.0, 500.0) == 0.0

    def test_distance_score_monotonic(self):
        scores = [_proximity_distance_score(d, 500.0) for d in [10, 50, 100, 200, 400]]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]

    def test_closing_score_receding(self):
        assert _proximity_closing_score(1.0) == 0.0

    def test_closing_score_approaching(self):
        score = _proximity_closing_score(-5.0)
        assert score == pytest.approx(50.0)

    def test_closing_score_fast_approach(self):
        score = _proximity_closing_score(-10.0)
        assert score == 100.0

    def test_closing_score_clamped(self):
        score = _proximity_closing_score(-20.0)
        assert score == 100.0  # Clamped

    def test_tca_very_short(self):
        score = _proximity_tca_score(300.0)  # 5 minutes
        assert score == 100.0

    def test_tca_long(self):
        score = _proximity_tca_score(86400.0)  # 24 hours
        assert score == 0.0

    def test_tca_infinite(self):
        assert _proximity_tca_score(float("inf")) == 0.0

    def test_tca_moderate(self):
        score = _proximity_tca_score(3600.0)  # 1 hour
        assert 0 < score < 100


# -----------------------------------------------------------------------
# Scoring Weights
# -----------------------------------------------------------------------

class TestScoringWeights:
    def test_default_weights_sum(self):
        w = ScoringWeights()
        assert w.total() == pytest.approx(1.0)

    def test_custom_weights(self):
        w = ScoringWeights(intent=0.5, anomaly=0.3, proximity=0.1, pattern=0.1)
        assert w.total() == pytest.approx(1.0)


# -----------------------------------------------------------------------
# ThreatScorer — Integration
# -----------------------------------------------------------------------

class TestThreatScorer:
    def test_no_inputs_returns_zero(self, scorer):
        result = scorer.score("SAT-01")
        assert result.score == 0.0
        assert result.tier == ThreatTier.MINIMAL
        assert result.object_id == "SAT-01"

    def test_nominal_intent_low_score(self, scorer):
        intent = _make_intent(IntentCategory.NOMINAL, ThreatLevel.NONE)
        result = scorer.score("SAT-01", intent_result=intent)
        assert result.score < 20
        assert result.tier == ThreatTier.MINIMAL

    def test_high_threat_intent(self, scorer, iss_asset):
        proximity = _make_proximity(iss_asset, distance=50.0, closing_rate=-2.0,
                                     tca=60.0, approaching=True)
        intent = _make_intent(
            IntentCategory.RENDEZVOUS, ThreatLevel.HIGH,
            proximity=proximity,
        )
        result = scorer.score("SAT-01", intent_result=intent)
        # Intent=100 (HIGH+RENDEZVOUS), Proximity=63, no anomaly/pattern
        # Weighted: 0.35*100 + 0.25*63 = 50.8 → MODERATE
        assert result.score >= 40
        assert result.tier in (ThreatTier.MODERATE, ThreatTier.ELEVATED, ThreatTier.CRITICAL)

    def test_anomaly_only(self, scorer):
        anomaly = _make_anomaly(percentile=98.0, is_anomaly=True, score=5.0)
        result = scorer.score("SAT-01", anomaly_result=anomaly)
        # Anomaly contributes 0.25 * 98 = 24.5
        assert result.anomaly_score == pytest.approx(98.0)
        assert result.score > 20

    def test_combined_intent_and_anomaly(self, scorer, iss_asset):
        proximity = _make_proximity(iss_asset, distance=100.0, closing_rate=-1.0,
                                     tca=300.0, approaching=True)
        intent = _make_intent(
            IntentCategory.SURVEILLANCE, ThreatLevel.ELEVATED,
            proximity=proximity,
            patterns=["SHADOWING"],
        )
        anomaly = _make_anomaly(percentile=99.0, is_anomaly=True, score=20.0)
        result = scorer.score("SAT-01", intent_result=intent, anomaly_result=anomaly)
        # All components firing — should be high
        assert result.score >= 60
        assert result.tier in (ThreatTier.ELEVATED, ThreatTier.CRITICAL)
        assert len(result.contributing_factors) >= 2

    def test_pattern_scores(self, scorer):
        intent = _make_intent(
            IntentCategory.SURVEILLANCE, ThreatLevel.ELEVATED,
            patterns=["PHASING", "EVASION"],
        )
        result = scorer.score("SAT-01", intent_result=intent)
        assert result.pattern_score >= 40  # EVASION=50, PHASING=40, max wins

    def test_score_clamped_to_100(self):
        # All weights doubled — should still clamp to 100
        scorer = ThreatScorer(
            weights=ScoringWeights(intent=1.0, anomaly=1.0, proximity=1.0, pattern=1.0)
        )
        proximity = _make_proximity(
            Asset("X", "X", OrbitalRegime.LEO, (6771.0, 0.0, 0.0)),
            distance=1.0, closing_rate=-10.0, tca=60.0, approaching=True,
        )
        intent = _make_intent(
            IntentCategory.EVASIVE, ThreatLevel.HIGH,
            proximity=proximity, patterns=["SHADOWING"],
        )
        anomaly = _make_anomaly(percentile=100.0, is_anomaly=True)
        result = scorer.score("SAT-01", intent_result=intent, anomaly_result=anomaly)
        assert result.score <= 100.0

    def test_score_never_negative(self, scorer):
        result = scorer.score("SAT-01")
        assert result.score >= 0.0

    def test_custom_weights_change_score(self, iss_asset):
        proximity = _make_proximity(iss_asset, distance=50.0, closing_rate=-2.0,
                                     tca=300.0, approaching=True)
        intent = _make_intent(
            IntentCategory.RENDEZVOUS, ThreatLevel.HIGH,
            proximity=proximity,
        )
        anomaly = _make_anomaly(percentile=50.0)

        # Intent-heavy weights
        scorer_intent = ThreatScorer(
            weights=ScoringWeights(intent=0.7, anomaly=0.1, proximity=0.1, pattern=0.1)
        )
        # Anomaly-heavy weights
        scorer_anomaly = ThreatScorer(
            weights=ScoringWeights(intent=0.1, anomaly=0.7, proximity=0.1, pattern=0.1)
        )

        r1 = scorer_intent.score("SAT-01", intent_result=intent, anomaly_result=anomaly)
        r2 = scorer_anomaly.score("SAT-01", intent_result=intent, anomaly_result=anomaly)

        # Intent-heavy should score higher because intent is HIGH
        assert r1.score > r2.score

    def test_proximity_contributes_when_close(self, scorer, iss_asset):
        close = _make_proximity(iss_asset, distance=10.0, closing_rate=-3.0,
                                 tca=120.0, approaching=True)
        far = _make_proximity(iss_asset, distance=5000.0, closing_rate=0.0)

        intent_close = _make_intent(
            IntentCategory.ORBIT_RAISING, ThreatLevel.MODERATE,
            proximity=close,
        )
        intent_far = _make_intent(
            IntentCategory.ORBIT_RAISING, ThreatLevel.MODERATE,
            proximity=far,
        )

        r_close = scorer.score("SAT-01", intent_result=intent_close)
        r_far = scorer.score("SAT-01", intent_result=intent_far)

        assert r_close.proximity_score > r_far.proximity_score
        assert r_close.score > r_far.score

    def test_result_has_all_fields(self, scorer):
        intent = _make_intent(IntentCategory.NOMINAL, ThreatLevel.NONE)
        anomaly = _make_anomaly()
        result = scorer.score("SAT-42", intent_result=intent, anomaly_result=anomaly)

        assert isinstance(result, ThreatScore)
        assert isinstance(result.tier, ThreatTier)
        assert isinstance(result.score, float)
        assert isinstance(result.contributing_factors, list)
        assert len(result.explanation) > 0
        assert result.intent_result is intent
        assert result.anomaly_result is anomaly


# -----------------------------------------------------------------------
# Explainer
# -----------------------------------------------------------------------

class TestThreatExplainer:
    def test_minimal_threat(self):
        text = ThreatExplainer.explain(
            "SAT-01", score=5.0, tier=ThreatTier.MINIMAL,
            intent_sub=0, anomaly_sub=0, proximity_sub=0, pattern_sub=0,
            factors=[],
        )
        assert "SAT-01" in text
        assert "5.0" in text
        assert "MINIMAL" in text
        assert "No action" in text

    def test_critical_threat_has_factors(self):
        text = ThreatExplainer.explain(
            "SAT-99", score=85.0, tier=ThreatTier.CRITICAL,
            intent_sub=90, anomaly_sub=98, proximity_sub=80, pattern_sub=60,
            factors=[
                "Intent: SURVEILLANCE at HIGH (confidence 95%)",
                "Anomaly: score 20.0 (P99, threshold 0.13)",
                "Proximity: 50 km from ISS, closing at 2.00 km/s",
            ],
        )
        assert "CRITICAL" in text
        assert "Immediate response" in text
        assert "SURVEILLANCE" in text
        assert "ISS" in text

    def test_explanation_shows_sub_scores(self):
        text = ThreatExplainer.explain(
            "SAT-50", score=45.0, tier=ThreatTier.MODERATE,
            intent_sub=40, anomaly_sub=50, proximity_sub=30, pattern_sub=0,
            factors=["Intent: ORBIT_RAISING at MODERATE"],
        )
        assert "intent=40" in text
        assert "anomaly=50" in text
        assert "proximity=30" in text


# -----------------------------------------------------------------------
# Scenario Tests
# -----------------------------------------------------------------------

class TestScenarios:
    """End-to-end scenarios testing realistic threat situations."""

    def test_normal_satellite(self, scorer):
        """A satellite in normal operations should score MINIMAL."""
        intent = _make_intent(IntentCategory.NOMINAL, ThreatLevel.NONE)
        anomaly = _make_anomaly(percentile=30.0, is_anomaly=False)
        result = scorer.score("SAT-NORMAL", intent_result=intent, anomaly_result=anomaly)
        assert result.tier == ThreatTier.MINIMAL
        assert result.score < 20

    def test_station_keeping_near_asset(self, scorer, iss_asset):
        """Station-keeping near ISS with no anomaly — MODERATE at most."""
        proximity = _make_proximity(iss_asset, distance=200.0, closing_rate=0.0,
                                     regime_match=True)
        intent = _make_intent(
            IntentCategory.STATION_KEEPING, ThreatLevel.NONE,
            proximity=proximity,
        )
        anomaly = _make_anomaly(percentile=40.0, is_anomaly=False)
        result = scorer.score("SAT-SK", intent_result=intent, anomaly_result=anomaly)
        assert result.score < 40

    def test_surveillance_with_anomaly(self, scorer, iss_asset):
        """Surveillance near asset + anomalous behavior — ELEVATED or CRITICAL."""
        proximity = _make_proximity(iss_asset, distance=80.0, closing_rate=-0.5,
                                     tca=600.0, approaching=True, regime_match=True)
        intent = _make_intent(
            IntentCategory.SURVEILLANCE, ThreatLevel.HIGH,
            proximity=proximity,
            patterns=["SHADOWING"],
        )
        anomaly = _make_anomaly(percentile=99.0, is_anomaly=True, score=15.0)
        result = scorer.score("SAT-SPY", intent_result=intent, anomaly_result=anomaly)
        assert result.tier in (ThreatTier.ELEVATED, ThreatTier.CRITICAL)
        assert result.score >= 60

    def test_fast_approach_unknown_intent(self, scorer, iss_asset):
        """Unknown object rapidly approaching ISS — should be high threat."""
        proximity = _make_proximity(iss_asset, distance=20.0, closing_rate=-8.0,
                                     tca=120.0, approaching=True)
        intent = _make_intent(
            IntentCategory.COLLISION_AVOIDANCE, ThreatLevel.HIGH,
            proximity=proximity,
        )
        result = scorer.score("DEBRIS-X", intent_result=intent)
        assert result.score >= 50
        assert result.tier in (ThreatTier.MODERATE, ThreatTier.ELEVATED, ThreatTier.CRITICAL)

    def test_deorbiting_satellite(self, scorer):
        """Deorbiting satellite far from assets — LOW threat."""
        intent = _make_intent(IntentCategory.DEORBIT, ThreatLevel.LOW)
        result = scorer.score("SAT-EOL", intent_result=intent)
        assert result.score < 30
        assert result.tier in (ThreatTier.MINIMAL, ThreatTier.LOW)
