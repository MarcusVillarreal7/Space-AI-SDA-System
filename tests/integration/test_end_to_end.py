"""
End-to-End Integration Tests for the Threat Assessment Pipeline.

Tests the full chain: raw track data → maneuver derivation → intent
classification → anomaly detection → threat scoring → ThreatAssessment.

Test categories:
  - Pipeline construction (with and without anomaly checkpoint)
  - Single-object assessment for each scenario type
  - Batch assessment
  - Output structure and field validation
  - Latency requirements
"""

import numpy as np
import pytest

from src.ml.threat_assessment import ThreatAssessmentPipeline, ThreatAssessment
from src.ml.intent.intent_classifier import IntentCategory, ThreatLevel
from src.ml.threat.threat_scorer import ThreatTier


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_leo_track(n_timesteps=100, dt=60.0, seed=42):
    """Generate a normal LEO satellite track (circular orbit ~400km)."""
    rng = np.random.RandomState(seed)
    r = 6771.0  # km (LEO)
    omega = 7.66 / r  # angular velocity rad/s

    timestamps = np.arange(n_timesteps) * dt
    theta = omega * timestamps

    positions = np.column_stack([
        r * np.cos(theta) + rng.randn(n_timesteps) * 0.01,
        r * np.sin(theta) + rng.randn(n_timesteps) * 0.01,
        rng.randn(n_timesteps) * 0.01,
    ])
    velocities = np.column_stack([
        -7.66 * np.sin(theta) + rng.randn(n_timesteps) * 0.001,
        7.66 * np.cos(theta) + rng.randn(n_timesteps) * 0.001,
        rng.randn(n_timesteps) * 0.001,
    ])
    return positions, velocities, timestamps


def _make_maneuvering_track(n_timesteps=100, dt=60.0, seed=42):
    """Generate a track with a large maneuver at the midpoint."""
    positions, velocities, timestamps = _make_leo_track(n_timesteps, dt, seed)

    # Inject a large velocity change at the midpoint
    mid = n_timesteps // 2
    velocities[mid:, 0] += 0.5  # 0.5 km/s delta-V (major maneuver)
    # Propagate the position change
    for i in range(mid, n_timesteps):
        positions[i] = positions[i - 1] + velocities[i] * dt

    return positions, velocities, timestamps


def _make_approaching_iss_track(n_timesteps=100, dt=60.0, seed=42):
    """Generate a track approaching ISS (6771, 0, 0) with closing velocity."""
    rng = np.random.RandomState(seed)
    timestamps = np.arange(n_timesteps) * dt

    # Start 200km away from ISS, approach linearly
    start_r = 6971.0
    end_r = 6790.0
    r_values = np.linspace(start_r, end_r, n_timesteps)

    positions = np.column_stack([
        r_values + rng.randn(n_timesteps) * 0.01,
        np.zeros(n_timesteps),
        np.zeros(n_timesteps),
    ])
    # Approaching at ~0.03 km/s radially + orbital velocity
    velocities = np.column_stack([
        np.full(n_timesteps, -0.03) + rng.randn(n_timesteps) * 0.001,
        np.full(n_timesteps, 7.5),
        np.zeros(n_timesteps),
    ])
    return positions, velocities, timestamps


def _make_geo_track(n_timesteps=100, dt=60.0, seed=42):
    """Generate a GEO satellite track at 45° longitude (away from any asset)."""
    rng = np.random.RandomState(seed)
    r = 42164.0  # GEO radius km
    omega = 3.07 / r

    timestamps = np.arange(n_timesteps) * dt
    # Start at 45° longitude — far from all asset positions
    # (assets are at 0°, 90°, 180°, 270°)
    theta0 = np.pi / 4.0
    theta = theta0 + omega * timestamps

    positions = np.column_stack([
        r * np.cos(theta) + rng.randn(n_timesteps) * 0.01,
        r * np.sin(theta) + rng.randn(n_timesteps) * 0.01,
        rng.randn(n_timesteps) * 0.01,
    ])
    velocities = np.column_stack([
        -3.07 * np.sin(theta) + rng.randn(n_timesteps) * 0.0001,
        3.07 * np.cos(theta) + rng.randn(n_timesteps) * 0.0001,
        rng.randn(n_timesteps) * 0.0001,
    ])
    return positions, velocities, timestamps


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def pipeline():
    """Pipeline without anomaly detector (no checkpoint needed)."""
    return ThreatAssessmentPipeline(anomaly_checkpoint=None)


@pytest.fixture
def pipeline_with_anomaly():
    """Pipeline with anomaly detector loaded from checkpoint."""
    ckpt = "checkpoints/phase3_anomaly"
    from pathlib import Path
    if not Path(ckpt).exists():
        pytest.skip("Anomaly checkpoint not available")
    return ThreatAssessmentPipeline(anomaly_checkpoint=ckpt, device="cpu")


# -----------------------------------------------------------------------
# Pipeline Construction
# -----------------------------------------------------------------------

class TestPipelineConstruction:
    def test_creates_without_anomaly(self):
        p = ThreatAssessmentPipeline(anomaly_checkpoint=None)
        assert p.intent_classifier is not None
        assert p.threat_scorer is not None
        assert p.anomaly_detector is None

    def test_creates_with_anomaly_checkpoint(self, pipeline_with_anomaly):
        assert pipeline_with_anomaly.anomaly_detector is not None
        assert pipeline_with_anomaly.anomaly_detector._fitted

    def test_creates_with_nonexistent_checkpoint(self):
        p = ThreatAssessmentPipeline(anomaly_checkpoint="/nonexistent/path")
        assert p.anomaly_detector is None  # Gracefully skips


# -----------------------------------------------------------------------
# Single Object Assessment
# -----------------------------------------------------------------------

class TestSingleAssessment:
    def test_normal_leo_satellite(self, pipeline):
        pos, vel, ts = _make_leo_track()
        result = pipeline.assess("LEO-NORMAL", pos, vel, ts)

        assert isinstance(result, ThreatAssessment)
        assert result.object_id == "LEO-NORMAL"
        assert result.maneuver_class in range(6)
        assert 0.0 <= result.maneuver_confidence <= 1.0
        assert result.intent_result is not None
        assert result.threat_score is not None
        assert result.threat_score.score <= 100.0
        assert result.num_observations == 100
        assert result.latency_ms > 0

    def test_maneuvering_satellite(self, pipeline):
        pos, vel, ts = _make_maneuvering_track()
        result = pipeline.assess("LEO-MANEUVER", pos, vel, ts)

        assert result.object_id == "LEO-MANEUVER"
        # Should detect some maneuver activity
        assert result.intent_result is not None
        assert result.threat_score is not None

    def test_approaching_iss(self, pipeline):
        pos, vel, ts = _make_approaching_iss_track()
        result = pipeline.assess("APPROACH-ISS", pos, vel, ts)

        # Should have proximity context to ISS
        assert result.intent_result is not None
        if result.intent_result.proximity is not None:
            prox = result.intent_result.proximity
            assert prox.nearest_asset is not None
            # Should be relatively close to ISS
            assert prox.distance_km < 500

    def test_geo_satellite(self, pipeline):
        pos, vel, ts = _make_geo_track()
        result = pipeline.assess("GEO-SAT", pos, vel, ts)

        assert result.object_id == "GEO-SAT"
        assert result.threat_score.tier in (ThreatTier.MINIMAL, ThreatTier.LOW)

    def test_with_maneuver_override(self, pipeline):
        pos, vel, ts = _make_leo_track()
        result = pipeline.assess(
            "OVERRIDE", pos, vel, ts,
            maneuver_class_override=4,  # Major Maneuver
            confidence_override=0.95,
        )
        assert result.maneuver_class == 4
        assert result.maneuver_confidence == 0.95
        assert result.maneuver_name == "Major Maneuver"

    def test_short_track(self, pipeline):
        """Minimal track (2 observations) should still work."""
        pos = np.array([[6800.0, 0.0, 0.0], [6800.1, 0.0, 0.0]])
        vel = np.array([[0.0, 7.5, 0.0], [0.0, 7.5, 0.0]])
        ts = np.array([0.0, 60.0])
        result = pipeline.assess("SHORT", pos, vel, ts)
        assert isinstance(result, ThreatAssessment)
        assert result.num_observations == 2

    def test_single_observation(self, pipeline):
        """Single observation should not crash."""
        pos = np.array([[6800.0, 0.0, 0.0]])
        vel = np.array([[0.0, 7.5, 0.0]])
        ts = np.array([0.0])
        result = pipeline.assess("SINGLE", pos, vel, ts)
        assert result.num_observations == 1
        assert result.maneuver_class == 0  # Default to normal


# -----------------------------------------------------------------------
# Anomaly Integration
# -----------------------------------------------------------------------

class TestAnomalyIntegration:
    def test_normal_satellite_not_anomalous(self, pipeline_with_anomaly):
        pos, vel, ts = _make_leo_track()
        result = pipeline_with_anomaly.assess("NORMAL-ANOM", pos, vel, ts)
        assert result.anomaly_result is not None
        assert isinstance(result.anomaly_result.anomaly_score, float)
        assert result.anomaly_result.anomaly_score >= 0

    def test_anomaly_feeds_into_threat_score(self, pipeline_with_anomaly):
        pos, vel, ts = _make_leo_track()
        result = pipeline_with_anomaly.assess("ANOM-FEED", pos, vel, ts)
        # Anomaly sub-score should be populated
        assert result.threat_score.anomaly_score >= 0


# -----------------------------------------------------------------------
# Batch Assessment
# -----------------------------------------------------------------------

class TestBatchAssessment:
    def test_batch_multiple_objects(self, pipeline):
        objects = []
        for i, gen in enumerate([_make_leo_track, _make_geo_track, _make_maneuvering_track]):
            pos, vel, ts = gen(seed=i)
            objects.append({
                "object_id": f"BATCH-{i}",
                "positions": pos,
                "velocities": vel,
                "timestamps": ts,
            })

        results = pipeline.assess_batch(objects)
        assert len(results) == 3
        assert all(isinstance(r, ThreatAssessment) for r in results)
        assert [r.object_id for r in results] == ["BATCH-0", "BATCH-1", "BATCH-2"]

    def test_batch_empty_list(self, pipeline):
        results = pipeline.assess_batch([])
        assert results == []


# -----------------------------------------------------------------------
# Output Validation
# -----------------------------------------------------------------------

class TestOutputValidation:
    def test_all_fields_populated(self, pipeline):
        pos, vel, ts = _make_leo_track()
        result = pipeline.assess("FIELDS", pos, vel, ts)

        # Core fields
        assert isinstance(result.object_id, str)
        assert isinstance(result.maneuver_class, int)
        assert isinstance(result.maneuver_name, str)
        assert isinstance(result.maneuver_confidence, float)
        assert isinstance(result.latency_ms, float)
        assert isinstance(result.num_observations, int)
        assert isinstance(result.observation_window_s, float)

        # Intent
        assert result.intent_result.intent in IntentCategory
        assert result.intent_result.threat_level in ThreatLevel

        # Threat score
        ts_result = result.threat_score
        assert 0.0 <= ts_result.score <= 100.0
        assert isinstance(ts_result.tier, ThreatTier)
        assert len(ts_result.explanation) > 0

    def test_observation_window_correct(self, pipeline):
        pos, vel, ts = _make_leo_track(n_timesteps=50, dt=120.0)
        result = pipeline.assess("WINDOW", pos, vel, ts)
        expected_window = 49 * 120.0  # (n-1) * dt
        assert result.observation_window_s == pytest.approx(expected_window)

    def test_latency_reasonable(self, pipeline):
        """Single assessment should complete in <100ms."""
        pos, vel, ts = _make_leo_track()
        result = pipeline.assess("LATENCY", pos, vel, ts)
        assert result.latency_ms < 100.0


# -----------------------------------------------------------------------
# Scenario Integration Tests
# -----------------------------------------------------------------------

class TestScenarios:
    def test_benign_satellite_low_threat(self, pipeline):
        """A satellite with constant velocity (no maneuvers) should score low."""
        rng = np.random.RandomState(42)
        n = 100
        ts = np.arange(n) * 60.0
        # Constant velocity, straight-line trajectory — no delta-V
        pos = np.column_stack([
            50000.0 + np.arange(n) * 0.01,  # Far from any asset
            np.zeros(n),
            np.zeros(n),
        ]) + rng.randn(n, 3) * 0.0001
        vel = np.column_stack([
            np.full(n, 0.01),
            np.full(n, 2.0),
            np.zeros(n),
        ]) + rng.randn(n, 3) * 0.0001
        result = pipeline.assess("BENIGN", pos, vel, ts)
        assert result.threat_score.score < 30
        assert result.threat_score.tier in (ThreatTier.MINIMAL, ThreatTier.LOW)

    def test_approaching_asset_elevated_threat(self, pipeline):
        """Object approaching ISS with maneuver override should elevate."""
        pos, vel, ts = _make_approaching_iss_track()
        result = pipeline.assess(
            "APPROACH-ISS", pos, vel, ts,
            maneuver_class_override=4,  # Major maneuver
            confidence_override=0.9,
        )
        # Should be at least MODERATE due to proximity + high threat intent
        assert result.threat_score.score >= 20

    def test_different_orbits_different_scores(self, pipeline):
        """LEO normal vs approaching ISS should produce different scores."""
        pos_leo, vel_leo, ts_leo = _make_leo_track()
        pos_app, vel_app, ts_app = _make_approaching_iss_track()

        r_leo = pipeline.assess("LEO", pos_leo, vel_leo, ts_leo)
        r_app = pipeline.assess(
            "APPROACHER", pos_app, vel_app, ts_app,
            maneuver_class_override=3,
            confidence_override=0.85,
        )

        # Approaching object should have higher threat
        assert r_app.threat_score.score > r_leo.threat_score.score
