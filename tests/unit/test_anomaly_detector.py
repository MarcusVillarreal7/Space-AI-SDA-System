"""
Unit tests for the Anomaly Detection module.

Tests:
  - Behavior feature extraction (19D vector)
  - Autoencoder architecture and forward pass
  - Anomaly detector fit/score/detect pipeline
  - Explainability output
  - Persistence (save/load round-trip)
  - Edge cases
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.ml.anomaly.behavior_features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    BehaviorFeatureExtractor,
    BehaviorProfile,
    ManeuverRecord,
)
from src.ml.anomaly.autoencoder import AutoencoderConfig, BehaviorAutoencoder
from src.ml.anomaly.anomaly_detector import AnomalyDetector, AnomalyResult
from src.ml.anomaly.explainer import AnomalyExplainer


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

def _make_maneuvers(classes, start_t=0.0, dt=3600.0):
    """Helper: build ManeuverRecord list from class indices."""
    return [
        ManeuverRecord(
            timestamp=start_t + i * dt,
            maneuver_class=c,
            delta_v_magnitude=0.05 if c in {3, 4, 5} else 0.0,
        )
        for i, c in enumerate(classes)
    ]


@pytest.fixture
def extractor():
    return BehaviorFeatureExtractor()


@pytest.fixture
def normal_profiles(extractor):
    """Generate 100 'normal' profiles with minor random variation."""
    rng = np.random.RandomState(42)
    profiles = []
    for i in range(100):
        maneuvers = _make_maneuvers(
            [0, 0, 0, 2, 0, 0, 2, 0],  # Mostly normal + station-keeping
            dt=3600.0 + rng.randn() * 100,
        )
        pos = (6800.0 + rng.randn() * 10, rng.randn() * 10, rng.randn() * 10)
        vel = (rng.randn() * 0.1, 7.5 + rng.randn() * 0.1, rng.randn() * 0.1)
        profiles.append(
            extractor.extract(f"SAT-{i:03d}", maneuvers, pos, vel)
        )
    return profiles


@pytest.fixture
def anomalous_profile(extractor):
    """A clearly anomalous profile: rapid major maneuvers, high delta-V."""
    maneuvers = _make_maneuvers(
        [4, 4, 4, 4, 4, 4, 4, 4, 3, 3],  # All major + minor maneuvers
        dt=600.0,  # Every 10 minutes
    )
    for m in maneuvers:
        m.delta_v_magnitude = 2.5  # Very high
    return extractor.extract(
        "ANOMALY-01", maneuvers,
        position_km=(42164.0, 0.0, 0.0),  # GEO
        velocity_km_s=(0.0, 3.07, 0.0),
    )


@pytest.fixture
def fitted_detector(normal_profiles):
    """An AnomalyDetector fitted on normal profiles."""
    detector = AnomalyDetector(device="cpu")
    detector.fit(normal_profiles, epochs=30, batch_size=32, lr=1e-3)
    return detector


# -----------------------------------------------------------------------
# Behavior Features
# -----------------------------------------------------------------------

class TestBehaviorFeatures:
    def test_feature_dim_is_19(self):
        assert FEATURE_DIM == 19
        assert len(FEATURE_NAMES) == 19

    def test_extract_returns_correct_shape(self, extractor):
        maneuvers = _make_maneuvers([0, 2, 3, 0, 4])
        profile = extractor.extract("TEST", maneuvers, (6800.0, 0.0, 0.0), (0.0, 7.5, 0.0))
        assert profile.features.shape == (19,)
        assert profile.features.dtype == np.float32

    def test_extract_metadata(self, extractor):
        maneuvers = _make_maneuvers([0, 2, 3], dt=3600.0)
        profile = extractor.extract("SAT-X", maneuvers, (6800.0, 0.0, 0.0), (0.0, 7.5, 0.0))
        assert profile.object_id == "SAT-X"
        assert profile.num_observations == 3
        assert profile.observation_window_s == pytest.approx(7200.0)

    def test_empty_maneuvers(self, extractor):
        profile = extractor.extract("EMPTY", [], (6800.0, 0.0, 0.0), (0.0, 7.5, 0.0))
        assert profile.features.shape == (19,)
        assert profile.num_observations == 0
        assert profile.features[0] == 0.0  # maneuver count

    def test_maneuver_count_correct(self, extractor):
        # 2 active maneuvers (class 3 and 4)
        maneuvers = _make_maneuvers([0, 0, 3, 0, 4])
        profile = extractor.extract("T", maneuvers, (6800.0, 0.0, 0.0), (0.0, 7.5, 0.0))
        assert profile.features[0] == 2.0  # maneuver_count

    def test_maneuver_rate_scales_with_frequency(self, extractor):
        slow = _make_maneuvers([3, 3], dt=86400.0)  # 1 per day
        fast = _make_maneuvers([3, 3], dt=3600.0)   # 1 per hour
        p_slow = extractor.extract("S", slow, (6800.0, 0.0, 0.0), (0.0, 7.5, 0.0))
        p_fast = extractor.extract("F", fast, (6800.0, 0.0, 0.0), (0.0, 7.5, 0.0))
        assert p_fast.features[1] > p_slow.features[1]  # faster rate

    def test_regime_encoding_leo(self, extractor):
        profile = extractor.extract("L", [], (6800.0, 0.0, 0.0), (0.0, 7.5, 0.0))
        assert profile.features[15] == 1.0  # LEO
        assert profile.features[16] == 0.0  # not MEO
        assert profile.features[17] == 0.0  # not GEO
        assert profile.features[18] == 0.0  # not HEO

    def test_regime_encoding_geo(self, extractor):
        profile = extractor.extract("G", [], (42164.0, 0.0, 0.0), (0.0, 3.07, 0.0))
        assert profile.features[15] == 0.0  # not LEO
        assert profile.features[17] == 1.0  # GEO

    def test_altitude_feature(self, extractor):
        leo = extractor.extract("L", [], (6800.0, 0.0, 0.0), (0.0, 7.5, 0.0))
        geo = extractor.extract("G", [], (42164.0, 0.0, 0.0), (0.0, 3.07, 0.0))
        assert geo.features[7] > leo.features[7]  # GEO altitude > LEO altitude

    def test_speed_feature(self, extractor):
        profile = extractor.extract("V", [], (6800.0, 0.0, 0.0), (0.0, 7.5, 0.0))
        assert profile.features[10] == pytest.approx(7.5, abs=0.01)

    def test_classification_entropy_all_same(self, extractor):
        maneuvers = _make_maneuvers([0, 0, 0, 0, 0])
        profile = extractor.extract("U", maneuvers, (6800.0, 0.0, 0.0), (0.0, 7.5, 0.0))
        assert profile.features[13] == pytest.approx(0.0, abs=0.01)  # zero entropy

    def test_classification_entropy_diverse(self, extractor):
        maneuvers = _make_maneuvers([0, 1, 2, 3, 4, 5])
        profile = extractor.extract("D", maneuvers, (6800.0, 0.0, 0.0), (0.0, 7.5, 0.0))
        assert profile.features[13] > 0.9  # high normalized entropy

    def test_timing_regularity_perfect(self, extractor):
        maneuvers = _make_maneuvers([3, 3, 3, 3], dt=3600.0)
        profile = extractor.extract("R", maneuvers, (6800.0, 0.0, 0.0), (0.0, 7.5, 0.0))
        assert profile.features[6] == pytest.approx(1.0, abs=0.01)  # perfect regularity

    def test_dominant_class_fraction(self, extractor):
        # 4 out of 5 are class 0
        maneuvers = _make_maneuvers([0, 0, 0, 0, 3])
        profile = extractor.extract("D", maneuvers, (6800.0, 0.0, 0.0), (0.0, 7.5, 0.0))
        assert profile.features[12] == pytest.approx(0.8, abs=0.01)


# -----------------------------------------------------------------------
# Autoencoder
# -----------------------------------------------------------------------

class TestAutoencoder:
    def test_default_config(self):
        config = AutoencoderConfig()
        assert config.input_dim == 19
        assert config.latent_dim == 6

    def test_forward_shape(self):
        model = BehaviorAutoencoder()
        x = torch.randn(8, 19)
        recon, latent = model(x)
        assert recon.shape == (8, 19)
        assert latent.shape == (8, 6)

    def test_encode_shape(self):
        model = BehaviorAutoencoder()
        x = torch.randn(4, 19)
        latent = model.encode(x)
        assert latent.shape == (4, 6)

    def test_reconstruction_error_shape(self):
        model = BehaviorAutoencoder()
        x = torch.randn(4, 19)
        err = model.reconstruction_error(x)
        assert err.shape == (4,)
        assert (err >= 0).all()

    def test_param_count(self):
        model = BehaviorAutoencoder()
        count = model.param_count()
        # 19*32+32 + 32*16+16 + 16*6+6 + 6*16+16 + 16*32+32 + 32*19+19
        # Plus LayerNorm params: 2*32 + 2*16 + 2*16 + 2*32
        assert count > 0
        assert count < 10000  # Should be small

    def test_config_round_trip(self):
        config = AutoencoderConfig(latent_dim=8, dropout=0.2)
        d = config.to_dict()
        config2 = AutoencoderConfig.from_dict(d)
        assert config2.latent_dim == 8
        assert config2.dropout == 0.2

    def test_from_config(self):
        config_dict = {"input_dim": 19, "hidden_dims": [32, 16], "latent_dim": 6}
        model = BehaviorAutoencoder.from_config(config_dict)
        x = torch.randn(2, 19)
        recon, _ = model(x)
        assert recon.shape == (2, 19)

    def test_custom_architecture(self):
        config = AutoencoderConfig(hidden_dims=(64, 32), latent_dim=8)
        model = BehaviorAutoencoder(config)
        x = torch.randn(4, 19)
        recon, latent = model(x)
        assert recon.shape == (4, 19)
        assert latent.shape == (4, 8)


# -----------------------------------------------------------------------
# Anomaly Detector
# -----------------------------------------------------------------------

class TestAnomalyDetector:
    def test_unfitted_raises(self):
        detector = AnomalyDetector(device="cpu")
        profile = BehaviorProfile("X", np.zeros(19, dtype=np.float32))
        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.score(profile)

    def test_fit_returns_metrics(self, normal_profiles):
        detector = AnomalyDetector(device="cpu")
        metrics = detector.fit(normal_profiles, epochs=10, batch_size=32)
        assert "final_loss" in metrics
        assert "threshold" in metrics
        assert "param_count" in metrics
        assert metrics["n_training"] == 100
        assert metrics["threshold"] > 0

    def test_threshold_is_positive(self, fitted_detector):
        assert fitted_detector.threshold > 0
        assert fitted_detector._fitted

    def test_normal_score_below_threshold(self, fitted_detector, normal_profiles):
        # Most normal profiles should score below threshold
        scores = fitted_detector.score_batch(normal_profiles)
        below = (scores <= fitted_detector.threshold).sum()
        assert below >= 90  # At least 90% should be below threshold (95th percentile)

    def test_anomaly_score_higher(self, fitted_detector, normal_profiles, anomalous_profile):
        normal_score = fitted_detector.score(normal_profiles[0])
        anomaly_score = fitted_detector.score(anomalous_profile)
        assert anomaly_score > normal_score

    def test_detect_returns_anomaly_result(self, fitted_detector, anomalous_profile):
        result = fitted_detector.detect(anomalous_profile)
        assert isinstance(result, AnomalyResult)
        assert result.object_id == "ANOMALY-01"
        assert result.anomaly_score > 0
        assert result.threshold > 0
        assert isinstance(result.is_anomaly, bool)
        assert 0 <= result.percentile <= 100
        assert len(result.top_features) == 3
        assert len(result.explanation) > 0

    def test_detect_normal_is_not_anomaly(self, fitted_detector, normal_profiles):
        result = fitted_detector.detect(normal_profiles[0])
        # Should usually be normal (not guaranteed, but very likely)
        assert result.anomaly_score <= fitted_detector.threshold * 2

    def test_detect_anomaly_flagged(self, fitted_detector, anomalous_profile):
        result = fitted_detector.detect(anomalous_profile)
        assert result.is_anomaly is True
        assert result.percentile > 90

    def test_score_batch_shape(self, fitted_detector, normal_profiles):
        scores = fitted_detector.score_batch(normal_profiles[:10])
        assert scores.shape == (10,)
        assert all(s >= 0 for s in scores)


# -----------------------------------------------------------------------
# Explainer
# -----------------------------------------------------------------------

class TestExplainer:
    def test_top_features_returns_correct_count(self):
        explainer = AnomalyExplainer()
        errors = np.random.rand(19)
        top = explainer.top_features(errors, n=3)
        assert len(top) == 3
        assert all(f in FEATURE_NAMES for f in top)

    def test_top_features_order(self):
        explainer = AnomalyExplainer()
        errors = np.zeros(19)
        errors[7] = 10.0   # altitude_km
        errors[0] = 5.0    # maneuver_count
        top = explainer.top_features(errors, n=2)
        assert top[0] == "altitude_km"
        assert top[1] == "maneuver_count"

    def test_explain_anomaly(self):
        explainer = AnomalyExplainer()
        profile = BehaviorProfile("SAT-99", np.zeros(19), observation_window_s=7200, num_observations=10)
        text = explainer.explain(
            anomaly_score=0.5, threshold=0.1,
            is_anomaly=True, percentile=99.5,
            per_feature_error=np.random.rand(19),
            profile=profile,
        )
        assert "ANOMALOUS" in text
        assert "SAT-99" in text
        assert "10 observations" in text

    def test_explain_normal(self):
        explainer = AnomalyExplainer()
        profile = BehaviorProfile("SAT-01", np.zeros(19))
        text = explainer.explain(
            anomaly_score=0.01, threshold=0.1,
            is_anomaly=False, percentile=30.0,
            per_feature_error=np.zeros(19),
            profile=profile,
        )
        assert "NORMAL" in text
        assert "expected parameters" in text


# -----------------------------------------------------------------------
# Persistence (save/load round-trip)
# -----------------------------------------------------------------------

class TestPersistence:
    def test_save_load_round_trip(self, fitted_detector, normal_profiles):
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_detector.save(tmpdir)

            loaded = AnomalyDetector.load(tmpdir, device="cpu")
            assert loaded._fitted
            assert loaded.threshold == pytest.approx(fitted_detector.threshold)

            # Scores should match
            original_score = fitted_detector.score(normal_profiles[0])
            loaded_score = loaded.score(normal_profiles[0])
            assert loaded_score == pytest.approx(original_score, abs=1e-5)

    def test_save_creates_files(self, fitted_detector):
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_detector.save(tmpdir)
            assert (Path(tmpdir) / "autoencoder.pt").exists()
            assert (Path(tmpdir) / "anomaly_meta.json").exists()
            assert (Path(tmpdir) / "training_errors.npy").exists()
