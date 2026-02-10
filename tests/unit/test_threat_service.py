"""
Tests for ThreatService — assessment, tier summary, caching.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.api.threat_service import ThreatService


class TestThreatServiceDefaults:
    def test_default_assessment(self):
        result = ThreatService._default_assessment(42)
        assert result["object_id"] == 42
        assert result["threat_score"] == 0.0
        assert result["threat_tier"] == "MINIMAL"
        assert result["maneuver_class"] == "Normal"
        assert result["contributing_factors"] == []

    def test_tier_summary_empty(self):
        service = ThreatService()
        mock_catalog = MagicMock()
        mock_catalog.n_objects = 100
        with patch.dict("src.api.main.app_state", {"catalog": mock_catalog}):
            summary = service.get_tier_summary()
            assert "MINIMAL" in summary
            assert "CRITICAL" in summary
            assert summary["MINIMAL"] == 100  # All unassessed = MINIMAL

    def test_get_threat_tiers_empty(self):
        service = ThreatService()
        tiers = service.get_threat_tiers()
        assert tiers == {}


class TestThreatServiceAssessment:
    def test_assess_with_no_pipeline(self):
        service = ThreatService()
        service._pipeline = None
        service._pipeline_loaded = True  # Skip loading

        positions = np.random.randn(20, 3) * 7000
        velocities = np.random.randn(20, 3) * 7
        timestamps = np.arange(20, dtype=float) * 60

        # Mock cache functions
        with patch("src.api.threat_service.get_cached_assessment", return_value=None):
            with patch("src.api.threat_service.cache_assessment"):
                result = service.assess_object(
                    42, positions, velocities, timestamps, use_cache=False
                )
                assert result["object_id"] == 42
                assert result["threat_tier"] == "MINIMAL"

    def test_assess_uses_cache(self):
        service = ThreatService()
        cached = {
            "object_id": 42,
            "threat_score": 75.0,
            "threat_tier": "ELEVATED",
            "intent_score": 80.0,
            "anomaly_score": 60.0,
            "proximity_score": 30.0,
            "pattern_score": 20.0,
            "maneuver_class": "Major Maneuver",
            "maneuver_confidence": 0.9,
            "contributing_factors": ["test"],
            "explanation": "cached",
            "latency_ms": 5.0,
        }

        positions = np.random.randn(20, 3) * 7000
        velocities = np.random.randn(20, 3) * 7
        timestamps = np.arange(20, dtype=float) * 60

        with patch("src.api.threat_service.get_cached_assessment", return_value=cached):
            result = service.assess_object(
                42, positions, velocities, timestamps, timestep=0
            )
            assert result["threat_score"] == 75.0
            assert result["threat_tier"] == "ELEVATED"

    def test_assess_skips_cache_when_disabled(self):
        service = ThreatService()
        service._pipeline = None
        service._pipeline_loaded = True

        positions = np.random.randn(20, 3) * 7000
        velocities = np.random.randn(20, 3) * 7
        timestamps = np.arange(20, dtype=float) * 60

        with patch("src.api.threat_service.get_cached_assessment") as mock_cache:
            with patch("src.api.threat_service.cache_assessment"):
                service.assess_object(
                    42, positions, velocities, timestamps, use_cache=False
                )
                mock_cache.assert_not_called()


PARQUET_PATH = Path("data/processed/ml_train_1k/ground_truth.parquet")


@pytest.mark.skipif(not PARQUET_PATH.exists(), reason="ground_truth.parquet not found")
class TestThreatServiceIntegration:
    def test_assess_real_object(self):
        """Test with real data and real pipeline (CPU)."""
        from src.api.data_manager import SpaceCatalog

        catalog = SpaceCatalog()
        catalog.load(PARQUET_PATH)

        service = ThreatService()
        data = catalog.get_positions_and_velocities(42)
        assert data is not None
        positions, velocities, timestamps = data

        with patch("src.api.threat_service.get_cached_assessment", return_value=None):
            with patch("src.api.threat_service.cache_assessment"):
                result = service.assess_object(
                    42, positions[-20:], velocities[-20:], timestamps[-20:],
                    use_cache=False,
                )

        assert result["object_id"] == 42
        assert 0 <= result["threat_score"] <= 100
        assert result["threat_tier"] in (
            "MINIMAL", "LOW", "MODERATE", "ELEVATED", "CRITICAL"
        )
        assert result["latency_ms"] > 0
        assert result["maneuver_class"] in (
            "Normal", "Drift/Decay", "Station-keeping",
            "Minor Maneuver", "Major Maneuver", "Deorbit",
        )
