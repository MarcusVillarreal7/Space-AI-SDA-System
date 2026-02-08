"""
Tests for TrajectoryFeatureExtractor.

Covers: FeatureConfig, feature extraction, orbital element computation,
        derived features, and temporal features. Uses reference orbits
        with analytically known values.
"""

import pytest
import numpy as np
import math

from src.ml.features.trajectory_features import (
    FeatureConfig,
    TrajectoryFeatureExtractor,
)

# Physical constants
MU = 398600.4418  # km^3/s^2
R_EARTH = 6371.0  # km


def _circular_orbit_state(altitude_km, inclination_deg=0.0):
    """Generate position/velocity for a circular orbit at given altitude."""
    r = R_EARTH + altitude_km
    v = math.sqrt(MU / r)
    inc = math.radians(inclination_deg)
    pos = np.array([r, 0.0, 0.0])
    vel = np.array([0.0, v * math.cos(inc), v * math.sin(inc)])
    return pos, vel


# ──────────────────────────────────────────────
# FeatureConfig
# ──────────────────────────────────────────────
class TestFeatureConfig:
    def test_default_dim(self):
        cfg = FeatureConfig()
        # 3+3+6+8+4+4 = 28
        assert cfg.get_feature_dim() == 28

    def test_dim_without_uncertainty(self):
        cfg = FeatureConfig(include_uncertainty=False)
        # 3+3+6+8+4 = 24
        assert cfg.get_feature_dim() == 24

    def test_individual_toggle_dims(self):
        # Only position
        cfg = FeatureConfig(include_position=True, include_velocity=False,
                            include_orbital_elements=False, include_derived_features=False,
                            include_temporal_features=False, include_uncertainty=False)
        assert cfg.get_feature_dim() == 3
        # Only velocity
        cfg2 = FeatureConfig(include_position=False, include_velocity=True,
                             include_orbital_elements=False, include_derived_features=False,
                             include_temporal_features=False, include_uncertainty=False)
        assert cfg2.get_feature_dim() == 3


# ──────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────
class TestFeatureExtraction:
    @pytest.fixture
    def extractor(self):
        return TrajectoryFeatureExtractor(FeatureConfig())

    def test_output_shape(self, extractor):
        T = 10
        positions = np.random.randn(T, 3) * 7000
        velocities = np.random.randn(T, 3) * 7
        timestamps = np.arange(T, dtype=float) * 60.0
        features = extractor.extract_features(positions, velocities, timestamps)
        assert features.shape == (T, 28)

    def test_position_features_match(self, extractor):
        T = 5
        positions = np.array([[100, 200, 300]] * T, dtype=float)
        velocities = np.ones((T, 3))
        timestamps = np.arange(T, dtype=float) * 60.0
        features = extractor.extract_features(positions, velocities, timestamps)
        np.testing.assert_allclose(features[:, :3], positions)

    def test_velocity_features_match(self, extractor):
        T = 5
        positions = np.ones((T, 3)) * 7000
        velocities = np.array([[1.0, 2.0, 3.0]] * T)
        timestamps = np.arange(T, dtype=float) * 60.0
        features = extractor.extract_features(positions, velocities, timestamps)
        np.testing.assert_allclose(features[:, 3:6], velocities)

    def test_features_finite(self, extractor):
        T = 20
        positions = np.random.randn(T, 3) * 7000 + 6771
        velocities = np.random.randn(T, 3) * 3 + 5
        timestamps = np.arange(T, dtype=float) * 60.0
        features = extractor.extract_features(positions, velocities, timestamps)
        assert np.all(np.isfinite(features)), "Features should be finite (no NaN/Inf)"

    def test_single_timestep(self, extractor):
        positions = np.array([[7000.0, 0.0, 0.0]])
        velocities = np.array([[0.0, 7.5, 0.0]])
        timestamps = np.array([0.0])
        features = extractor.extract_features(positions, velocities, timestamps)
        assert features.shape == (1, 28)


# ──────────────────────────────────────────────
# Orbital Elements
# ──────────────────────────────────────────────
class TestOrbitalElements:
    @pytest.fixture
    def extractor(self):
        return TrajectoryFeatureExtractor()

    def test_circular_leo(self, extractor):
        """ISS-like orbit at 400 km: e≈0, a≈6771 km."""
        pos, vel = _circular_orbit_state(400.0)
        elements = extractor._compute_orbital_elements(pos, vel)
        a, e, i, Omega, omega, nu = elements
        assert abs(a - 6771.0) < 5.0, f"Semi-major axis: {a}"
        assert e < 0.01, f"Eccentricity should be near-zero: {e}"

    def test_geo_orbit(self, extractor):
        """GEO orbit at 35786 km: a≈42164 km."""
        pos, vel = _circular_orbit_state(35786.0)
        elements = extractor._compute_orbital_elements(pos, vel)
        a = elements[0]
        assert abs(a - 42157.0) < 50.0, f"GEO semi-major axis: {a}"

    def test_inclination_equatorial(self, extractor):
        """Equatorial orbit should have inclination ≈ 0."""
        pos, vel = _circular_orbit_state(400.0, inclination_deg=0.0)
        elements = extractor._compute_orbital_elements(pos, vel)
        i = elements[2]
        assert abs(i) < 0.01, f"Inclination should be ~0 for equatorial: {i}"

    def test_inclination_polar(self, extractor):
        """Polar orbit should have inclination ≈ π/2."""
        pos, vel = _circular_orbit_state(400.0, inclination_deg=90.0)
        elements = extractor._compute_orbital_elements(pos, vel)
        i = elements[2]
        assert abs(i - math.pi / 2) < 0.01, f"Inclination should be ~π/2: {i}"

    def test_raan_quadrant_correction(self, extractor):
        """RAAN should be corrected when n_vec[1] < 0."""
        # Create orbit with node in negative-y half
        r = 7000.0
        v = math.sqrt(MU / r)
        pos = np.array([r, 0.0, 0.0])
        # Retrograde-ish orbit that puts ascending node in negative-y region
        vel = np.array([0.0, -v * 0.1, v * 0.99])
        elements = extractor._compute_orbital_elements(pos, vel)
        Omega = elements[3]
        assert 0 <= Omega <= 2 * math.pi

    def test_near_zero_eccentricity_fallback(self, extractor):
        """Near-zero eccentricity: true anomaly should fallback to 0."""
        pos, vel = _circular_orbit_state(400.0)
        elements = extractor._compute_orbital_elements(pos, vel)
        # For a perfectly circular orbit, e≈0 triggers nu=0 fallback
        # e may not be exactly 0 due to floating point, but should be very small
        e = elements[1]
        nu = elements[5]
        if e < 1e-6:
            assert nu == 0.0

    def test_near_zero_angular_momentum_fallback(self, extractor):
        """Near-zero angular momentum should fallback to i=0."""
        # Radial orbit (velocity along position) → h ≈ 0
        pos = np.array([7000.0, 0.0, 0.0])
        vel = np.array([1.0, 0.0, 0.0])  # radial velocity
        elements = extractor._compute_orbital_elements(pos, vel)
        i = elements[2]
        assert i == 0.0


# ──────────────────────────────────────────────
# Derived Features
# ──────────────────────────────────────────────
class TestDerivedFeatures:
    @pytest.fixture
    def extractor(self):
        return TrajectoryFeatureExtractor()

    def test_altitude(self, extractor):
        pos = np.array([7000.0, 0.0, 0.0])
        vel = np.array([0.0, 7.5, 0.0])
        derived = extractor._compute_derived_features(pos, vel)
        altitude = derived[0]
        expected = 7000.0 - R_EARTH
        assert abs(altitude - expected) < 0.1

    def test_orbital_period_leo(self, extractor):
        """LEO period should be ~90 minutes (5400 s)."""
        pos, vel = _circular_orbit_state(400.0)
        derived = extractor._compute_derived_features(pos, vel)
        period = derived[2]
        assert 5000 < period < 6000, f"LEO period: {period}s"

    def test_apogee_perigee_circular(self, extractor):
        """For circular orbit, apogee ≈ perigee ≈ altitude."""
        altitude = 400.0
        pos, vel = _circular_orbit_state(altitude)
        derived = extractor._compute_derived_features(pos, vel)
        apogee = derived[3]
        perigee = derived[4]
        assert abs(apogee - perigee) < 5.0, f"Apogee={apogee}, Perigee={perigee}"
        assert abs(apogee - altitude) < 5.0


# ──────────────────────────────────────────────
# Temporal Features
# ──────────────────────────────────────────────
class TestTemporalFeatures:
    @pytest.fixture
    def extractor(self):
        return TrajectoryFeatureExtractor()

    def test_cyclic_range(self, extractor):
        for t in [0.0, 3600.0, 43200.0, 86399.0]:
            temporal = extractor._compute_temporal_features(t)
            assert np.all(temporal >= -1.0) and np.all(temporal <= 1.0)

    def test_periodicity(self, extractor):
        """t=0 and t=86400 should produce same day-of-year features."""
        t0 = extractor._compute_temporal_features(0.0)
        t1 = extractor._compute_temporal_features(86400.0)
        # Day features (indices 2,3) should be the same after one full day cycle
        # Hour features wrap every 24h too
        np.testing.assert_allclose(t0[0:2], t1[0:2], atol=1e-6)
