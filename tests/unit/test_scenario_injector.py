"""
Tests for ScenarioInjector — validates threat scenario injection into catalog.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from pathlib import Path


# -----------------------------------------------------------------------
# Minimal mock catalog for testing
# -----------------------------------------------------------------------

class MockCatalog:
    """Lightweight catalog mock with enough structure for scenario injection."""

    def __init__(self, n_objects=1000, n_timesteps=1440):
        self.n_objects = n_objects
        self.n_timesteps = n_timesteps
        self.object_ids = np.arange(n_objects, dtype=int)
        self.object_names = [f"SAT-{i}" for i in range(n_objects)]
        self.object_types = ["PAYLOAD"] * n_objects
        self.regimes = ["LEO"] * n_objects
        self.positions = np.random.randn(n_objects, n_timesteps, 3) * 7000
        self.velocities = np.random.randn(n_objects, n_timesteps, 3) * 7
        self.ref_altitudes = np.random.uniform(200, 36000, n_objects)
        self.ref_speeds = np.random.uniform(3, 8, n_objects)
        self.latitudes = np.zeros((n_objects, n_timesteps))
        self.longitudes = np.zeros((n_objects, n_timesteps))
        self.altitudes = np.zeros((n_objects, n_timesteps))
        self.timestamps = [MagicMock() for _ in range(n_timesteps)]

    def get_object_index(self, object_id: int):
        matches = np.where(self.object_ids == object_id)[0]
        return int(matches[0]) if len(matches) > 0 else None


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestScenarioInjector:
    """Tests for the ScenarioInjector class."""

    def test_inject_modifies_seven_objects(self):
        """ScenarioInjector should modify objects 990-996."""
        from src.api.scenario_injector import ScenarioInjector
        catalog = MockCatalog()
        original_names = catalog.object_names[990:997].copy()

        with patch("src.api.scenario_injector.ScenarioInjector._recompute_geodetic"):
            injector = ScenarioInjector()
            modified = injector.inject(catalog)

        assert len(modified) == 7
        assert set(modified) == {990, 991, 992, 993, 994, 995, 996}

        # Names should be changed
        for i, oid in enumerate(range(990, 997)):
            idx = catalog.get_object_index(oid)
            assert catalog.object_names[idx] != original_names[i]

    def test_inject_renames_objects(self):
        """Injected objects should have adversary designations."""
        from src.api.scenario_injector import ScenarioInjector
        catalog = MockCatalog()

        with patch("src.api.scenario_injector.ScenarioInjector._recompute_geodetic"):
            injector = ScenarioInjector()
            injector.inject(catalog)

        expected_names = [
            "COSMOS-2558", "LUCH/OLYMP", "COSMOS-2542",
            "DEBRIS-KZ-1A", "SJ-17", "SHIJIAN-21", "OBJECT-2024-999A"
        ]
        for i, name in enumerate(expected_names):
            idx = catalog.get_object_index(990 + i)
            assert catalog.object_names[idx] == name

    def test_inject_sets_object_types(self):
        """Injected objects should have correct object types (993=DEBRIS, others=PAYLOAD)."""
        from src.api.scenario_injector import ScenarioInjector
        catalog = MockCatalog()

        with patch("src.api.scenario_injector.ScenarioInjector._recompute_geodetic"):
            injector = ScenarioInjector()
            injector.inject(catalog)

        for oid in range(990, 997):
            idx = catalog.get_object_index(oid)
            if oid == 993:
                assert catalog.object_types[idx] == "DEBRIS", \
                    f"Object 993 (DEBRIS-KZ-1A) should be DEBRIS"
            else:
                assert catalog.object_types[idx] == "PAYLOAD", \
                    f"Object {oid} should be PAYLOAD"

    def test_inject_positions_change(self):
        """Injected objects should have different position arrays."""
        from src.api.scenario_injector import ScenarioInjector
        catalog = MockCatalog()
        originals = {}
        for oid in range(990, 997):
            idx = catalog.get_object_index(oid)
            originals[oid] = catalog.positions[idx].copy()

        with patch("src.api.scenario_injector.ScenarioInjector._recompute_geodetic"):
            injector = ScenarioInjector()
            injector.inject(catalog)

        for oid in range(990, 997):
            idx = catalog.get_object_index(oid)
            assert not np.array_equal(catalog.positions[idx], originals[oid]), \
                f"Object {oid} positions should have changed"

    def test_inject_velocities_change(self):
        """Injected objects should have different velocity arrays."""
        from src.api.scenario_injector import ScenarioInjector
        catalog = MockCatalog()
        originals = {}
        for oid in range(990, 997):
            idx = catalog.get_object_index(oid)
            originals[oid] = catalog.velocities[idx].copy()

        with patch("src.api.scenario_injector.ScenarioInjector._recompute_geodetic"):
            injector = ScenarioInjector()
            injector.inject(catalog)

        for oid in range(990, 997):
            idx = catalog.get_object_index(oid)
            assert not np.array_equal(catalog.velocities[idx], originals[oid]), \
                f"Object {oid} velocities should have changed"

    def test_inject_positions_are_physical(self):
        """Injected positions should be physically reasonable (not NaN/Inf)."""
        from src.api.scenario_injector import ScenarioInjector
        catalog = MockCatalog()

        with patch("src.api.scenario_injector.ScenarioInjector._recompute_geodetic"):
            injector = ScenarioInjector()
            injector.inject(catalog)

        for oid in range(990, 997):
            idx = catalog.get_object_index(oid)
            pos = catalog.positions[idx]
            vel = catalog.velocities[idx]
            assert np.all(np.isfinite(pos)), f"Object {oid} has non-finite positions"
            assert np.all(np.isfinite(vel)), f"Object {oid} has non-finite velocities"

    def test_inject_altitudes_updated(self):
        """Reference altitudes should be updated for injected objects."""
        from src.api.scenario_injector import ScenarioInjector
        catalog = MockCatalog()
        # Set original altitudes to a known value
        for oid in range(990, 997):
            idx = catalog.get_object_index(oid)
            catalog.ref_altitudes[idx] = -1.0

        with patch("src.api.scenario_injector.ScenarioInjector._recompute_geodetic"):
            injector = ScenarioInjector()
            injector.inject(catalog)

        for oid in range(990, 997):
            idx = catalog.get_object_index(oid)
            assert catalog.ref_altitudes[idx] != -1.0, \
                f"Object {oid} altitude should be updated"

    def test_non_injected_objects_unchanged(self):
        """Objects outside 990-996 should not be modified."""
        from src.api.scenario_injector import ScenarioInjector
        catalog = MockCatalog()
        orig_pos_0 = catalog.positions[0].copy()
        orig_pos_500 = catalog.positions[500].copy()
        orig_pos_989 = catalog.positions[989].copy()

        with patch("src.api.scenario_injector.ScenarioInjector._recompute_geodetic"):
            injector = ScenarioInjector()
            injector.inject(catalog)

        assert np.array_equal(catalog.positions[0], orig_pos_0)
        assert np.array_equal(catalog.positions[500], orig_pos_500)
        assert np.array_equal(catalog.positions[989], orig_pos_989)

    def test_inject_handles_small_catalog(self):
        """If catalog has <997 objects, available objects are injected, missing are skipped."""
        from src.api.scenario_injector import ScenarioInjector
        catalog = MockCatalog(n_objects=995)  # Missing 995, 996

        with patch("src.api.scenario_injector.ScenarioInjector._recompute_geodetic"):
            injector = ScenarioInjector()
            modified = injector.inject(catalog)

        # Should only inject objects that exist (990-994)
        assert all(oid < 995 for oid in modified)
        assert len(modified) == 5

    def test_rendezvous_closes_distance(self):
        """COSMOS-2558 rendezvous should reduce offset from target over time."""
        from src.api.scenario_injector import ScenarioInjector, _propagate_kepler, MU, EARTH_RADIUS
        catalog = MockCatalog()

        with patch("src.api.scenario_injector.ScenarioInjector._recompute_geodetic"):
            injector = ScenarioInjector()
            injector.inject(catalog)

        idx = catalog.get_object_index(990)  # COSMOS-2558
        pos = catalog.positions[idx]

        # Propagate ISS to compare at same timesteps
        iss_r0 = np.array([6771.0, 0.0, 0.0])
        iss_v0 = np.array([0.0, 7.66, 0.0])
        iss_pos = np.zeros_like(pos)
        iss_pos[0] = iss_r0
        r, v = iss_r0, iss_v0
        for i in range(1, len(iss_pos)):
            r, v = _propagate_kepler(r, v, 60.0)
            iss_pos[i] = r

        dist_start = np.linalg.norm(pos[0] - iss_pos[0])
        dist_end = np.linalg.norm(pos[-1] - iss_pos[-1])
        # Should close significantly (at least 50% closer)
        assert dist_end < dist_start * 0.5, \
            f"Rendezvous should close distance (start={dist_start:.0f}, end={dist_end:.0f})"


class TestTrajectoryGeneration:
    """Tests for individual trajectory generation functions."""

    def test_generate_orbit_shape(self):
        from src.api.scenario_injector import _generate_orbit
        pos, vel = _generate_orbit(400, 51.6, 0, 100)
        assert pos.shape == (100, 3)
        assert vel.shape == (100, 3)

    def test_generate_orbit_altitude(self):
        from src.api.scenario_injector import _generate_orbit, EARTH_RADIUS
        pos, vel = _generate_orbit(400, 0, 0, 100)
        radii = np.linalg.norm(pos, axis=1)
        altitudes = radii - EARTH_RADIUS
        # Euler integration has energy drift; check the orbit stays reasonable
        assert np.all(altitudes > 100), f"Min altitude {altitudes.min():.0f} km too low"
        assert np.all(altitudes < 700), f"Max altitude {altitudes.max():.0f} km too high"
        # Mean altitude should be near 400 km
        assert abs(altitudes.mean() - 400) < 200

    def test_generate_orbit_velocity_magnitude(self):
        from src.api.scenario_injector import _generate_orbit, _circular_velocity, EARTH_RADIUS
        pos, vel = _generate_orbit(400, 0, 0, 10)
        v_expected = _circular_velocity(EARTH_RADIUS + 400)
        v_actual = np.linalg.norm(vel[0])
        assert abs(v_actual - v_expected) / v_expected < 0.01

    def test_propagate_kepler_conserves_energy(self):
        from src.api.scenario_injector import _propagate_kepler, MU
        r0 = np.array([7000.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.546, 0.0])
        E0 = 0.5 * np.dot(v0, v0) - MU / np.linalg.norm(r0)

        r, v = r0, v0
        for _ in range(100):
            r, v = _propagate_kepler(r, v, 60.0)
        E1 = 0.5 * np.dot(v, v) - MU / np.linalg.norm(r)

        # Euler integration drifts, but should be within ~5% for 100 steps
        assert abs(E1 - E0) / abs(E0) < 0.05

    def test_sudden_maneuver_changes_velocity(self):
        from src.api.scenario_injector import _inject_sudden_maneuver
        pos, vel = _inject_sudden_maneuver(600, 65, 1440, maneuver_timestep=100)
        # Velocity at maneuver point should differ from smooth trajectory
        # by ~0.5 km/s (the delta-V applied)
        dv = np.linalg.norm(vel[101] - vel[99])
        assert dv > 0.1  # Should see the maneuver effect


class TestScenarioDefinitions:
    """Tests for scenario metadata."""

    def test_all_scenarios_have_unique_ids(self):
        from src.api.scenario_injector import ScenarioInjector
        ids = [s.object_idx for s in ScenarioInjector.SCENARIOS]
        assert len(ids) == len(set(ids))

    def test_scenario_count(self):
        from src.api.scenario_injector import ScenarioInjector
        assert len(ScenarioInjector.SCENARIOS) == 7

    def test_expected_tiers(self):
        from src.api.scenario_injector import ScenarioInjector
        tiers = [s.expected_tier for s in ScenarioInjector.SCENARIOS]
        assert tiers.count("CRITICAL") == 2
        assert tiers.count("ELEVATED") == 5

    def test_all_scenarios_have_names(self):
        from src.api.scenario_injector import ScenarioInjector
        for s in ScenarioInjector.SCENARIOS:
            assert len(s.name) > 0
            assert len(s.description) > 0
