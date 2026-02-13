"""
Tests for SpaceCatalog — data loading, geodetic conversion, timestep lookup.
"""

import numpy as np
import pytest
from pathlib import Path

from src.api.data_manager import SpaceCatalog, _compute_gmst, _classify_regime


PARQUET_PATH = Path("data/processed/ml_train_1k/ground_truth.parquet")
PARQUET_EXISTS = PARQUET_PATH.exists()


class TestClassifyRegime:
    def test_leo(self):
        assert _classify_regime(400) == "LEO"
        assert _classify_regime(0) == "LEO"
        assert _classify_regime(1999) == "LEO"

    def test_meo(self):
        assert _classify_regime(2000) == "MEO"
        assert _classify_regime(20000) == "MEO"

    def test_geo(self):
        assert _classify_regime(35786) == "GEO"
        assert _classify_regime(35000) == "GEO"

    def test_heo(self):
        assert _classify_regime(36500) == "HEO"
        assert _classify_regime(100000) == "HEO"


class TestComputeGMST:
    def test_returns_float(self):
        import pandas as pd
        ts = pd.Timestamp("2026-02-07T12:00:00Z")
        result = _compute_gmst(ts)
        assert isinstance(result, float)

    def test_in_range(self):
        import pandas as pd
        ts = pd.Timestamp("2026-02-07T12:00:00Z")
        result = _compute_gmst(ts)
        assert 0 <= result < 2 * np.pi

    def test_varies_with_time(self):
        import pandas as pd
        ts1 = pd.Timestamp("2026-02-07T00:00:00Z")
        ts2 = pd.Timestamp("2026-02-07T06:00:00Z")
        assert _compute_gmst(ts1) != _compute_gmst(ts2)


@pytest.mark.skipif(not PARQUET_EXISTS, reason="ground_truth.parquet not found")
class TestSpaceCatalogLoading:
    @pytest.fixture(scope="class")
    def catalog(self):
        cat = SpaceCatalog()
        cat.load(PARQUET_PATH)
        return cat

    def test_is_loaded(self, catalog):
        assert catalog.is_loaded is True

    def test_object_count(self, catalog):
        assert catalog.n_objects == 1000

    def test_timestep_count(self, catalog):
        assert catalog.n_timesteps == 1440

    def test_positions_shape(self, catalog):
        assert catalog.positions.shape == (1000, 1440, 3)

    def test_velocities_shape(self, catalog):
        assert catalog.velocities.shape == (1000, 1440, 3)

    def test_latitudes_shape(self, catalog):
        assert catalog.latitudes.shape == (1000, 1440)

    def test_longitudes_shape(self, catalog):
        assert catalog.longitudes.shape == (1000, 1440)

    def test_altitudes_shape(self, catalog):
        assert catalog.altitudes.shape == (1000, 1440)

    def test_latitudes_in_range(self, catalog):
        assert np.all(catalog.latitudes >= -90)
        assert np.all(catalog.latitudes <= 90)

    def test_longitudes_in_range(self, catalog):
        assert np.all(catalog.longitudes >= -180)
        assert np.all(catalog.longitudes <= 180)

    def test_altitudes_positive(self, catalog):
        # All satellites should be above Earth's surface
        assert np.all(catalog.altitudes > 0)

    def test_object_ids(self, catalog):
        assert len(catalog.object_ids) == 1000
        assert catalog.object_ids[0] == 0

    def test_object_names(self, catalog):
        assert len(catalog.object_names) == 1000
        assert catalog.object_names[0] == "CALSPHERE 1"

    def test_regimes(self, catalog):
        assert len(catalog.regimes) == 1000
        assert all(r in ("LEO", "MEO", "GEO", "HEO") for r in catalog.regimes)

    def test_object_types_loaded(self, catalog):
        assert len(catalog.object_types) == 1000
        # All types should be valid
        assert all(t in ("PAYLOAD", "DEBRIS", "ROCKET_BODY") for t in catalog.object_types)

    def test_time_isos(self, catalog):
        assert len(catalog.time_isos) == 1440
        assert "2026-02-07" in catalog.time_isos[0]


@pytest.mark.skipif(not PARQUET_EXISTS, reason="ground_truth.parquet not found")
class TestSpaceCatalogQueries:
    @pytest.fixture(scope="class")
    def catalog(self):
        cat = SpaceCatalog()
        cat.load(PARQUET_PATH)
        return cat

    def test_get_all_positions_at_timestep(self, catalog):
        positions = catalog.get_all_positions_at_timestep(0)
        assert len(positions) == 1000
        assert "id" in positions[0]
        assert "lat" in positions[0]
        assert "lon" in positions[0]
        assert "alt_km" in positions[0]
        assert "name" in positions[0]
        assert "object_type" in positions[0]

    def test_get_all_positions_clamps_timestep(self, catalog):
        positions = catalog.get_all_positions_at_timestep(99999)
        assert len(positions) == 1000

    def test_get_object_index(self, catalog):
        assert catalog.get_object_index(0) == 0
        assert catalog.get_object_index(999) is not None
        assert catalog.get_object_index(-1) is None
        assert catalog.get_object_index(9999) is None

    def test_get_object_trajectory(self, catalog):
        traj = catalog.get_object_trajectory(0, start=0, end=5)
        assert len(traj) == 5
        assert traj[0]["timestep"] == 0
        assert "lat" in traj[0]
        assert "position_x" in traj[0]
        assert "velocity_x" in traj[0]

    def test_get_object_trajectory_full(self, catalog):
        traj = catalog.get_object_trajectory(0)
        assert len(traj) == 1440

    def test_get_object_trajectory_invalid_id(self, catalog):
        traj = catalog.get_object_trajectory(9999)
        assert traj == []

    def test_get_object_summary(self, catalog):
        summary = catalog.get_object_summary(0)
        assert summary is not None
        assert summary["id"] == 0
        assert summary["name"] == "CALSPHERE 1"
        assert summary["object_type"] in ("PAYLOAD", "DEBRIS", "ROCKET_BODY")
        assert summary["regime"] in ("LEO", "MEO", "GEO", "HEO")
        assert summary["altitude_km"] > 0

    def test_get_object_summary_invalid(self, catalog):
        assert catalog.get_object_summary(9999) is None

    def test_get_all_summaries(self, catalog):
        summaries = catalog.get_all_summaries()
        assert len(summaries) == 1000
        assert "object_type" in summaries[0]

    def test_get_positions_and_velocities(self, catalog):
        result = catalog.get_positions_and_velocities(0)
        assert result is not None
        positions, velocities, timestamps = result
        assert positions.shape == (1440, 3)
        assert velocities.shape == (1440, 3)
        assert len(timestamps) == 1440
        assert timestamps[0] == 0.0  # First timestamp is 0

    def test_get_positions_and_velocities_invalid(self, catalog):
        assert catalog.get_positions_and_velocities(9999) is None

    def test_get_positions_and_velocities_window(self, catalog):
        result = catalog.get_positions_and_velocities(0, start=10, end=20)
        assert result is not None
        positions, velocities, timestamps = result
        assert positions.shape == (10, 3)


class TestSpaceCatalogEmpty:
    def test_not_loaded(self):
        cat = SpaceCatalog()
        assert cat.is_loaded is False
        assert cat.object_types == []
        assert cat.get_all_positions_at_timestep(0) == []

    def test_file_not_found(self):
        cat = SpaceCatalog()
        with pytest.raises(FileNotFoundError):
            cat.load("nonexistent.parquet")
