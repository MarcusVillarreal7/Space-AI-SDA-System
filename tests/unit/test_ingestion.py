"""Tests for IngestionService and SpaceCatalog.add_object()."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.api.data_manager import SpaceCatalog
from src.api.ingestion import IngestionService, _classify_type_from_name


class TestClassifyTypeFromName:
    def test_payload(self):
        assert _classify_type_from_name("ISS (ZARYA)") == "PAYLOAD"

    def test_debris(self):
        assert _classify_type_from_name("COSMOS 2251 DEB") == "DEBRIS"

    def test_debris_start(self):
        assert _classify_type_from_name("DEB [UNKNOWN]") == "DEBRIS"

    def test_rocket_body(self):
        assert _classify_type_from_name("SL-4 R/B") == "ROCKET_BODY"

    def test_rocket_body_rb(self):
        assert _classify_type_from_name("CZ-3B RB") == "ROCKET_BODY"


class TestSpaceCatalogAddObject:
    @pytest.fixture
    def catalog(self, tmp_path):
        """Create a minimal catalog with 2 objects and 10 timesteps."""
        n_obj, n_ts = 2, 10
        timestamps = pd.date_range("2026-01-01", periods=n_ts, freq="min")
        rows = []
        for oid in range(n_obj):
            for i, t in enumerate(timestamps):
                rows.append({
                    "object_id": oid,
                    "object_name": f"SAT-{oid}",
                    "time": t,
                    "position_x": 7000.0 + oid * 100 + i,
                    "position_y": 100.0,
                    "position_z": 50.0,
                    "velocity_x": 0.0,
                    "velocity_y": 7.5,
                    "velocity_z": 0.0,
                    "altitude_km": 600.0 + oid * 10,
                    "speed_km_s": 7.5,
                    "object_type": "PAYLOAD",
                })
        df = pd.DataFrame(rows)
        path = tmp_path / "ground_truth.parquet"
        df.to_parquet(path)

        cat = SpaceCatalog()
        cat.load(path)
        return cat

    def test_add_object_increases_count(self, catalog):
        assert catalog.n_objects == 2
        positions = np.random.randn(catalog.n_timesteps, 3) + np.array([7000, 100, 50])
        velocities = np.tile([0, 7.5, 0], (catalog.n_timesteps, 1))
        catalog.add_object(
            object_id=999,
            name="NEW-SAT",
            object_type="PAYLOAD",
            positions=positions,
            velocities=velocities,
            timestamps=catalog.timestamps,
        )
        assert catalog.n_objects == 3
        assert 999 in catalog.object_ids
        assert catalog.object_names[-1] == "NEW-SAT"

    def test_add_object_arrays_consistent(self, catalog):
        positions = np.random.randn(catalog.n_timesteps, 3) + np.array([7000, 100, 50])
        velocities = np.tile([0, 7.5, 0], (catalog.n_timesteps, 1))
        catalog.add_object(
            object_id=888,
            name="TEST",
            object_type="DEBRIS",
            positions=positions,
            velocities=velocities,
            timestamps=catalog.timestamps,
        )
        assert catalog.positions.shape == (3, catalog.n_timesteps, 3)
        assert catalog.latitudes.shape == (3, catalog.n_timesteps)
        assert len(catalog.ref_altitudes) == 3
        assert catalog.object_types[-1] == "DEBRIS"

    def test_add_object_timestep_mismatch_raises(self, catalog):
        positions = np.random.randn(5, 3)
        velocities = np.random.randn(5, 3)
        with pytest.raises(ValueError, match="Timestep count mismatch"):
            catalog.add_object(
                object_id=777,
                name="BAD",
                object_type="PAYLOAD",
                positions=positions,
                velocities=velocities,
                timestamps=[pd.Timestamp("2026-01-01")] * 5,
            )

    def test_added_object_visible_in_positions_at_timestep(self, catalog):
        positions = np.random.randn(catalog.n_timesteps, 3) + np.array([7000, 100, 50])
        velocities = np.tile([0, 7.5, 0], (catalog.n_timesteps, 1))
        catalog.add_object(
            object_id=555,
            name="VISIBLE-SAT",
            object_type="PAYLOAD",
            positions=positions,
            velocities=velocities,
            timestamps=catalog.timestamps,
        )
        all_pos = catalog.get_all_positions_at_timestep(0)
        ids = [p["id"] for p in all_pos]
        assert 555 in ids


class TestIngestionService:
    @pytest.fixture
    def mock_catalog(self):
        """Minimal mock catalog for IngestionService."""
        cat = MagicMock(spec=SpaceCatalog)
        cat.object_ids = np.array([0, 1, 2])
        cat.n_objects = 3
        cat.timestamps = [pd.Timestamp("2026-01-01 00:00:00")] * 10
        cat.n_timesteps = 10
        cat.regimes = ["LEO", "LEO", "LEO"]
        cat.ref_altitudes = np.array([400.0, 500.0, 600.0])
        cat.get_object_index.return_value = 3
        return cat

    def test_get_next_id(self, mock_catalog):
        service = IngestionService(mock_catalog)
        assert service._get_next_id() == 3
        assert service._get_next_id() == 4

    def test_ingest_tle(self, mock_catalog):
        """Test ingestion with mocked SGP4."""
        # Mock the TLE
        mock_tle = MagicMock()
        mock_tle.epoch = 26001.5

        # Mock the propagator
        mock_prop = MagicMock()
        mock_state = MagicMock()
        mock_state.position = np.array([7000.0, 100.0, 50.0])
        mock_state.velocity = np.array([0.0, 7.5, 0.0])
        mock_prop.propagate.return_value = mock_state

        # Mock catalog methods for logging
        mock_catalog.regimes = ["LEO", "LEO", "LEO", "LEO"]
        mock_catalog.ref_altitudes = np.array([400.0, 500.0, 600.0, 622.0])

        service = IngestionService(mock_catalog)

        with patch("src.simulation.tle_loader.TLE.from_lines", return_value=mock_tle), \
             patch("src.simulation.orbital_mechanics.SGP4Propagator", return_value=mock_prop), \
             patch("src.api.database.log_ingestion"):
            result = service.ingest_tle(
                "TEST SAT",
                "1 99999U 26001A   26001.50000000 ...",
                "2 99999  51.6435 ...",
            )

        assert result["name"] == "TEST SAT"
        assert result["object_type"] == "PAYLOAD"
        assert result["timesteps"] == 10
        mock_catalog.add_object.assert_called_once()
        assert service._ingested_count == 1
