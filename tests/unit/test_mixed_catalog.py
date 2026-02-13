"""
Tests for mixed catalog builder — TLE classification logic and catalog sampling.
"""

import pytest
from unittest.mock import MagicMock


def _make_tle(name: str, catalog_number: int = 99999):
    """Create a minimal mock TLE object."""
    tle = MagicMock()
    tle.name = name
    tle.catalog_number = catalog_number
    tle.line1 = f"1 {catalog_number:5d}U 24001A   26037.50000000  .00000000  00000-0  00000-0 0  0000"
    tle.line2 = f"2 {catalog_number:5d}  51.6400   0.0000 0000000   0.0000   0.0000 15.50000000 00000"
    return tle


class TestClassifyTleType:
    def test_active_satellite_is_payload(self):
        from scripts.build_mixed_catalog import classify_tle_type
        tle = _make_tle("ISS (ZARYA)")
        assert classify_tle_type(tle, "active.tle") == "PAYLOAD"

    def test_debris_source_is_debris(self):
        from scripts.build_mixed_catalog import classify_tle_type
        tle = _make_tle("COSMOS 2251 DEB")
        assert classify_tle_type(tle, "cosmos-2251-deb.tle") == "DEBRIS"

    def test_deb_in_name_is_debris(self):
        from scripts.build_mixed_catalog import classify_tle_type
        tle = _make_tle("FENGYUN 1C DEB")
        assert classify_tle_type(tle, "active.tle") == "DEBRIS"

    def test_rb_in_name_is_rocket_body(self):
        from scripts.build_mixed_catalog import classify_tle_type
        tle = _make_tle("CZ-2D R/B")
        assert classify_tle_type(tle, "active.tle") == "ROCKET_BODY"

    def test_rocket_body_rb_suffix(self):
        from scripts.build_mixed_catalog import classify_tle_type
        tle = _make_tle("FALCON 9 R/B")
        assert classify_tle_type(tle, "last-30-days.tle") == "ROCKET_BODY"

    def test_generic_name_is_payload(self):
        from scripts.build_mixed_catalog import classify_tle_type
        tle = _make_tle("STARLINK-1234")
        assert classify_tle_type(tle, "active.tle") == "PAYLOAD"

    def test_debris_source_overrides_name(self):
        """Even if name looks like a payload, debris source file wins."""
        from scripts.build_mixed_catalog import classify_tle_type
        tle = _make_tle("SOME SATELLITE")
        assert classify_tle_type(tle, "fengyun-1c-deb.tle") == "DEBRIS"


class TestSampleCatalog:
    def _make_pool(self, n: int, type_str: str):
        return [(
            _make_tle(f"{type_str}-{i}", catalog_number=1000 * hash(type_str) + i),
            f"{type_str.lower()}.tle",
        ) for i in range(n)]

    def test_correct_total(self):
        from scripts.build_mixed_catalog import sample_catalog
        by_type = {
            "PAYLOAD": self._make_pool(700, "PAYLOAD"),
            "DEBRIS": self._make_pool(400, "DEBRIS"),
            "ROCKET_BODY": self._make_pool(100, "ROCKET_BODY"),
        }
        catalog = sample_catalog(by_type, n_payloads=600, n_debris=350, n_rocket_bodies=50, seed=42)
        assert len(catalog) == 1000

    def test_scenario_slots_are_payload(self):
        from scripts.build_mixed_catalog import sample_catalog, SCENARIO_SLOTS
        by_type = {
            "PAYLOAD": self._make_pool(700, "PAYLOAD"),
            "DEBRIS": self._make_pool(400, "DEBRIS"),
            "ROCKET_BODY": self._make_pool(100, "ROCKET_BODY"),
        }
        catalog = sample_catalog(by_type, n_payloads=600, n_debris=350, n_rocket_bodies=50, seed=42)
        for slot in SCENARIO_SLOTS:
            if slot < len(catalog):
                _, obj_type = catalog[slot]
                assert obj_type == "PAYLOAD", f"Slot {slot} should be PAYLOAD, got {obj_type}"

    def test_deterministic_with_seed(self):
        from scripts.build_mixed_catalog import sample_catalog
        by_type = {
            "PAYLOAD": self._make_pool(700, "PAYLOAD"),
            "DEBRIS": self._make_pool(400, "DEBRIS"),
            "ROCKET_BODY": self._make_pool(100, "ROCKET_BODY"),
        }
        c1 = sample_catalog(by_type, seed=42)
        c2 = sample_catalog(by_type, seed=42)
        assert len(c1) == len(c2)
        for (tle1, t1), (tle2, t2) in zip(c1, c2):
            assert tle1.name == tle2.name
            assert t1 == t2

    def test_handles_insufficient_pool(self):
        """If fewer TLEs available than requested, use all available."""
        from scripts.build_mixed_catalog import sample_catalog
        by_type = {
            "PAYLOAD": self._make_pool(100, "PAYLOAD"),
            "DEBRIS": self._make_pool(50, "DEBRIS"),
            "ROCKET_BODY": self._make_pool(10, "ROCKET_BODY"),
        }
        catalog = sample_catalog(by_type, n_payloads=600, n_debris=350, n_rocket_bodies=50, seed=42)
        # Should still produce a list (just smaller than requested total)
        assert len(catalog) > 0
        types = [t for _, t in catalog]
        assert "PAYLOAD" in types
