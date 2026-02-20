"""
Integration tests for all REST API endpoints via TestClient.
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

PARQUET_PATH = Path("data/processed/ml_train_1k/ground_truth.parquet")


@pytest.fixture(scope="module")
def client():
    """Create a test client with the real app."""
    if not PARQUET_PATH.exists():
        pytest.skip("ground_truth.parquet not found")

    from src.api.main import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["objects_loaded"] == 1000
        assert data["timesteps"] == 1440


class TestObjectsEndpoints:
    def test_list_objects(self, client):
        r = client.get("/api/objects?limit=5")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 5
        assert "id" in data[0]
        assert "name" in data[0]
        assert "regime" in data[0]
        assert "altitude_km" in data[0]
        assert "threat_tier" in data[0]

    def test_list_objects_default(self, client):
        r = client.get("/api/objects")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1000

    def test_list_objects_filter_regime(self, client):
        r = client.get("/api/objects?regime=LEO&limit=10")
        assert r.status_code == 200
        data = r.json()
        assert all(obj["regime"] == "LEO" for obj in data)

    def test_list_objects_pagination(self, client):
        r = client.get("/api/objects?limit=5&offset=10")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 5

    def test_get_object(self, client):
        r = client.get("/api/objects/0")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == 0
        assert data["name"] == "CALSPHERE 1"
        assert "trajectory" in data
        assert len(data["trajectory"]) == 1440

    def test_get_object_with_trajectory_range(self, client):
        r = client.get("/api/objects/0?trajectory_start=0&trajectory_end=5")
        assert r.status_code == 200
        data = r.json()
        assert len(data["trajectory"]) == 5

    def test_get_object_not_found(self, client):
        r = client.get("/api/objects/9999")
        assert r.status_code == 404


class TestSimulationEndpoints:
    def test_simulation_status(self, client):
        r = client.get("/api/simulation/status")
        assert r.status_code == 200
        data = r.json()
        assert "is_playing" in data
        assert "speed" in data
        assert "timestep" in data
        assert "max_timestep" in data
        assert data["max_timestep"] == 1439

    def test_pause(self, client):
        r = client.post("/api/simulation/pause")
        assert r.status_code == 200

    def test_play(self, client):
        r = client.post("/api/simulation/play")
        assert r.status_code == 200

    def test_set_speed(self, client):
        r = client.post("/api/simulation/speed?speed=360")
        assert r.status_code == 200
        data = r.json()
        assert data["speed"] == 360.0

    def test_seek(self, client):
        r = client.post("/api/simulation/seek?timestep=100")
        assert r.status_code == 200
        data = r.json()
        assert data["timestep"] == 100

    def test_seek_out_of_range(self, client):
        r = client.post("/api/simulation/seek?timestep=99999")
        assert r.status_code == 400


class TestThreatEndpoints:
    def test_threat_summary(self, client):
        r = client.get("/api/threat/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1000
        assert "by_tier" in data
        assert "MINIMAL" in data["by_tier"]

    def test_assess_object(self, client):
        r = client.get("/api/threat/object/42")
        assert r.status_code == 200
        data = r.json()
        assert data["object_id"] == 42
        assert 0 <= data["threat_score"] <= 100
        assert data["threat_tier"] in (
            "MINIMAL", "LOW", "MODERATE", "ELEVATED", "CRITICAL"
        )
        assert "maneuver_class" in data
        assert "contributing_factors" in data
        assert data["latency_ms"] >= 0

    def test_assess_object_not_found(self, client):
        r = client.get("/api/threat/object/9999")
        assert r.status_code == 404

    def test_get_alerts(self, client):
        r = client.get("/api/threat/alerts?limit=10")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)


class TestMetricsEndpoint:
    def test_metrics(self, client):
        r = client.get("/api/metrics")
        assert r.status_code == 200
        data = r.json()
        assert data["objects_tracked"] == 1000
        assert data["uptime_seconds"] >= 0


class TestMonitoringEndpoints:
    def test_monitoring_predictions_empty(self, client):
        """GET /api/monitoring/predictions returns list (may be empty initially)."""
        r = client.get("/api/monitoring/predictions?limit=10")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)

    def test_monitoring_stats(self, client):
        """GET /api/monitoring/stats returns aggregate stats."""
        r = client.get("/api/monitoring/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_predictions" in data
        assert "avg_latency_ms" in data
        assert "tier_distribution" in data
        assert "type_distribution" in data
        assert isinstance(data["tier_distribution"], dict)

    def test_monitoring_ingestions_empty(self, client):
        """GET /api/monitoring/ingestions returns list."""
        r = client.get("/api/monitoring/ingestions?limit=10")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)

    def test_monitoring_predictions_after_assess(self, client):
        """Prediction log endpoint returns entries with correct schema."""
        # Directly log a prediction to isolate the monitoring endpoint test
        # from pipeline availability (which varies between local and CI).
        from src.api.database import log_prediction
        log_prediction(
            object_id=5,
            object_name="TEST-OBJ-5",
            object_type="PAYLOAD",
            threat_tier="MINIMAL",
            threat_score=0.0,
            latency_ms=1.0,
        )

        r = client.get("/api/monitoring/predictions?limit=5")
        assert r.status_code == 200
        data = r.json()
        assert len(data) >= 1
        entry = data[0]
        assert "object_id" in entry
        assert "threat_tier" in entry
        assert "latency_ms" in entry
        assert "timestamp" in entry


class TestIngestionEndpoints:
    def test_ingestion_status(self, client):
        """GET /api/ingest/status returns counts."""
        r = client.get("/api/ingest/status")
        assert r.status_code == 200
        data = r.json()
        assert "ingested_count" in data
        assert "catalog_size" in data
        assert data["catalog_size"] >= 1000

    def test_ingest_tle_bad_data(self, client):
        """POST /api/ingest/tle with invalid TLE returns 400."""
        r = client.post("/api/ingest/tle", json={
            "name": "BAD SAT",
            "tle_line1": "invalid line 1",
            "tle_line2": "invalid line 2",
        })
        assert r.status_code == 400


class TestWebSocketConnection:
    def test_websocket_connect(self, client):
        with client.websocket_connect("/api/ws") as ws:
            data = ws.receive_json()
            assert data["type"] == "positions"
            assert len(data["objects"]) >= 1000
            assert "lat" in data["objects"][0]
            assert "lon" in data["objects"][0]
