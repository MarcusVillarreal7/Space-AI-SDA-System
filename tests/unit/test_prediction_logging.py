"""Tests for prediction logging and monitoring database functions."""

import os
import pytest

from src.api.database import (
    Base,
    PredictionLog,
    IngestionLog,
    get_prediction_logs,
    get_prediction_stats,
    get_ingestion_logs,
    init_db,
    log_ingestion,
    log_prediction,
)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Use a fresh in-memory SQLite DB for each test."""
    import src.api.database as db_mod

    # Reset module-level state
    db_mod._engine = None
    db_mod._SessionLocal = None

    # Ensure no DATABASE_URL so we get SQLite
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # Point SQLite at temp dir
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "test.db")

    init_db()
    yield
    db_mod._engine = None
    db_mod._SessionLocal = None


class TestPredictionLogging:
    def test_log_prediction_and_retrieve(self):
        log_prediction(
            object_id=42,
            object_name="SAT-42",
            object_type="PAYLOAD",
            threat_tier="ELEVATED",
            threat_score=0.75,
            maneuver_class="Major Maneuver",
            anomaly_score=0.3,
            proximity_score=0.8,
            latency_ms=5.2,
        )
        logs = get_prediction_logs(limit=10)
        assert len(logs) == 1
        assert logs[0]["object_id"] == 42
        assert logs[0]["threat_tier"] == "ELEVATED"
        assert logs[0]["threat_score"] == 0.75
        assert logs[0]["maneuver_class"] == "Major Maneuver"

    def test_multiple_predictions(self):
        for i in range(5):
            log_prediction(
                object_id=i,
                object_name=f"SAT-{i}",
                object_type="PAYLOAD",
                threat_tier="MINIMAL",
                threat_score=0.1 * i,
                latency_ms=3.0,
            )
        logs = get_prediction_logs(limit=3)
        assert len(logs) == 3
        # Newest first
        assert logs[0]["object_id"] == 4

    def test_prediction_stats_empty(self):
        stats = get_prediction_stats()
        assert stats["total_predictions"] == 0
        assert stats["avg_latency_ms"] == 0.0
        assert stats["tier_distribution"] == {}

    def test_prediction_stats_with_data(self):
        log_prediction(
            object_id=1, object_name="A", object_type="PAYLOAD",
            threat_tier="MINIMAL", threat_score=0.1, latency_ms=2.0,
        )
        log_prediction(
            object_id=2, object_name="B", object_type="DEBRIS",
            threat_tier="ELEVATED", threat_score=0.8, latency_ms=8.0,
        )
        log_prediction(
            object_id=3, object_name="C", object_type="PAYLOAD",
            threat_tier="MINIMAL", threat_score=0.2, latency_ms=5.0,
        )
        stats = get_prediction_stats()
        assert stats["total_predictions"] == 3
        assert stats["avg_latency_ms"] == 5.0
        assert stats["tier_distribution"]["MINIMAL"] == 2
        assert stats["tier_distribution"]["ELEVATED"] == 1
        assert stats["type_distribution"]["PAYLOAD"] == 2
        assert stats["type_distribution"]["DEBRIS"] == 1


class TestIngestionLogging:
    def test_log_ingestion_and_retrieve(self):
        log_ingestion(
            object_id=1001,
            object_name="NEW-SAT",
            object_type="PAYLOAD",
            source="manual",
            tle_epoch="26001.5",
        )
        logs = get_ingestion_logs(limit=10)
        assert len(logs) == 1
        assert logs[0]["object_id"] == 1001
        assert logs[0]["source"] == "manual"
        assert logs[0]["tle_epoch"] == "26001.5"

    def test_ingestion_pagination(self):
        for i in range(10):
            log_ingestion(
                object_id=2000 + i,
                object_name=f"SAT-{2000+i}",
                object_type="DEBRIS",
                source="batch",
            )
        page1 = get_ingestion_logs(limit=3, offset=0)
        page2 = get_ingestion_logs(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0]["object_id"] != page2[0]["object_id"]


class TestDatabaseInit:
    def test_sqlite_fallback(self, tmp_path, monkeypatch):
        """Without DATABASE_URL, init_db uses SQLite."""
        import src.api.database as db_mod
        db_mod._engine = None
        db_mod._SessionLocal = None
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "fallback.db")
        init_db()
        assert db_mod._engine is not None
        assert "sqlite" in str(db_mod._engine.url)

    def test_postgres_url_fixup(self, monkeypatch):
        """DATABASE_URL with postgres:// should be corrected to postgresql://."""
        import src.api.database as db_mod
        db_mod._engine = None
        db_mod._SessionLocal = None
        # We can't actually connect to Postgres in tests, but we can verify
        # the URL correction logic. We'll mock create_engine.
        from unittest.mock import patch, MagicMock
        monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@host/db")
        mock_engine = MagicMock()
        mock_engine.url = "postgresql://user:pass@host/db"
        with patch.object(db_mod, "create_engine", return_value=mock_engine) as mock_ce:
            with patch.object(Base.metadata, "create_all"):
                init_db()
            # Verify the corrected URL was passed
            args = mock_ce.call_args
            assert args[0][0] == "postgresql://user:pass@host/db"
            assert args[1]["pool_pre_ping"] is True
