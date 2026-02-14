"""
Database layer — PostgreSQL in production, SQLite fallback for local dev.

Reads DATABASE_URL from environment. Falls back to local SQLite when unset.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from sqlalchemy import Column, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DB_PATH = Path("data/dashboard.db")


class Base(DeclarativeBase):
    pass


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    object_id = Column(Integer, index=True, nullable=False)
    object_name = Column(String, nullable=False)
    threat_tier = Column(String, nullable=False)
    threat_score = Column(Float, nullable=False)
    message = Column(Text, nullable=False)
    timestamp = Column(String, nullable=False)  # ISO format


class AssessmentCache(Base):
    __tablename__ = "assessment_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    object_id = Column(Integer, index=True, nullable=False)
    timestep = Column(Integer, nullable=False)
    result_json = Column(Text, nullable=False)
    created_at = Column(Float, nullable=False)  # Unix timestamp


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    object_id = Column(Integer, index=True, nullable=False)
    object_name = Column(String, nullable=False)
    object_type = Column(String, nullable=False)
    threat_tier = Column(String, nullable=False)
    threat_score = Column(Float, nullable=False)
    maneuver_class = Column(String, nullable=True)
    intent_category = Column(String, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    proximity_score = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=False)
    model_version = Column(String, nullable=True)
    timestamp = Column(String, nullable=False)


class IngestionLog(Base):
    __tablename__ = "ingestion_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    object_id = Column(Integer, nullable=False)
    object_name = Column(String, nullable=False)
    object_type = Column(String, nullable=False)
    source = Column(String, nullable=False)  # "manual", "celestrak", "batch"
    tle_epoch = Column(String, nullable=True)
    timestamp = Column(String, nullable=False)


# Engine and session
_engine = None
_SessionLocal = None


def init_db() -> None:
    """Initialize the database, creating tables if needed.

    Uses DATABASE_URL env var for PostgreSQL when set.
    Falls back to local SQLite otherwise.
    """
    global _engine, _SessionLocal

    database_url = os.environ.get("DATABASE_URL")

    if database_url:
        # Railway sets postgres:// but SQLAlchemy requires postgresql://
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        _engine = create_engine(database_url, echo=False, pool_pre_ping=True)
        logger.info("Database initialized with PostgreSQL")
    else:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
        logger.info("Database initialized at %s (SQLite fallback)", DB_PATH)

    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine)


def get_session() -> Session:
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


def store_alert(
    object_id: int,
    object_name: str,
    threat_tier: str,
    threat_score: float,
    message: str,
) -> Alert:
    """Store a new alert in the database."""
    session = get_session()
    alert = Alert(
        object_id=object_id,
        object_name=object_name,
        threat_tier=threat_tier,
        threat_score=threat_score,
        message=message,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    session.add(alert)
    session.commit()
    alert_id = alert.id
    session.close()
    return alert


def get_alerts(limit: int = 50, offset: int = 0) -> list[dict]:
    """Get recent alerts, newest first."""
    session = get_session()
    alerts = (
        session.query(Alert)
        .order_by(Alert.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    result = [
        {
            "id": a.id,
            "object_id": a.object_id,
            "object_name": a.object_name,
            "threat_tier": a.threat_tier,
            "threat_score": a.threat_score,
            "message": a.message,
            "timestamp": a.timestamp,
        }
        for a in alerts
    ]
    session.close()
    return result


def clear_alerts() -> int:
    """Delete all alerts. Returns number of deleted rows."""
    session = get_session()
    count = session.query(Alert).delete()
    session.commit()
    session.close()
    logger.info("Cleared %d alerts from database", count)
    return count


def clear_assessment_cache() -> int:
    """Delete all cached assessments. Returns number of deleted rows."""
    session = get_session()
    count = session.query(AssessmentCache).delete()
    session.commit()
    session.close()
    logger.info("Cleared %d cached assessments from database", count)
    return count


def cache_assessment(object_id: int, timestep: int, result: dict) -> None:
    """Cache a threat assessment result."""
    session = get_session()
    entry = AssessmentCache(
        object_id=object_id,
        timestep=timestep,
        result_json=json.dumps(result),
        created_at=time.time(),
    )
    session.add(entry)
    session.commit()
    session.close()


def get_cached_assessment(object_id: int, timestep: int) -> dict | None:
    """Retrieve a cached assessment if available."""
    session = get_session()
    entry = (
        session.query(AssessmentCache)
        .filter_by(object_id=object_id, timestep=timestep)
        .first()
    )
    session.close()
    if entry:
        return json.loads(entry.result_json)
    return None


# --- Prediction logging ---

def log_prediction(
    object_id: int,
    object_name: str,
    object_type: str,
    threat_tier: str,
    threat_score: float,
    maneuver_class: str | None = None,
    intent_category: str | None = None,
    anomaly_score: float | None = None,
    proximity_score: float | None = None,
    latency_ms: float = 0.0,
    model_version: str | None = None,
) -> None:
    """Log a prediction to the prediction_logs table."""
    session = get_session()
    entry = PredictionLog(
        object_id=object_id,
        object_name=object_name,
        object_type=object_type,
        threat_tier=threat_tier,
        threat_score=threat_score,
        maneuver_class=maneuver_class,
        intent_category=intent_category,
        anomaly_score=anomaly_score,
        proximity_score=proximity_score,
        latency_ms=latency_ms,
        model_version=model_version,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    session.add(entry)
    session.commit()
    session.close()


def get_prediction_logs(limit: int = 50, offset: int = 0) -> list[dict]:
    """Get recent prediction logs, newest first."""
    session = get_session()
    logs = (
        session.query(PredictionLog)
        .order_by(PredictionLog.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    result = [
        {
            "id": p.id,
            "object_id": p.object_id,
            "object_name": p.object_name,
            "object_type": p.object_type,
            "threat_tier": p.threat_tier,
            "threat_score": p.threat_score,
            "maneuver_class": p.maneuver_class,
            "intent_category": p.intent_category,
            "anomaly_score": p.anomaly_score,
            "proximity_score": p.proximity_score,
            "latency_ms": p.latency_ms,
            "model_version": p.model_version,
            "timestamp": p.timestamp,
        }
        for p in logs
    ]
    session.close()
    return result


def get_prediction_stats() -> dict:
    """Aggregate prediction statistics."""
    session = get_session()
    from sqlalchemy import func

    total = session.query(func.count(PredictionLog.id)).scalar() or 0
    avg_latency = session.query(func.avg(PredictionLog.latency_ms)).scalar() or 0.0

    # Tier distribution
    tier_rows = (
        session.query(PredictionLog.threat_tier, func.count(PredictionLog.id))
        .group_by(PredictionLog.threat_tier)
        .all()
    )
    tier_dist = {row[0]: row[1] for row in tier_rows}

    # Type distribution
    type_rows = (
        session.query(PredictionLog.object_type, func.count(PredictionLog.id))
        .group_by(PredictionLog.object_type)
        .all()
    )
    type_dist = {row[0]: row[1] for row in type_rows}

    session.close()
    return {
        "total_predictions": total,
        "avg_latency_ms": round(float(avg_latency), 2),
        "tier_distribution": tier_dist,
        "type_distribution": type_dist,
    }


# --- Ingestion logging ---

def log_ingestion(
    object_id: int,
    object_name: str,
    object_type: str,
    source: str = "manual",
    tle_epoch: str | None = None,
) -> None:
    """Log a TLE ingestion event."""
    session = get_session()
    entry = IngestionLog(
        object_id=object_id,
        object_name=object_name,
        object_type=object_type,
        source=source,
        tle_epoch=tle_epoch,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    session.add(entry)
    session.commit()
    session.close()


def get_ingestion_logs(limit: int = 50, offset: int = 0) -> list[dict]:
    """Get recent ingestion logs, newest first."""
    session = get_session()
    logs = (
        session.query(IngestionLog)
        .order_by(IngestionLog.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    result = [
        {
            "id": i.id,
            "object_id": i.object_id,
            "object_name": i.object_name,
            "object_type": i.object_type,
            "source": i.source,
            "tle_epoch": i.tle_epoch,
            "timestamp": i.timestamp,
        }
        for i in logs
    ]
    session.close()
    return result
