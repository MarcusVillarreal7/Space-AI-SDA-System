"""
SQLite database for alert history and assessment cache.
"""

from __future__ import annotations

import json
import logging
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


# Engine and session
_engine = None
_SessionLocal = None


def init_db() -> None:
    """Initialize the database, creating tables if needed."""
    global _engine, _SessionLocal
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine)
    logger.info("Database initialized at %s", DB_PATH)


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
