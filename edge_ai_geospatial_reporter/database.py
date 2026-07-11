"""
database.py - SQLAlchemy models and data-access functions for the
Edge-AI Geospatial Infrastructure & Asset Reporter.

Stores every detected anomaly with its timestamp, type, confidence, and
geolocation so the Streamlit dashboard and reporter can query it back out.
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from typing import List, Optional, Sequence

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    create_engine,
    desc,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

import config

Base = declarative_base()


class DetectionLog(Base):
    """A single anomaly/asset detection event."""

    __tablename__ = "detection_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=dt.datetime.utcnow, nullable=False, index=True)
    anomaly_type = Column(String(64), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    source_image = Column(String(512), nullable=True)
    bbox_x1 = Column(Float, nullable=True)
    bbox_y1 = Column(Float, nullable=True)
    bbox_x2 = Column(Float, nullable=True)
    bbox_y2 = Column(Float, nullable=True)
    notes = Column(String(256), nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "anomaly_type": self.anomaly_type,
            "confidence": round(self.confidence, 4),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "source_image": self.source_image,
            "bbox": [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            "notes": self.notes,
        }


_engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})
_SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    """Create all tables if they do not already exist. Safe to call repeatedly."""
    Base.metadata.create_all(_engine)


@contextmanager
def get_session():
    """Context-managed SQLAlchemy session with automatic commit/rollback."""
    session: Session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def insert_detection(
    anomaly_type: str,
    confidence: float,
    latitude: float,
    longitude: float,
    source_image: Optional[str] = None,
    bbox: Optional[Sequence[float]] = None,
    notes: Optional[str] = None,
    timestamp: Optional[dt.datetime] = None,
) -> int:
    """Insert a single detection record and return its new primary key."""
    bbox = bbox or (None, None, None, None)
    with get_session() as session:
        record = DetectionLog(
            timestamp=timestamp or dt.datetime.utcnow(),
            anomaly_type=anomaly_type,
            confidence=float(confidence),
            latitude=float(latitude),
            longitude=float(longitude),
            source_image=source_image,
            bbox_x1=bbox[0],
            bbox_y1=bbox[1],
            bbox_x2=bbox[2],
            bbox_y2=bbox[3],
            notes=notes,
        )
        session.add(record)
        session.flush()
        return record.id


def insert_detections_bulk(detections: List[dict]) -> int:
    """Insert many detection dicts at once. Returns the number inserted."""
    if not detections:
        return 0
    with get_session() as session:
        records = [
            DetectionLog(
                timestamp=d.get("timestamp") or dt.datetime.utcnow(),
                anomaly_type=d["anomaly_type"],
                confidence=float(d["confidence"]),
                latitude=float(d["latitude"]),
                longitude=float(d["longitude"]),
                source_image=d.get("source_image"),
                bbox_x1=d.get("bbox", (None,) * 4)[0],
                bbox_y1=d.get("bbox", (None,) * 4)[1],
                bbox_x2=d.get("bbox", (None,) * 4)[2],
                bbox_y2=d.get("bbox", (None,) * 4)[3],
                notes=d.get("notes"),
            )
            for d in detections
        ]
        session.add_all(records)
        session.flush()
        return len(records)


def get_recent_anomalies(limit: int = 10) -> List[dict]:
    """Return the `limit` most recent anomalies, newest first."""
    with get_session() as session:
        rows = (
            session.query(DetectionLog)
            .order_by(desc(DetectionLog.timestamp))
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in rows]


def get_all_detections(limit: Optional[int] = None) -> List[dict]:
    """Return all detections (optionally capped), newest first. Used for the
    dashboard's live dataframe and map."""
    with get_session() as session:
        query = session.query(DetectionLog).order_by(desc(DetectionLog.timestamp))
        if limit:
            query = query.limit(limit)
        return [r.to_dict() for r in query.all()]


def get_detections_by_type(anomaly_type: str, limit: int = 100) -> List[dict]:
    """Return the most recent detections of a single anomaly type."""
    with get_session() as session:
        rows = (
            session.query(DetectionLog)
            .filter(DetectionLog.anomaly_type == anomaly_type)
            .order_by(desc(DetectionLog.timestamp))
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in rows]


def get_detections_since(since: dt.datetime) -> List[dict]:
    """Return all detections recorded at or after `since`."""
    with get_session() as session:
        rows = (
            session.query(DetectionLog)
            .filter(DetectionLog.timestamp >= since)
            .order_by(desc(DetectionLog.timestamp))
            .all()
        )
        return [r.to_dict() for r in rows]


def get_summary_stats() -> dict:
    """Aggregate stats used by the report generator: total count, counts per
    anomaly type, average confidence, and the observed date range."""
    with get_session() as session:
        total = session.query(func.count(DetectionLog.id)).scalar() or 0
        avg_conf = session.query(func.avg(DetectionLog.confidence)).scalar() or 0.0
        first_seen = session.query(func.min(DetectionLog.timestamp)).scalar()
        last_seen = session.query(func.max(DetectionLog.timestamp)).scalar()

        type_counts_rows = (
            session.query(DetectionLog.anomaly_type, func.count(DetectionLog.id))
            .group_by(DetectionLog.anomaly_type)
            .order_by(desc(func.count(DetectionLog.id)))
            .all()
        )
        type_counts = {t: c for t, c in type_counts_rows}

        return {
            "total_detections": total,
            "average_confidence": round(float(avg_conf), 4) if avg_conf else 0.0,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "counts_by_type": type_counts,
        }


def clear_all_detections() -> int:
    """Danger zone: wipe every detection record. Returns rows deleted."""
    with get_session() as session:
        count = session.query(DetectionLog).count()
        session.query(DetectionLog).delete()
        return count


# Ensure the schema exists as soon as this module is imported anywhere.
init_db()
