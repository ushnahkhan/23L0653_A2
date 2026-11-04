"""
db.py - request log for VisionGuard (SQLite by default, any SQLAlchemy URL via DATABASE_URL)
Only metadata is stored - never the image itself.
"""
import os
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table,
                        create_engine, insert, select, text)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
meta = MetaData()

predictions = Table(
    "predictions", meta,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("filename", String(120)),          # sanitized name only
    Column("file_size", Integer),
    Column("status", String(20)),             # "ok" or "rejected"
    Column("error", String(200)),
    Column("label", String(20)),
    Column("confidence", Float),
    Column("p_ai", Float),
    Column("threshold", Float),
    Column("model_version", String(20)),
    Column("latency_ms", Float),
)


def init_db():
    meta.create_all(engine)


def log_request(**fields) -> int:
    with engine.begin() as conn:
        res = conn.execute(insert(predictions).values(created_at=datetime.now(timezone.utc), **fields))
        return res.inserted_primary_key[0]


def get(row_id: int):
    with engine.connect() as conn:
        return conn.execute(select(predictions).where(predictions.c.id == row_id)).mappings().first()


def recent(limit: int = 50):
    with engine.connect() as conn:
        q = select(predictions).order_by(predictions.c.id.desc()).limit(limit)
        return conn.execute(q).mappings().all()


def ping():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))