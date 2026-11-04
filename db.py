"""
db.py - permanent, queryable audit trail of every VisionGuard request.
SQLite by default; any SQLAlchemy URL (e.g. PostgreSQL) via DATABASE_URL.
Stores metadata only - never the image.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table, create_engine,
                        func, insert, inspect, select, text)

import config

engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
meta = MetaData()

predictions = Table(
    "predictions", meta,
    Column("id", Integer, primary_key=True),
    Column("request_id", String(36), unique=True),                    # global ID of the request
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),   # WHEN (UTC)
    # ---- input (no image stored) ----
    Column("filename", String(120)),                                  # sanitized name
    Column("file_size", Integer),
    Column("image_format", String(10)),
    Column("image_width", Integer),
    Column("image_height", Integer),
    Column("image_sha256", String(64), index=True),                   # fingerprint, not the image
    # ---- WHAT was decided ----
    Column("status", String(20), index=True),                         # ok / rejected
    Column("error", String(200)),
    Column("label", String(20), index=True),
    Column("p_ai", Float),
    Column("threshold", Float),
    # ---- HOW confidently ----
    Column("confidence", Float),
    # ---- BY WHICH model ----
    Column("model_name", String(80)),
    Column("model_version", String(20)),
    Column("model_run_id", String(64)),
    Column("model_source", String(80)),
    Column("model_input", Integer),
    Column("latency_ms", Float),
)


def init_db():
    """Create the table, and add any new columns to an older table (simple migration)."""
    meta.create_all(engine)
    existing = {c["name"] for c in inspect(engine).get_columns("predictions")}
    with engine.begin() as conn:
        for col in predictions.columns:
            if col.name not in existing:
                conn.execute(text(f"ALTER TABLE predictions ADD COLUMN {col.name} "
                                    f"{col.type.compile(engine.dialect)}"))
                print(f"[db] added column {col.name}")


def log_request(**fields) -> int:
    fields.setdefault("request_id", str(uuid.uuid4()))
    with engine.begin() as conn:                       # transaction: committed or nothing
        res = conn.execute(insert(predictions).values(created_at=datetime.now(timezone.utc),
                                                      **fields))
        return res.inserted_primary_key[0]


def get(row_id: int):
    with engine.connect() as conn:
        return conn.execute(select(predictions).where(predictions.c.id == row_id)).mappings().first()


def query(status=None, label=None, page=1, per_page=25):
    """Filtered, paginated history (newest first). Returns (rows, total)."""
    cond = []
    if status:
        cond.append(predictions.c.status == status)
    if label:
        cond.append(predictions.c.label == label)
    with engine.connect() as conn:
        total = conn.execute(select(func.count()).select_from(predictions).where(*cond)).scalar()
        rows = conn.execute(select(predictions).where(*cond)
                            .order_by(predictions.c.id.desc())
                            .limit(per_page).offset((page - 1) * per_page)).mappings().all()
    return rows, total


def stats():
    p = predictions.c
    with engine.connect() as conn:
        count = lambda *w: conn.execute(select(func.count()).select_from(predictions).where(*w)).scalar()
        avg_conf = conn.execute(select(func.avg(p.confidence)).where(p.status == "ok")).scalar()
        by_model = conn.execute(
            select(p.model_name, p.model_version, p.label, func.count().label("n"),
                    func.avg(p.confidence).label("avg_conf"))
            .where(p.status == "ok").group_by(p.model_name, p.model_version, p.label)
                        .order_by(p.model_version)).mappings().all()
        return {"total": count(), "ok": count(p.status == "ok"),
                "rejected": count(p.status == "rejected"),
                "ai": count(p.label == "AI-generated"), "real": count(p.label == "Real"),
                "avg_conf": avg_conf, "by_model": by_model}


def ping():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))