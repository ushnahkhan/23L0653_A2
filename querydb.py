"""
querydb.py - example SQL queries on the VisionGuard audit trail (Task 5)
Run: python querydb.py
"""
import pandas as pd

from db import engine

QUERIES = {
    "Last 5 decisions (what, when, which model, how confident)": """
        SELECT id, created_at, filename, label, ROUND(confidence, 3) AS confidence,
               model_name, model_version, SUBSTR(model_run_id, 1, 8) AS run
        FROM predictions WHERE status = 'ok' ORDER BY id DESC LIMIT 5""",
    "Decisions per model version": """
        SELECT model_version, label, COUNT(*) AS n, ROUND(AVG(confidence), 3) AS avg_confidence
        FROM predictions WHERE status = 'ok' GROUP BY model_version, label""",
    "Low-confidence decisions (< 65 %)": """
        SELECT id, created_at, filename, label, ROUND(confidence, 3) AS confidence
        FROM predictions WHERE status = 'ok' AND confidence < 0.65 ORDER BY id DESC""",
    "Rejection reasons": """
        SELECT error, COUNT(*) AS n FROM predictions WHERE status = 'rejected' GROUP BY error""",
    "Same image uploaded more than once (by SHA-256)": """
        SELECT SUBSTR(image_sha256, 1, 12) AS image_hash, COUNT(*) AS times, MIN(label) AS label
        FROM predictions WHERE image_sha256 IS NOT NULL
        GROUP BY image_sha256 HAVING COUNT(*) > 1""",
}

pd.set_option("display.width", 200)
for title, sql in QUERIES.items():
    print(f"\n=== {title} ===")
    df = pd.read_sql(sql, engine)
    print(df.to_string(index=False) if len(df) else "(no rows)")