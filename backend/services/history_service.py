"""SQLite-backed detection history. Stores metadata only, not raw images
(the Grad-CAM/original are kept as small artifact files on disk, referenced
by path, so the DB stays lightweight)."""
import sqlite3
import json
import os
from contextlib import contextmanager

from config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    created_at TEXT NOT NULL,
    prediction TEXT NOT NULL,
    fake_probability REAL NOT NULL,
    real_probability REAL NOT NULL,
    sha256 TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    file_size INTEGER,
    report_available INTEGER DEFAULT 0,
    analysis_json TEXT NOT NULL
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute(_SCHEMA)


def save_analysis(analysis: dict):
    with _connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO history
               (id, filename, created_at, prediction, fake_probability, real_probability,
                sha256, width, height, file_size, report_available, analysis_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                analysis["analysis_id"],
                analysis["filename"],
                analysis["timestamp"],
                analysis["prediction"],
                analysis["fake_probability"],
                analysis["real_probability"],
                analysis["hash"]["sha256"],
                analysis["image"]["width"],
                analysis["image"]["height"],
                analysis["image"]["file_size_bytes"],
                0,
                json.dumps(analysis),
            ),
        )


def mark_report_available(analysis_id: str):
    with _connect() as conn:
        conn.execute("UPDATE history SET report_available = 1 WHERE id = ?", (analysis_id,))


def list_history(limit: int = 200, prediction_filter: str | None = None, search: str | None = None):
    query = "SELECT * FROM history"
    clauses, params = [], []
    if prediction_filter:
        clauses.append("prediction = ?")
        params.append(prediction_filter.upper())
    if search:
        clauses.append("filename LIKE ?")
        params.append(f"%{search}%")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_summary(r) for r in rows]


def get_analysis(analysis_id: str):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM history WHERE id = ?", (analysis_id,)).fetchone()
    if not row:
        return None
    return json.loads(row["analysis_json"])


def delete_analysis(analysis_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM history WHERE id = ?", (analysis_id,))
        return cur.rowcount > 0


def dashboard_stats():
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM history").fetchone()["c"]
        fake = conn.execute("SELECT COUNT(*) c FROM history WHERE prediction = 'FAKE'").fetchone()["c"]
        real = conn.execute("SELECT COUNT(*) c FROM history WHERE prediction = 'REAL'").fetchone()["c"]
        avg_conf = conn.execute(
            "SELECT AVG(MAX(fake_probability, real_probability)) a FROM history"
        ).fetchone()["a"]
        recent = conn.execute(
            "SELECT * FROM history ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        timeline = conn.execute(
            """SELECT substr(created_at, 1, 10) as day, prediction, COUNT(*) c
               FROM history GROUP BY day, prediction ORDER BY day ASC"""
        ).fetchall()
    return {
        "total_analyses": total,
        "fake_detected": fake,
        "real_detected": real,
        "average_model_probability": round(avg_conf, 2) if avg_conf else 0,
        "recent": [_row_summary(r) for r in recent],
        "timeline": [dict(t) for t in timeline],
    }


def _row_summary(row):
    return {
        "analysis_id": row["id"],
        "filename": row["filename"],
        "timestamp": row["created_at"],
        "prediction": row["prediction"],
        "fake_probability": row["fake_probability"],
        "real_probability": row["real_probability"],
        "sha256": row["sha256"],
        "width": row["width"],
        "height": row["height"],
        "file_size_bytes": row["file_size"],
        "report_available": bool(row["report_available"]),
    }
