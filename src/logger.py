import sqlite3
from datetime import datetime

def init_db(db_path="ops.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS api_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        endpoint TEXT,
        model TEXT,
        filename TEXT,
        status_code INTEGER,
        latency_ms INTEGER,
        response_bytes INTEGER,
        error_code TEXT,
        error_message TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_call(payload: dict, db_path="ops.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO api_calls (ts, endpoint, model, filename, status_code, latency_ms, response_bytes, error_code, error_message)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        payload.get("endpoint"),
        payload.get("model"),
        payload.get("filename"),
        payload.get("status_code"),
        payload.get("latency_ms"),
        payload.get("response_bytes"),
        payload.get("error_code"),
        payload.get("error_message"),
    ))
    conn.commit()
    conn.close()
