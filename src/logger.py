import sqlite3
from datetime import datetime, timezone
import uuid
RUN_ID = str(uuid.uuid4())

def init_db(db_path="ops.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 최신 스키마: api_calls에 run_id/stage/api_name/ok/error_type 포함
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
        error_message TEXT,
        run_id TEXT,
        stage TEXT,
        api_name TEXT,
        ok INTEGER,
        error_type TEXT
    )
    """)

    # 인덱스(선택이지만 추천: 조회/집계 빨라짐)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_api_calls_run_id ON api_calls(run_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_api_calls_ts ON api_calls(ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_api_calls_api_name ON api_calls(api_name)")

    conn.commit()
    conn.close()

def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")

def _infer_api_name(endpoint: str | None, model: str | None) -> str:
    ep = (endpoint or "").lower()
    md = (model or "").lower()

    if "document" in ep or "digitization" in ep or "parse" in md:
        return "document_parse"
    if "embedding" in ep or "embedding" in md:
        return "embeddings"
    if "chat" in ep or "solar" in ep or "chat" in md or "solar" in md:
        return "solar_chat"
    return model or "unknown"

def _classify_error_type(status_code: int | None, error_message: str | None) -> str | None:
    if status_code == 429:
        return "rate_limited"
    if status_code is not None and 500 <= status_code <= 599:
        return "server_error"
    if status_code is not None and 400 <= status_code <= 499:
        return "client_error"
    if error_message and "timeout" in error_message.lower():
        return "timeout"
    return None

def log_call(payload: dict, db_path="ops.db"):
    endpoint = payload.get("endpoint")
    model = payload.get("model")
    filename = payload.get("filename")
    status_code = payload.get("status_code")
    error_message = payload.get("error_message")

    # 없으면 기본값
    run_id = payload.get("run_id") or RUN_ID
    stage = payload.get("stage") or "unknown"

    api_name = payload.get("api_name") or _infer_api_name(endpoint, model)

    ok = payload.get("ok")
    if ok is None:
        ok = 1 if (status_code is not None and 200 <= int(status_code) <= 299) else 0

    error_type = payload.get("error_type") or _classify_error_type(
        int(status_code) if status_code is not None else None,
        error_message
    )

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO api_calls
    (ts, endpoint, model, filename, status_code, latency_ms, response_bytes, error_code, error_message,
     run_id, stage, api_name, ok, error_type)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        _utc_now_iso(),
        endpoint,
        model,
        filename,
        status_code,
        payload.get("latency_ms"),
        payload.get("response_bytes"),
        payload.get("error_code"),
        error_message,
        run_id,
        stage,
        api_name,
        ok,
        error_type,
    ))
    conn.commit()
    conn.close()

