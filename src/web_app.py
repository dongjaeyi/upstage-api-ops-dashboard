from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from flask import Flask, render_template, request
import sqlite3
from datetime import datetime, timedelta, timezone

DB_PATH = "ops.db"

@dataclass
class RunResult:
    ok: bool
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    output_json_path: Path
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def repo_root() -> Path:
    # src/web_app.py -> repo root is parent of src
    return Path(__file__).resolve().parents[1]


def build_overview(pipeline_mode: str, cmd: list[str] | None = None) -> dict:
    """
    Returns per-step status for the Pipeline Overview UI.
    status: "run" | "skipped" | "reuse"
    """
    mode = (pipeline_mode or "all").strip()
    cmd = cmd or []

    def has(flag: str) -> bool:
        return flag in cmd

    if mode == "all":
        # all mode includes ingest/index, but may be skipped via flags
        ingest = "skipped" if has("--skip_ingest") else "run"
        index = "skipped" if has("--skip_index") else "run"
        chunking = "reuse"  # chunks are typically already in DB after ingest
    else:
        # summary mode reuses existing DB/index (does not run ingest/index)
        ingest = "reuse"
        index = "reuse"
        chunking = "reuse"

    return {
        "mode": mode,
        "ingest": ingest,
        "chunking": chunking,
        "embedding": "run",
        "search": "run",
        "rewrite": "run",
        "output": "run",
        # index is not shown in your current UI nodes, but we keep it in case you add it later
        "index": index,
    }


def ensure_outputs_dir(root: Path) -> Path:
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_jd_to_temp(root: Path, jd_text: str) -> Path:
    out_dir = ensure_outputs_dir(root)
    fd, tmp_path = tempfile.mkstemp(prefix="jd_", suffix=".txt", dir=str(out_dir))
    os.close(fd)
    p = Path(tmp_path)
    p.write_text(jd_text, encoding="utf-8")
    return p


def save_upload_to_outputs(root: Path, file_storage) -> Path:
    out_dir = ensure_outputs_dir(root)
    filename = file_storage.filename or "uploaded_jd.txt"
    safe_name = "".join(ch for ch in filename if ch.isalnum() or ch in ("-", "_", ".", " ")).strip()
    if not safe_name:
        safe_name = "uploaded_jd.txt"
    target = out_dir / safe_name
    file_storage.save(str(target))
    return target


def build_pipeline_cmd(
    jd_file: Path,
    confidence: str,
    resume_mode: str,
    top_k: int,
    final_k: int,
    max_per_resume: int,
    max_chars: int,
    pipeline_mode: str = "all",
) -> list[str]:
    # Always run from repo root: python -m src.pipeline ...
    py = sys.executable

    if pipeline_mode == "summary":
        cmd = [
            py, "-m", "src.pipeline", "summary",
            "--jd_file", str(jd_file),
            "--confidence", confidence,
            "--resume_mode", resume_mode,
            "--top_k", str(top_k),
            "--final_k", str(final_k),
            "--max_per_resume", str(max_per_resume),
            "--max_chars", str(max_chars),
        ]
    else:
        # Fast mode: all + skip ingest/index
        cmd = [
            py, "-m", "src.pipeline", "all",
            "--skip_ingest", "--skip_index",
            "--jd_file", str(jd_file),
            "--confidence", confidence,
            "--resume_mode", resume_mode,
            "--top_k", str(top_k),
            "--final_k", str(final_k),
            "--max_per_resume", str(max_per_resume),
            "--max_chars", str(max_chars),
        ]

    return cmd


def run_pipeline(root: Path, cmd: list[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def read_summary_json(root: Path) -> Tuple[Optional[dict], Path, Optional[str]]:
    out_path = root / "outputs" / "rewrite_summary.json"
    if not out_path.exists():
        return None, out_path, "outputs/rewrite_summary.json not found. Pipeline may have failed before writing output."
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        return data, out_path, None
    except Exception as e:
        return None, out_path, f"Failed to parse rewrite_summary.json: {e}"


def parse_int(form_value: str, default: int, min_v: int, max_v: int) -> int:
    try:
        v = int(form_value)
    except Exception:
        return default
    return max(min_v, min(max_v, v))


# ----- Flask app setup -----
ROOT = repo_root()
SRC_DIR = Path(__file__).resolve().parent  # .../src

app = Flask(
    __name__,
    template_folder=str(SRC_DIR / "templates"),
    static_folder=str(SRC_DIR / "static"),
)


@app.get("/")
def index():
    defaults = {
        "confidence": "balanced",
        "resume_mode": "diverse",
        "top_k": 12,
        "final_k": 6,
        "max_per_resume": 3,
        "max_chars": 500,
        "pipeline_mode": "all",
        "jd_text": "",
    }
    overview = build_overview(defaults["pipeline_mode"], cmd=None)

    api_scope = request.args.get("api_scope", "48h")
    api_panel = fetch_api_panel(db_path="ops.db", scope=api_scope, hours=48)


    return render_template(
        "index.html",
        result=rr,
        defaults=defaults,
        overview=overview,
        api_panel=api_panel,
        api_scope=api_scope,
    )

@app.post("/run")
def run():
    api_scope = request.form.get("api_scope", "48h")
    api_panel = fetch_api_panel(db_path="ops.db", scope=api_scope, hours=48)
    root = ROOT  # always repo root
    api_panel = fetch_api_panel(db_path="ops.db", hours=48)
    jd_text = (request.form.get("jd_text") or "").strip()
    jd_file_upload = request.files.get("jd_file")

    confidence = (request.form.get("confidence") or "balanced").strip()
    resume_mode = (request.form.get("resume_mode") or "diverse").strip()
    pipeline_mode = (request.form.get("pipeline_mode") or "all").strip()

    top_k = parse_int(request.form.get("top_k", ""), default=12, min_v=1, max_v=50)
    final_k = parse_int(request.form.get("final_k", ""), default=6, min_v=1, max_v=50)
    max_per_resume = parse_int(request.form.get("max_per_resume", ""), default=3, min_v=1, max_v=20)
    max_chars = parse_int(request.form.get("max_chars", ""), default=500, min_v=100, max_v=2000)

    # Decide JD source: upload preferred if present, else textarea
    jd_path: Optional[Path] = None
    if jd_file_upload and jd_file_upload.filename:
        jd_path = save_upload_to_outputs(root, jd_file_upload)
    elif jd_text:
        jd_path = save_jd_to_temp(root, jd_text)
    else:
        defaults = {
            "confidence": confidence,
            "resume_mode": resume_mode,
            "top_k": top_k,
            "final_k": final_k,
            "max_per_resume": max_per_resume,
            "max_chars": max_chars,
            "pipeline_mode": pipeline_mode,
            "jd_text": jd_text,
        }
        rr = RunResult(
            ok=False,
            cmd=[],
            returncode=2,
            stdout="",
            stderr="",
            output_json_path=root / "outputs" / "rewrite_summary.json",
            error="JD text or JD file is required.",
        )
        overview = build_overview(pipeline_mode, cmd=[])
        return render_template(
            "index.html",
            result=rr,
            defaults=defaults,
            overview=overview,
            api_panel=api_panel,
            api_scope=api_scope,  # ✅ 추가
        )

    cmd = build_pipeline_cmd(
        jd_file=jd_path,
        confidence=confidence,
        resume_mode=resume_mode,
        top_k=top_k,
        final_k=final_k,
        max_per_resume=max_per_resume,
        max_chars=max_chars,
        pipeline_mode=pipeline_mode,
    )

    code, out, err = run_pipeline(root, cmd)
    data, out_json_path, json_err = read_summary_json(root)

    ok = (code == 0) and (data is not None) and (json_err is None)
    rr = RunResult(
        ok=ok,
        cmd=cmd,
        returncode=code,
        stdout=out,
        stderr=err,
        output_json_path=out_json_path,
        data=data if ok else data,
        error=None if ok else (json_err or f"Pipeline failed (return code {code}). See stderr."),
    )

    defaults = {
        "confidence": confidence,
        "resume_mode": resume_mode,
        "top_k": top_k,
        "final_k": final_k,
        "max_per_resume": max_per_resume,
        "max_chars": max_chars,
        "pipeline_mode": pipeline_mode,
        "jd_text": jd_text,
    }

    overview = build_overview(pipeline_mode, cmd=cmd)
    
    return render_template(
        "index.html",
        result=rr,
        defaults=defaults,
        overview=overview,
        api_panel=api_panel,
        api_scope=api_scope,
    )

def fetch_api_panel(db_path: str = "ops.db", scope: str = "48h", hours: int = 48) -> dict:
    """
    scope:
      - "48h": last N hours
      - "run": latest run_id only
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # latest run id (for scope=run)
    latest_run_id_row = cur.execute(
        "SELECT run_id FROM api_calls WHERE run_id IS NOT NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    latest_run_id = latest_run_id_row[0] if latest_run_id_row else None

    # where clause
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")

    if scope == "run" and latest_run_id:
        where_sql = "run_id = ?"
        where_args = (latest_run_id,)
        window_label = f"this run ({latest_run_id[:8]})"
        since_label = None
    else:
        where_sql = "ts >= ?"
        where_args = (since,)
        window_label = f"last {hours}h"
        since_label = since

    # KPI
    total_calls = cur.execute(
        f"SELECT COUNT(*) FROM api_calls WHERE {where_sql}",
        where_args
    ).fetchone()[0]

    total_time = cur.execute(
        f"SELECT COALESCE(SUM(latency_ms),0) FROM api_calls WHERE {where_sql}",
        where_args
    ).fetchone()[0]

    failures = cur.execute(
        f"SELECT COALESCE(SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END),0) FROM api_calls WHERE {where_sql}",
        where_args
    ).fetchone()[0]

    hotspot = cur.execute(
        f"""
        SELECT api_name
        FROM api_calls
        WHERE {where_sql}
        GROUP BY api_name
        ORDER BY SUM(latency_ms) DESC
        LIMIT 1
        """,
        where_args
    ).fetchone()
    hotspot = hotspot[0] if hotspot and hotspot[0] else None

    # (2) Rate/Failure breakdown counts
    rate_429 = cur.execute(
        f"SELECT COALESCE(SUM(CASE WHEN status_code=429 THEN 1 ELSE 0 END),0) FROM api_calls WHERE {where_sql}",
        where_args
    ).fetchone()[0]

    err_5xx = cur.execute(
        f"SELECT COALESCE(SUM(CASE WHEN status_code BETWEEN 500 AND 599 THEN 1 ELSE 0 END),0) FROM api_calls WHERE {where_sql}",
        where_args
    ).fetchone()[0]

    timeouts = cur.execute(
        f"SELECT COALESCE(SUM(CASE WHEN error_type='timeout' THEN 1 ELSE 0 END),0) FROM api_calls WHERE {where_sql}",
        where_args
    ).fetchone()[0]

    # breakdown table
    breakdown_rows = cur.execute(
        f"""
        SELECT
          api_name,
          COUNT(*) AS calls,
          ROUND(AVG(latency_ms),0) AS avg_ms,
          MAX(latency_ms) AS max_ms,
          SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END) AS fails,
          SUM(CASE WHEN status_code=429 OR error_type='rate_limited' THEN 1 ELSE 0 END) AS rate_limited
        FROM api_calls
        WHERE {where_sql}
        GROUP BY api_name
        ORDER BY calls DESC
        """,
        where_args
    ).fetchall()

    breakdown = [{
        "api_name": r[0] or "unknown",
        "calls": int(r[1] or 0),
        "avg_ms": int(r[2] or 0),
        "max_ms": int(r[3] or 0),
        "fails": int(r[4] or 0),
        "rate_limited": int(r[5] or 0),
    } for r in breakdown_rows]

    # recent events
    recent_rows = cur.execute(
        f"""
        SELECT ts, stage, api_name, ok, latency_ms, status_code, error_type
        FROM api_calls
        WHERE {where_sql}
        ORDER BY id DESC
        LIMIT 10
        """,
        where_args
    ).fetchall()

    recent = [{
        "ts": r[0],
        "stage": r[1] or "",
        "api_name": r[2] or "",
        "ok": bool(r[3]),
        "latency_ms": int(r[4] or 0),
        "status_code": r[5],
        "error_type": r[6] or "",
    } for r in recent_rows]

    conn.close()

    return {
        "scope": scope,
        "window_label": window_label,
        "since": since_label,
        "hours": hours,
        "latest_run_id": latest_run_id,
        "kpis": {
            "total_calls": total_calls,
            "total_time_ms": total_time,
            "failures": failures,
            "hotspot": hotspot,
            "rate_429": int(rate_429 or 0),
            "err_5xx": int(err_5xx or 0),
            "timeouts": int(timeouts or 0),
        },
        "breakdown": breakdown,
        "recent": recent,
    }


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
