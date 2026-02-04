# src/resume_ingest.py
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.chunking import chunk_from_parse_json
from src.upstage_client import call_document_parse


def normalize_parse_output(obj):
    # tuple이면 dict/list 우선
    if isinstance(obj, tuple):
        obj = next((x for x in obj if isinstance(x, (dict, list))), obj[0])

    # list이면 안에 dict가 있으면 그 dict를 선택 (현재 네 케이스)
    if isinstance(obj, list):
        d = next((x for x in obj if isinstance(x, dict)), None)
        if d is not None:
            return d
        if len(obj) == 1 and isinstance(obj[0], dict):
            return obj[0]

    return obj


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS resumes (
        resume_id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        file_sha256 TEXT NOT NULL,
        pdf_path TEXT NOT NULL,
        parse_json_path TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS resume_chunks (
        chunk_id TEXT PRIMARY KEY,
        resume_id TEXT NOT NULL,
        section TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        source TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(resume_id) REFERENCES resumes(resume_id)
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_resume_chunks_resume_id ON resume_chunks(resume_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_resume_chunks_section ON resume_chunks(section);")
    conn.commit()


def upsert_resume(conn: sqlite3.Connection, resume_id: str, filename: str, file_sha: str,
                  pdf_path: str, parse_json_path: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO resumes(resume_id, filename, file_sha256, pdf_path, parse_json_path, created_at)
    VALUES(?,?,?,?,?,?)
    ON CONFLICT(resume_id) DO UPDATE SET
      filename=excluded.filename,
      file_sha256=excluded.file_sha256,
      pdf_path=excluded.pdf_path,
      parse_json_path=excluded.parse_json_path;
    """, (resume_id, filename, file_sha, pdf_path, parse_json_path, now))
    conn.commit()


def insert_chunks(conn: sqlite3.Connection, resume_id: str, chunks) -> None:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()

    # remove existing chunks for idempotency
    cur.execute("DELETE FROM resume_chunks WHERE resume_id=?", (resume_id,))

    rows = []
    for c in chunks:
        chunk_id = f"{resume_id}:{c.chunk_index}"
        rows.append((chunk_id, resume_id, c.section, c.chunk_index, c.text, c.source, now))

    cur.executemany("""
    INSERT INTO resume_chunks(chunk_id, resume_id, section, chunk_index, text, source, created_at)
    VALUES(?,?,?,?,?,?,?)
    """, rows)
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", type=str, required=True, help="Folder containing resume PDFs")
    ap.add_argument("--db", type=str, default="ops.db", help="SQLite db path")
    ap.add_argument("--out_dir", type=str, default="data/parsed/resumes", help="Where to save parse JSON")
    ap.add_argument("--force", action="store_true", help="Re-parse even if parse JSON exists")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db)
    ensure_tables(conn)

    # Collect PDFs (include .pdf and .PDF)
    pdfs = sorted([p for p in pdf_dir.glob("*.pdf") if p.is_file()])
    pdfs += sorted([p for p in pdf_dir.glob("*.PDF") if p.is_file()])

    # Deduplicate by resolved path
    seen = set()
    pdfs2 = []
    for p in pdfs:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        pdfs2.append(p)
    pdfs = pdfs2

    if not pdfs:
        raise SystemExit(f"No PDFs found in: {pdf_dir}")

    print(f"[INFO] Found PDFs: {len(pdfs)} in {pdf_dir}")
    for p in pdfs:
        print(" -", p.name)

    ok = 0
    failed = 0

    for pdf in pdfs:
        try:
            file_sha = sha256_file(pdf)
            resume_id = file_sha[:16]  # 파일 내용 기반 ID
            json_path = out_dir / f"{resume_id}.json"

            # Load or parse
            if json_path.exists() and not args.force:
                parse_json = json.loads(json_path.read_text(encoding="utf-8"))
                parse_json = normalize_parse_output(parse_json)
            else:
                raw = call_document_parse(str(pdf))
                parse_json = normalize_parse_output(raw)
                json_path.write_text(
                    json.dumps(parse_json, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

            # DB write
            upsert_resume(
                conn=conn,
                resume_id=resume_id,
                filename=pdf.name,
                file_sha=file_sha,
                pdf_path=str(pdf.resolve()),
                parse_json_path=str(json_path.resolve()),
            )

            chunks = chunk_from_parse_json(parse_json)
            insert_chunks(conn, resume_id, chunks)

            print(f"[OK] {pdf.name} -> resume_id={resume_id} chunks={len(chunks)}")
            ok += 1

        except Exception as e:
            print(f"[ERR] {pdf.name} failed: {e}")
            failed += 1
            continue

    conn.close()
    print(f"[DONE] ok={ok} failed={failed}")


if __name__ == "__main__":
    main()
