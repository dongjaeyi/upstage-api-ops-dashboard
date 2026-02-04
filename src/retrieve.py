from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import faiss
except ImportError as e:
    raise SystemExit("faiss is not installed. Run: pip install faiss-cpu") from e

from src.upstage_client import call_embedding


def load_meta(meta_path: str) -> Dict[int, Any]:
    """
    returns: {faiss_id(int): chunk_id(any)}
    """
    mapping: Dict[int, Any] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mapping[int(rec["faiss_id"])] = rec["chunk_id"]
    return mapping


def fetch_chunk_text(db_path: str, chunk_id: Any) -> str:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    row = cur.execute(
        "SELECT text FROM resume_chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    conn.close()
    return (row[0] if row and row[0] else "")


def retrieve(jd_text: str, db: str, index_path: str, meta_path: str, top_k: int, model: str):
    if not jd_text.strip():
        raise SystemExit("JD text is empty.")

    index = faiss.read_index(index_path)
    meta = load_meta(meta_path)

    emb = call_embedding([jd_text.strip()], model=model)
    if not emb:
        raise SystemExit("Embedding API returned empty embeddings for JD.")
    q = np.array(emb, dtype="float32")
    faiss.normalize_L2(q)

    scores, ids = index.search(q, top_k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results = []
    for rank, (fid, score) in enumerate(zip(ids, scores), start=1):
        if fid < 0:
            continue
        chunk_id = meta.get(int(fid))
        text = fetch_chunk_text(db, chunk_id)
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "faiss_id": int(fid),
                "chunk_id": chunk_id,
                "text": text,
            }
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default="ops.db")
    ap.add_argument("--index_path", type=str, default="data/index/faiss_resume.index")
    ap.add_argument("--meta_path", type=str, default="data/index/faiss_meta.jsonl")
    ap.add_argument("--jd_file", type=str, default="")
    ap.add_argument("--jd_text", type=str, default="")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--model", type=str, default="embedding-query")
    args = ap.parse_args()

    jd_text = args.jd_text
    if args.jd_file:
        jd_text = Path(args.jd_file).read_text(encoding="utf-8")

    results = retrieve(
        jd_text=jd_text,
        db=args.db,
        index_path=args.index_path,
        meta_path=args.meta_path,
        top_k=args.top_k,
        model=args.model,
    )

    for r in results:
        print("=" * 80)
        print(
            f"rank={r['rank']} score={r['score']:.4f} faiss_id={r['faiss_id']} chunk_id={r['chunk_id']}"
        )
        print((r["text"] or "").strip())


if __name__ == "__main__":
    main()
