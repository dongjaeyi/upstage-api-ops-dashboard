from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import List, Tuple, Any, Dict

import numpy as np

try:
    import faiss  # faiss-cpu
except ImportError as e:
    raise SystemExit("faiss is not installed. Run: pip install faiss-cpu") from e

from src.upstage_client import call_embedding


def fetch_chunks(db_path: str) -> List[Tuple[Any, str]]:
    """
    Assumes resume_chunks has (chunk_id, text).
    If your DB schema differs, adjust this query or implement column detection.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT chunk_id, text FROM resume_chunks ORDER BY chunk_id"
    ).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def batched(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default="ops.db")
    ap.add_argument("--out_dir", type=str, default="data/index")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--model", type=str, default="embedding-query")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "faiss_resume.index"
    meta_path = out_dir / "faiss_meta.jsonl"

    if (index_path.exists() or meta_path.exists()) and not args.force:
        raise SystemExit(
            "Index/meta already exists. Use --force to overwrite.\n"
            f"- {index_path}\n- {meta_path}"
        )

    pairs = fetch_chunks(args.db)
    if not pairs:
        raise SystemExit("No chunks found in DB. Run: python -m src.resume_ingest ... first")

    # remove empty texts (embedding API disallows empty strings)
    pairs = [(cid, t) for cid, t in pairs if isinstance(t, str) and t.strip()]
    if not pairs:
        raise SystemExit("All chunk texts are empty after filtering.")

    chunk_ids = [cid for cid, _ in pairs]
    texts = [t.strip() for _, t in pairs]

    print(f"[INFO] chunks={len(texts)} batch={args.batch} model={args.model}")

    vectors_list: List[np.ndarray] = []
    meta_records: List[Dict[str, Any]] = []

    row_id = 0  # this becomes faiss_id (0..N-1)

    for batch_idx, batch in enumerate(batched(list(zip(chunk_ids, texts)), args.batch), start=1):
        ids_b = [x[0] for x in batch]
        txt_b = [x[1] for x in batch]

        emb = call_embedding(txt_b, model=args.model)  # expects list[str] -> list[list[float]]
        if not emb:
            raise SystemExit("Embedding API returned empty embeddings.")

        arr = np.array(emb, dtype="float32")
        vectors_list.append(arr)

        for cid in ids_b:
            meta_records.append({"faiss_id": row_id, "chunk_id": cid})
            row_id += 1

        print(f"[INFO] batch={batch_idx} embedded={len(ids_b)} total={row_id}")

    vectors = np.vstack(vectors_list).astype("float32")

    # cosine similarity: normalize + inner product
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # save files
    faiss.write_index(index, str(index_path))
    with meta_path.open("w", encoding="utf-8") as f:
        for rec in meta_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[OK] FAISS index saved:", index_path)
    print("[OK] Meta jsonl saved:", meta_path)
    print(f"[INFO] ntotal={index.ntotal} dim={index.d}")


if __name__ == "__main__":
    main()
