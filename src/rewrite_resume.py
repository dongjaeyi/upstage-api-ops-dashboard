from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import requests

try:
    import faiss
except ImportError as e:
    raise SystemExit("faiss is not installed. Run: pip install faiss-cpu") from e

from dotenv import load_dotenv
from src.upstage_client import call_embedding

CHAT_URL = "https://api.upstage.ai/v1/solar/chat/completions"


def safe_json_load(raw: str) -> Dict[str, Any]:
    """
    Robust JSON extraction: parse substring between first '{' and last '}'.
    Handles occasional extra text around JSON.
    """
    if not raw:
        raise ValueError("empty response")
    s = raw.strip()
    l = s.find("{")
    r = s.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError("no json object found")
    sub = s[l : r + 1]
    return json.loads(sub)


def load_meta(meta_path: str) -> Dict[int, Any]:
    """
    meta jsonl: {"faiss_id": <int>, "chunk_id": <any>}
    returns: {faiss_id: chunk_id}
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


def retrieve_topk_chunks(
    jd_text: str,
    db: str,
    index_path: str,
    meta_path: str,
    top_k: int,
    model: str,
) -> List[Dict[str, Any]]:
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

    chunks: List[Dict[str, Any]] = []
    seen_text: Set[str] = set()

    for fid, score in zip(ids, scores):
        if fid < 0:
            continue

        chunk_id = meta.get(int(fid))
        if chunk_id is None:
            continue

        text = fetch_chunk_text(db, chunk_id).strip()
        if not text:
            continue

        # dedupe by normalized text
        key = " ".join(text.split())
        if key in seen_text:
            continue
        seen_text.add(key)

        chunks.append(
            {"faiss_id": int(fid), "score": float(score), "chunk_id": chunk_id, "text": text}
        )

    return chunks


def call_solar_chat(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=180)
    resp.raise_for_status()
    j = resp.json()
    return j["choices"][0]["message"]["content"]


def build_guardrail_messages(
    jd_text: str,
    chunks: List[Dict[str, Any]],
    n_bullets: int,
) -> List[Dict[str, str]]:
    chunk_ids = [str(c["chunk_id"]) for c in chunks]
    chunk_block = "\n\n".join([f"[chunk_id={c['chunk_id']}] {c['text']}" for c in chunks])

    system = (
        "You are a resume rewriting assistant.\n"
        "STRICT GROUNDED RULES:\n"
        "1) Use ONLY the information in the provided chunks. Never add, guess, infer, or fabricate anything.\n"
        "2) Do NOT invent company names, job titles, dates, durations, metrics, tools, projects, or responsibilities.\n"
        "3) Every bullet MUST be directly supported by one or more chunks.\n"
        "4) Citations MUST be the EXACT chunk_id strings from the provided allowed list. No other ids, no numbers.\n"
        "5) If support is insufficient, omit that bullet.\n"
        "OUTPUT MUST BE VALID JSON ONLY.\n"
        "Schema:\n"
        "{\n"
        '  "bullets": [\n'
        '    {"bullet": "...", "citations": ["<chunk_id>", "<chunk_id>"]},\n'
        "    ...\n"
        "  ]\n"
        "}\n"
        "Return JSON only. No markdown. No extra text."
    )

    user = (
        f"JOB DESCRIPTION:\n{jd_text.strip()}\n\n"
        "ALLOWED chunk_id values (citations must use ONLY these exact strings):\n"
        + "\n".join(chunk_ids)
        + "\n\n"
        "RETRIEVED RESUME CHUNKS (ONLY SOURCE OF TRUTH):\n"
        + chunk_block
        + "\n\n"
        "Task:\n"
        f"- Write up to {n_bullets} strong resume bullets tailored to the JD.\n"
        "- Use ONLY the chunks.\n"
        "- Each bullet must include citations as chunk_id strings.\n"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_and_clean_output(out: Dict[str, Any], allowed_chunk_ids: Set[str]) -> Dict[str, Any]:
    """
    Enforce:
    - bullets is a list
    - each bullet has non-empty text
    - citations is list of allowed chunk_id strings
    Drop any bullet that has no valid citations.
    """
    bullets = out.get("bullets", [])
    if not isinstance(bullets, list):
        bullets = []

    cleaned = []
    for b in bullets:
        if not isinstance(b, dict):
            continue

        bullet = (b.get("bullet") or "").strip()
        cites = b.get("citations") or []

        # normalize citations to list[str]
        if isinstance(cites, (str, int, float)):
            cites = [cites]
        if not isinstance(cites, list):
            cites = []

        cites_str = [str(x).strip() for x in cites if str(x).strip()]
        cites_valid = [c for c in cites_str if c in allowed_chunk_ids]

        if not bullet:
            continue
        if not cites_valid:
            continue

        cleaned.append({"bullet": bullet, "citations": cites_valid})

    out["bullets"] = cleaned
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default="ops.db")
    ap.add_argument("--index_path", type=str, default="data/index/faiss_resume.index")
    ap.add_argument("--meta_path", type=str, default="data/index/faiss_meta.jsonl")
    ap.add_argument("--jd_file", type=str, default="")
    ap.add_argument("--jd_text", type=str, default="")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--n_bullets", type=int, default=8)
    ap.add_argument("--embed_model", type=str, default="embedding-query")
    ap.add_argument("--solar_model", type=str, default="solar-pro")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--out_file", type=str, default="outputs/rewrite_bullets.json")
    args = ap.parse_args()

    # Load JD text
    jd_text = args.jd_text
    if args.jd_file:
        p = Path(args.jd_file)
        if not p.exists():
            raise SystemExit(f"JD file not found: {p}. Create it or use --jd_text.")
        jd_text = p.read_text(encoding="utf-8")

    if not jd_text.strip():
        raise SystemExit("JD text is empty.")

    # API key
    load_dotenv()
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise SystemExit("UPSTAGE_API_KEY is missing in env or .env")

    # Retrieve chunks
    chunks = retrieve_topk_chunks(
        jd_text=jd_text,
        db=args.db,
        index_path=args.index_path,
        meta_path=args.meta_path,
        top_k=args.top_k,
        model=args.embed_model,
    )
    if not chunks:
        raise SystemExit("No retrieved chunks. Check index/meta/db.")

    allowed = {str(c["chunk_id"]) for c in chunks}

    messages = build_guardrail_messages(jd_text=jd_text, chunks=chunks, n_bullets=args.n_bullets)

    def request_once(extra_note: str = "") -> str:
        if not extra_note:
            return call_solar_chat(api_key, args.solar_model, messages, args.temperature, args.max_tokens)

        retry_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Your previous answer did not meet the required format.\n"
                    "Fix it now.\n"
                    "Requirements:\n"
                    "- Return VALID JSON only\n"
                    "- citations must be EXACT chunk_id strings from the allowed list\n"
                    "- No markdown, no extra text\n"
                    + extra_note
                ),
            }
        ]
        return call_solar_chat(api_key, args.solar_model, retry_messages, args.temperature, args.max_tokens)

    # Call Solar with one retry
    raw = request_once()
    try:
        out = safe_json_load(raw)
    except Exception:
        raw2 = request_once("\nReturn JSON only. Do not include code fences.")
        try:
            out = safe_json_load(raw2)
        except Exception:
            raise SystemExit("Solar did not return valid JSON after retry. Raw output:\n" + raw2)

    # Validate + clean citations
    out = validate_and_clean_output(out, allowed_chunk_ids=allowed)

    # Save output
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Wrote:", out_path)

    bullets = out.get("bullets", [])
    if not bullets:
        print("[WARN] No grounded bullets produced. Try increasing --top_k or adjust JD text.")
        return

    for i, b in enumerate(bullets, start=1):
        bullet = (b.get("bullet") or "").strip()
        cites = b.get("citations") or []
        print(f"{i}. {bullet}  [chunk_id: {', '.join(str(x) for x in cites)}]")


if __name__ == "__main__":
    main()
