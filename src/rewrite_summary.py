from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import requests
from time import perf_counter
from src.logger import log_call

try:
    import faiss
except ImportError as e:
    raise SystemExit("faiss is not installed. Run: pip install faiss-cpu") from e

from dotenv import load_dotenv
from src.upstage_client import call_embedding


CHAT_URL = "https://api.upstage.ai/v1/solar/chat/completions"


# -----------------------------
# Utilities
# -----------------------------
def safe_json_load(raw: str) -> Dict[str, Any]:
    if not raw:
        raise ValueError("empty response")
    s = raw.strip()
    l = s.find("{")
    r = s.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError("no json object found")
    return json.loads(s[l : r + 1])


def load_meta(meta_path: str) -> Dict[int, Any]:
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
    row = cur.execute("SELECT text FROM resume_chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
    conn.close()
    return (row[0] if row and row[0] else "")


def extract_resume_key(chunk_id: Any) -> str:
    """
    chunk_id like: "759ce1b626c0a415:1"
    resume_key = part before ':'
    If unknown format, fallback to str(chunk_id).
    """
    s = str(chunk_id)
    if ":" in s:
        return s.split(":", 1)[0]
    return s


def normalize_text_key(text: str) -> str:
    return " ".join((text or "").split()).strip()


def is_low_info(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) < 25:
        return True
    # date-like only
    if re.fullmatch(r"[0-9/\-\s]+", t):
        return True
    if "page" in t.lower() and "of" in t.lower():
        return True
    letters = sum(ch.isalpha() for ch in t)
    ratio = letters / max(len(t), 1)
    return ratio < 0.20


def is_summary_like(text: str) -> bool:
    """
    Heuristic: summary sentences often have years of experience, role label, compact claims.
    Tune as needed.
    """
    t = (text or "").strip()
    if not t:
        return False
    tl = t.lower()
    patterns = [
        "years of experience",
        "technical product manager",
        "product manager",
        "summary",
        "cross-functional",
        "stakeholder",
        "roadmap",
        "release readiness",
        "product operations",
    ]
    hit = sum(1 for p in patterns if p in tl)
    return hit >= 2 and len(t) <= 650


def call_solar_chat(api_key: str, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

    t0 = perf_counter()
    status_code = None
    resp_bytes = None
    err_msg = None
    ok = 0

    try:
        resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=180)
        status_code = resp.status_code
        resp_bytes = len(resp.content) if resp.content is not None else None
        resp.raise_for_status()

        j = resp.json()
        ok = 1
        return j["choices"][0]["message"]["content"]

    except Exception as e:
        err_msg = str(e)
        raise

    finally:
        latency_ms = int((perf_counter() - t0) * 1000)

        # filename은 꼭 필요하지 않아서 None으로 둬도 됨
        log_call({
            "endpoint": CHAT_URL,
            "model": model,
            "filename": None,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "response_bytes": resp_bytes,
            "error_code": None,
            "error_message": err_msg,
            "stage": "rewrite_summary",
            "api_name": "solar_chat",
            "ok": ok,
        })



def dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in chunks:
        key = normalize_text_key(c.get("text", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def diversify_chunks(
    chunks: List[Dict[str, Any]],
    mode: str,
    max_per_resume: int,
    resume_keys_allow: Set[str] | None,
    final_k: int,
) -> List[Dict[str, Any]]:
    """
    mode:
      - "all": keep score order
      - "diverse": round-robin across resume_key
      - "only": use only resume_keys_allow
    """
    if mode not in {"all", "diverse", "only"}:
        mode = "all"

    # optional allowlist filter
    if mode == "only":
        if not resume_keys_allow:
            # no allowlist provided, return empty
            return []
        chunks = [c for c in chunks if c["resume_key"] in resume_keys_allow]

    if mode == "all":
        # cap per resume if requested
        if max_per_resume > 0:
            counts = defaultdict(int)
            out = []
            for c in chunks:
                rk = c["resume_key"]
                if counts[rk] >= max_per_resume:
                    continue
                counts[rk] += 1
                out.append(c)
                if len(out) >= final_k:
                    break
            return out
        return chunks[:final_k]

    # diverse mode: round-robin
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        grouped[c["resume_key"]].append(c)

    # score order inside each group
    for rk in grouped:
        grouped[rk].sort(key=lambda x: x["score"], reverse=True)

    resume_keys = list(grouped.keys())
    # stable order by best score per resume (so strongest resumes go first)
    resume_keys.sort(key=lambda rk: grouped[rk][0]["score"] if grouped[rk] else -1, reverse=True)

    out = []
    counts = defaultdict(int)
    while len(out) < final_k:
        progressed = False
        for rk in resume_keys:
            if counts[rk] >= max_per_resume:
                continue
            if grouped[rk]:
                out.append(grouped[rk].pop(0))
                counts[rk] += 1
                progressed = True
                if len(out) >= final_k:
                    break
        if not progressed:
            break
    return out


# -----------------------------
# Confidence settings
# -----------------------------
CONF_SETTINGS = {
    "strict": {
        "temperature": 0.1,
        "banned_terms": ["expert", "proven", "industry-leading", "world-class", "best-in-class", "highly"],
        "style_note": (
            "STRICT STYLE:\n"
            "- Keep tone factual and conservative.\n"
            "- Avoid strong claims and superlatives.\n"
            "- Prefer 'experience in', 'worked on', 'supported' instead of 'expert' or 'proven'.\n"
        ),
    },
    "balanced": {
        "temperature": 0.2,
        "banned_terms": ["industry-leading", "world-class", "best-in-class"],
        "style_note": (
            "BALANCED STYLE:\n"
            "- Professional resume tone.\n"
            "- Moderate confidence words allowed (experienced, skilled), but no hype.\n"
        ),
    },
    "bold": {
        "temperature": 0.3,
        "banned_terms": ["world-class", "best-in-class"],  # still ban extreme hype
        "style_note": (
            "BOLD STYLE:\n"
            "- Confident but still grounded.\n"
            "- You may use 'proven' or 'expert' only if the supporting chunks clearly justify it.\n"
            "- Never invent facts.\n"
        ),
    },
}


def build_summary_messages(
    jd_text: str,
    chunks: List[Dict[str, Any]],
    max_chars: int,
    confidence: str,
) -> List[Dict[str, str]]:
    conf = CONF_SETTINGS.get(confidence, CONF_SETTINGS["balanced"])
    chunk_ids = [str(c["chunk_id"]) for c in chunks]
    chunk_block = "\n\n".join([f"[chunk_id={c['chunk_id']}] {c['text']}" for c in chunks])

    system = (
        "You are a resume summary rewriting assistant.\n"
        "STRICT GROUNDED RULES:\n"
        "1) Use ONLY the information in the provided chunks. Never add, guess, infer, or fabricate anything.\n"
        "2) Do NOT invent company names, job titles, dates, durations, metrics, tools, projects, or achievements.\n"
        "3) You may rephrase and reorder supported content to match the JD, but must stay fully grounded.\n"
        "4) Citations MUST be the EXACT chunk_id strings from the allowed list.\n"
        "   Provide 2 to 5 citations only, no duplicates.\n"
        "5) Output must be VALID JSON only.\n"
        f"{conf['style_note']}\n"
        "Output schema:\n"
        "{\n"
        '  "summary": "<=MAX_CHARS characters>",\n'
        '  "citations": ["<chunk_id>", "<chunk_id>"]\n'
        "}\n"
        "Return JSON only. No markdown. No extra text."
    ).replace("MAX_CHARS", str(max_chars))

    banned = CONF_SETTINGS.get(confidence, CONF_SETTINGS["balanced"])["banned_terms"]
    banned_line = ""
    if banned:
        banned_line = "BANNED WORDS (do not use): " + ", ".join(banned) + "\n"

    user = (
        f"JOB DESCRIPTION:\n{jd_text.strip()}\n\n"
        f"{banned_line}"
        "ALLOWED chunk_id values (citations must use ONLY these exact strings):\n"
        + "\n".join(chunk_ids)
        + "\n\n"
        "RETRIEVED RESUME CHUNKS (ONLY SOURCE OF TRUTH):\n"
        + chunk_block
        + "\n\n"
        f"Task:\n"
        f"- Write ONE resume summary tailored to the JD.\n"
        f"- The summary must be <= {max_chars} characters.\n"
        "- Use ONLY the chunks.\n"
        "- Return JSON with summary and citations.\n"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def truncate_to_complete_sentence(text: str, max_chars: int) -> str:
    """
    Truncate to <= max_chars, preferring to end at a completed sentence.
    Priority:
      1) last '.', '!', '?' within limit (and not too early)
      2) last space within limit
      3) hard cut
    """
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t

    cut = t[:max_chars]

    # Prefer sentence boundary
    last_end = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if last_end >= 60:  # avoid overly short outputs
        return cut[: last_end + 1].strip()

    # Fallback: word boundary
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
        return cut.rstrip(" ,.;:")

    # Hard cut fallback
    return cut.rstrip(" ,.;:")

def validate_summary_output(out: Dict[str, Any], allowed: Set[str], max_chars: int) -> Dict[str, Any]:
    """
    Always returns a dict: {"summary": str, "citations": list[str]}
    - Never returns None
    - Validates citations against allowed set
    - Truncates summary to a COMPLETE sentence when possible
    """
    if not isinstance(out, dict):
        out = {}

    summary = (out.get("summary") or "").strip()
    cites = out.get("citations") or []

    # normalize citations to list
    if isinstance(cites, (str, int, float)):
        cites = [cites]
    if not isinstance(cites, list):
        cites = []

    cites_str = [str(x).strip() for x in cites if str(x).strip()]

    # keep only allowed + dedupe preserve order + cap 5
    seen = set()
    cites_valid = []
    for c in cites_str:
        if c in allowed and c not in seen:
            seen.add(c)
            cites_valid.append(c)
    cites_valid = cites_valid[:5]

    # truncate to complete sentence (preferred)
    summary = truncate_to_complete_sentence(summary, max_chars)

    return {"summary": summary, "citations": cites_valid}




def find_banned_terms(summary: str, banned_terms: List[str]) -> List[str]:
    s = (summary or "").lower()
    bad = []
    for t in banned_terms:
        if t.lower() in s:
            bad.append(t)
    return bad


# -----------------------------
# Main retrieval + rewrite
# -----------------------------
def retrieve_candidates(
    jd_text: str,
    db: str,
    index_path: str,
    meta_path: str,
    top_k: int,
    model: str,
    source_scope: str,
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

    chunks = []
    for fid, score in zip(ids, scores):
        if fid < 0:
            continue
        chunk_id = meta.get(int(fid))
        if chunk_id is None:
            continue
        text = fetch_chunk_text(db, chunk_id).strip()
        if not text:
            continue
        if is_low_info(text):
            continue
        if source_scope == "summary_only" and not is_summary_like(text):
            continue

        chunks.append(
            {
                "faiss_id": int(fid),
                "score": float(score),
                "chunk_id": chunk_id,
                "resume_key": extract_resume_key(chunk_id),
                "text": text,
            }
        )

    chunks = dedupe_chunks(chunks)
    # keep by score descending
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default="ops.db")
    ap.add_argument("--index_path", type=str, default="data/index/faiss_resume.index")
    ap.add_argument("--meta_path", type=str, default="data/index/faiss_meta.jsonl")
    ap.add_argument("--jd_file", type=str, default="")
    ap.add_argument("--jd_text", type=str, default="")

    # retrieval behavior
    ap.add_argument("--top_k", type=int, default=24, help="candidate pool from FAISS (before diversification)")
    ap.add_argument("--final_k", type=int, default=10, help="final chunks used for summary prompt")
    ap.add_argument("--source_scope", type=str, default="all", choices=["all", "summary_only"])

    # multi-resume controls
    ap.add_argument("--resume_mode", type=str, default="diverse", choices=["all", "diverse", "only"])
    ap.add_argument("--max_per_resume", type=int, default=4)
    ap.add_argument("--resume_keys", type=str, default="", help="comma-separated resume_key allowlist for resume_mode=only")

    # generation controls
    ap.add_argument("--max_chars", type=int, default=500)
    ap.add_argument("--embed_model", type=str, default="embedding-query")
    ap.add_argument("--solar_model", type=str, default="solar-pro")
    ap.add_argument("--confidence", type=str, default="balanced", choices=["strict", "balanced", "bold"])
    ap.add_argument("--max_tokens", type=int, default=600)
    ap.add_argument("--out_file", type=str, default="outputs/rewrite_summary.json")
    args = ap.parse_args()

    jd_text = args.jd_text
    if args.jd_file:
        p = Path(args.jd_file)
        if not p.exists():
            raise SystemExit(f"JD file not found: {p}. Create it or use --jd_text.")
        jd_text = p.read_text(encoding="utf-8")
    if not jd_text.strip():
        raise SystemExit("JD text is empty.")

    load_dotenv()
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise SystemExit("UPSTAGE_API_KEY is missing in env or .env")

    conf = CONF_SETTINGS.get(args.confidence, CONF_SETTINGS["balanced"])
    temperature = conf["temperature"]

    # candidate retrieval pool
    candidates = retrieve_candidates(
        jd_text=jd_text,
        db=args.db,
        index_path=args.index_path,
        meta_path=args.meta_path,
        top_k=args.top_k,
        model=args.embed_model,
        source_scope=args.source_scope,
    )
    if not candidates:
        raise SystemExit("No retrieved chunks after filtering. Try increasing --top_k or set --source_scope all.")

    resume_keys_allow = None
    if args.resume_keys.strip():
        resume_keys_allow = {x.strip() for x in args.resume_keys.split(",") if x.strip()}

    # diversify / select final prompt chunks
    chunks = diversify_chunks(
        chunks=candidates,
        mode=args.resume_mode,
        max_per_resume=max(1, args.max_per_resume),
        resume_keys_allow=resume_keys_allow,
        final_k=max(2, args.final_k),
    )
    if not chunks:
        raise SystemExit("No chunks selected. If resume_mode=only, check --resume_keys value.")

    allowed_chunk_ids = {str(c["chunk_id"]) for c in chunks}

    messages = build_summary_messages(
        jd_text=jd_text,
        chunks=chunks,
        max_chars=args.max_chars,
        confidence=args.confidence,
    )

    def request_once(extra_note: str = "") -> str:
        if not extra_note:
            return call_solar_chat(api_key, args.solar_model, messages, temperature, args.max_tokens)
        retry_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Fix your previous answer.\n"
                    "Requirements:\n"
                    "- Return VALID JSON only\n"
                    f"- summary <= {args.max_chars} characters\n"
                    "- citations must be 2 to 5 EXACT chunk_id strings from allowed list (no duplicates)\n"
                    "- Do not use banned words\n"
                    "- No extra text\n"
                    + extra_note
                ),
            }
        ]
        return call_solar_chat(api_key, args.solar_model, retry_messages, temperature, args.max_tokens)

    # Call Solar with retry if JSON invalid
    raw = request_once()
    try:
        out = safe_json_load(raw)
    except Exception:
        raw2 = request_once("\nReturn JSON only. Do not include code fences.")
        try:
            out = safe_json_load(raw2)
        except Exception:
            raise SystemExit("Solar did not return valid JSON after retry. Raw output:\n" + raw2)

    out_clean = validate_summary_output(out, allowed=allowed_chunk_ids, max_chars=args.max_chars)

    # Enforce banned terms by confidence level (auto-fix once)
    banned_terms = CONF_SETTINGS.get(args.confidence, CONF_SETTINGS["balanced"])["banned_terms"]
    bad = find_banned_terms(out_clean["summary"], banned_terms)
    if bad:
        raw_fix = request_once(
            "\nRemove these banned words from the summary: " + ", ".join(bad) + "\nRewrite conservatively while staying grounded."
        )
        try:
            out2 = safe_json_load(raw_fix)
            out_clean2 = validate_summary_output(out2, allowed=allowed_chunk_ids, max_chars=args.max_chars)
            # accept fix only if it removed banned terms and still has citations
            if not find_banned_terms(out_clean2["summary"], banned_terms) and out_clean2["citations"]:
                out_clean = out_clean2
        except Exception:
            pass

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional: show which resumes contributed (useful for debugging)
    used_resume_keys = sorted({
        extract_resume_key(cid)
        for cid in out_clean.get("citations", [])
        if isinstance(cid, str) and cid.strip()
    })
    out_clean["used_resumes"] = used_resume_keys


    out_path.write_text(json.dumps(out_clean, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Wrote:", out_path)
    print("\nSUMMARY:")
    print(out_clean["summary"])
    print("\nCITATIONS:", ", ".join(out_clean["citations"]) if out_clean["citations"] else "(none)")
    print("\nUSED_RESUMES:", ", ".join(used_resume_keys) if used_resume_keys else "(none)")



if __name__ == "__main__":
    main()
