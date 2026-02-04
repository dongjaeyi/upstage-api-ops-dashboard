# src/pipeline.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import os
import uuid
from src.logger import init_db
init_db()




def run_module(module: str, args: list[str]) -> None:
    """
    Run: python -m <module> <args...>
    """
    cmd = [sys.executable, "-m", module] + args
    print("\n[PIPELINE] " + " ".join(cmd))
    subprocess.run(cmd, check=True)




def cmd_ingest(ns: argparse.Namespace) -> None:
    args = [
        "--pdf_dir", ns.pdf_dir,
        "--db", ns.db,
        "--out_dir", ns.out_dir,
    ]
    if ns.force:
        args.append("--force")
    run_module("src.resume_ingest", args)


def cmd_index(ns: argparse.Namespace) -> None:
    args = [
        "--db", ns.db,
        "--out_dir", ns.out_dir,
        "--batch", str(ns.batch),
    ]
    if ns.force:
        args.append("--force")
    run_module("src.embed_index", args)


def cmd_retrieve(ns: argparse.Namespace) -> None:
    args: list[str] = [
        "--db", ns.db,
        "--index_path", ns.index_path,
        "--meta_path", ns.meta_path,
        "--top_k", str(ns.top_k),
    ]
    if ns.jd_file:
        args += ["--jd_file", ns.jd_file]
    if ns.jd_text:
        args += ["--jd_text", ns.jd_text]
    run_module("src.retrieve", args)


def cmd_summary(ns: argparse.Namespace) -> None:
    args: list[str] = [
        "--db", ns.db,
        "--index_path", ns.index_path,
        "--meta_path", ns.meta_path,
        "--top_k", str(ns.top_k),
        "--final_k", str(ns.final_k),
        "--resume_mode", ns.resume_mode,
        "--max_per_resume", str(ns.max_per_resume),
        "--max_chars", str(ns.max_chars),
        "--confidence", ns.confidence,
        "--out_file", ns.out_file,
    ]
    if getattr(ns, "source_scope", None):
        args += ["--source_scope", ns.source_scope]

    if ns.jd_file:
        args += ["--jd_file", ns.jd_file]
    if ns.jd_text:
        args += ["--jd_text", ns.jd_text]

    run_module("src.rewrite_summary", args)


def cmd_resume(ns: argparse.Namespace) -> None:
    # rewrite_resume 쪽 옵션이 네 파일에서 정확히 어떤지에 따라 맞춰야 함
    # 현재는 일반적인 passthrough 형태로 구성 (네 rewrite_resume.py 옵션에 맞게 조정 가능)
    args: list[str] = [
        "--db", ns.db,
        "--index_path", ns.index_path,
        "--meta_path", ns.meta_path,
        "--top_k", str(ns.top_k),
        "--n_bullets", str(ns.n_bullets),
        "--out_file", ns.out_file,
    ]
    if ns.jd_file:
        args += ["--jd_file", ns.jd_file]
    if ns.jd_text:
        args += ["--jd_text", ns.jd_text]

    run_module("src.rewrite_resume", args)


def cmd_all(ns: argparse.Namespace) -> None:
    """
    One-shot MVP (configurable):
      - ingest (optional) -> index (optional) -> rewrite_summary (always)
    """
    # Resolve index paths (used by summary)
    index_out_dir = Path(ns.index_out_dir)
    index_path = index_out_dir / "faiss_resume.index"
    meta_path = index_out_dir / "faiss_meta.jsonl"

    # 1) ingest (optional)
    if not ns.skip_ingest:
        if not ns.pdf_dir:
            raise SystemExit("[PIPELINE] --pdf_dir is required unless --skip_ingest is set.")
        ingest_ns = argparse.Namespace(
            pdf_dir=ns.pdf_dir,
            db=ns.db,
            out_dir=ns.parse_out_dir,
            force=ns.force_ingest,
        )
        cmd_ingest(ingest_ns)
    else:
        print("[PIPELINE] skip_ingest=True (using existing DB/chunks)")

    # 2) index (optional)
    if not ns.skip_index:
        index_ns = argparse.Namespace(
            db=ns.db,
            out_dir=str(index_out_dir),
            batch=ns.batch,
            force=ns.force_index,
        )
        cmd_index(index_ns)
    else:
        print("[PIPELINE] skip_index=True (using existing FAISS files)")
        if not index_path.exists() or not meta_path.exists():
            raise SystemExit(
                "[PIPELINE] skip_index=True but index files not found.\n"
                f"  Missing: {index_path if not index_path.exists() else ''}\n"
                f"  Missing: {meta_path if not meta_path.exists() else ''}\n"
                "  Fix: run `python -m src.pipeline index --force` or unset --skip_index."
            )

    # 3) summary (always)
    summary_ns = argparse.Namespace(
        db=ns.db,
        index_path=str(index_path),
        meta_path=str(meta_path),
        jd_file=ns.jd_file,
        jd_text=ns.jd_text,
        top_k=ns.top_k,
        final_k=ns.final_k,
        resume_mode=ns.resume_mode,
        max_per_resume=ns.max_per_resume,
        max_chars=ns.max_chars,
        confidence=ns.confidence,
        out_file=ns.out_file,
        source_scope=ns.source_scope,
    )
    cmd_summary(summary_ns)



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ingest
    sp = sub.add_parser("ingest", help="Parse PDFs -> chunk -> save to SQLite")
    sp.add_argument("--pdf_dir", required=True)
    sp.add_argument("--db", default="ops.db")
    sp.add_argument("--out_dir", default="data/parsed/resumes")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_ingest)

    # index
    sp = sub.add_parser("index", help="Embed all chunks -> build FAISS index")
    sp.add_argument("--db", default="ops.db")
    sp.add_argument("--out_dir", default="data/index")
    sp.add_argument("--batch", type=int, default=64)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_index)

    # retrieve
    sp = sub.add_parser("retrieve", help="Retrieve top-k chunks for a JD")
    sp.add_argument("--db", default="ops.db")
    sp.add_argument("--index_path", default="data/index/faiss_resume.index")
    sp.add_argument("--meta_path", default="data/index/faiss_meta.jsonl")
    sp.add_argument("--jd_file", default="")
    sp.add_argument("--jd_text", default="")
    sp.add_argument("--top_k", type=int, default=10)
    sp.set_defaults(func=cmd_retrieve)

    # summary (MVP)
    sp = sub.add_parser("summary", help="Rewrite a 500-char summary with citations (MVP)")
    sp.add_argument("--db", default="ops.db")
    sp.add_argument("--index_path", default="data/index/faiss_resume.index")
    sp.add_argument("--meta_path", default="data/index/faiss_meta.jsonl")
    sp.add_argument("--jd_file", default="")
    sp.add_argument("--jd_text", default="")
    sp.add_argument("--top_k", type=int, default=80)
    sp.add_argument("--final_k", type=int, default=18)
    sp.add_argument("--resume_mode", default="diverse", choices=["all", "diverse", "only"])
    sp.add_argument("--max_per_resume", type=int, default=3)
    sp.add_argument("--max_chars", type=int, default=500)
    sp.add_argument("--confidence", default="balanced", choices=["strict", "balanced", "bold"])
    sp.add_argument("--out_file", default="outputs/rewrite_summary.json")
    sp.add_argument("--source_scope", default="all", choices=["all", "summary_only"])
    sp.set_defaults(func=cmd_summary)

    # resume (full bullets)
    sp = sub.add_parser("resume", help="Rewrite resume bullets with citations (heavier)")
    sp.add_argument("--db", default="ops.db")
    sp.add_argument("--index_path", default="data/index/faiss_resume.index")
    sp.add_argument("--meta_path", default="data/index/faiss_meta.jsonl")
    sp.add_argument("--jd_file", default="")
    sp.add_argument("--jd_text", default="")
    sp.add_argument("--top_k", type=int, default=10)
    sp.add_argument("--n_bullets", type=int, default=8)
    sp.add_argument("--out_file", default="outputs/rewrite_bullets.json")
    sp.set_defaults(func=cmd_resume)

    # all
    # all
    sp = sub.add_parser("all", help="One-shot: (optional) ingest -> (optional) index -> summary")
    sp.add_argument("--pdf_dir", default="", help="Required unless --skip_ingest is set")
    sp.add_argument("--db", default="ops.db")

    sp.add_argument("--parse_out_dir", default="data/parsed/resumes")
    sp.add_argument("--index_out_dir", default="data/index")

    sp.add_argument("--skip_ingest", action="store_true", help="Skip parsing/ingest step")
    sp.add_argument("--skip_index", action="store_true", help="Skip embedding/index step (requires existing index files)")

    sp.add_argument("--force_ingest", action="store_true")
    sp.add_argument("--force_index", action="store_true")

    sp.add_argument("--batch", type=int, default=64)

    sp.add_argument("--jd_file", default="")
    sp.add_argument("--jd_text", default="")

    sp.add_argument("--top_k", type=int, default=80)
    sp.add_argument("--final_k", type=int, default=18)
    sp.add_argument("--resume_mode", default="diverse", choices=["all", "diverse", "only"])
    sp.add_argument("--max_per_resume", type=int, default=3)
    sp.add_argument("--max_chars", type=int, default=500)
    sp.add_argument("--confidence", default="balanced", choices=["strict", "balanced", "bold"])
    sp.add_argument("--out_file", default="outputs/rewrite_summary.json")
    sp.add_argument("--source_scope", default="all", choices=["all", "summary_only"])
    sp.set_defaults(func=cmd_all)


    return p


def main() -> None:
    parser = build_parser()
    ns = parser.parse_args()

    # basic validation
    if getattr(ns, "jd_file", "") and getattr(ns, "jd_text", ""):
        print("[WARN] Both --jd_file and --jd_text provided. --jd_text will be ignored if modules prioritize file.")

    ns.func(ns)


if __name__ == "__main__":
    main()
