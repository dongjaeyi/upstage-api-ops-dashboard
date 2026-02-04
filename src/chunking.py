from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, List

SECTION_KEYWORDS = {
    "experience", "work experience", "professional experience", "employment",
    "education",
    "skills", "technical skills", "core skills",
    "projects", "selected projects",
    "summary", "profile", "objective",
    "certifications", "certificates",
    "publications",
    "awards",
    "activities", "leadership",
}

WHITESPACE_RE = re.compile(r"\s+")
BULLET_START_RE = re.compile(r"^\s*(?:[-*•\u2022\u25E6])\s+(.*)$")
NUMBERED_START_RE = re.compile(r"^\s*(\d+[\.\)])\s+(.*)$")


def _normalize_line(s: str) -> str:
    s = s.replace("\u00a0", " ").strip()
    s = WHITESPACE_RE.sub(" ", s)
    return s


@dataclass
class Chunk:
    section: str
    chunk_index: int
    text: str
    source: str


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out = []

    def handle_starttag(self, tag, attrs):
        if tag in ("br", "p", "div", "h1", "h2", "h3", "tr"):
            self.out.append("\n")
        if tag in ("td", "li"):
            self.out.append(" ")

    def handle_data(self, data):
        if data:
            self.out.append(data)

    def get_text(self) -> str:
        return "".join(self.out)


def _html_to_text(html: str) -> str:
    p = _HTMLText()
    p.feed(html)
    return p.get_text()


def extract_lines_from_parse(parse_json: Any) -> List[str]:
    raw_lines: List[str] = []

    # 1) elements reading-order (best structured)
    if isinstance(parse_json, dict) and isinstance(parse_json.get("elements"), list):
        for el in parse_json["elements"]:
            if not isinstance(el, dict):
                continue
            cat = el.get("category", "")
            html = (el.get("content") or {}).get("html", "")
            if not isinstance(html, str) or not html.strip():
                continue

            text = _normalize_line(_html_to_text(html))
            if not text:
                continue

            if cat == "list" and not text.startswith("-"):
                text = "- " + text

            raw_lines.append(text)

    # 2) also add whole document html (helps recover missed text)
    if isinstance(parse_json, dict):
        doc_html = (parse_json.get("content") or {}).get("html", "")
        if isinstance(doc_html, str) and doc_html.strip():
            for ln in _html_to_text(doc_html).splitlines():
                ln = _normalize_line(ln)
                if ln:
                    raw_lines.append(ln)

    # 3) de-dup while preserving order
    lines: List[str] = []
    seen = set()
    for ln in raw_lines:
        if ln not in seen:
            lines.append(ln)
            seen.add(ln)

    return lines



def _looks_like_heading(line: str) -> bool:
    if not line:
        return False
    l = line.strip().strip(":")
    low = l.lower()
    if low in SECTION_KEYWORDS:
        return True
    if len(l) <= 40 and l.count(" ") <= 6 and not l.endswith("."):
        if l.isupper():
            return True
        words = l.split()
        if words and sum(w[:1].isupper() for w in words) >= max(1, len(words) - 1):
            return True
    return False


def chunk_resume_lines(lines: List[str]) -> List[Chunk]:
    chunks: List[Chunk] = []
    section = "unknown"
    current: List[str] = []
    idx = 0

    def flush():
        nonlocal idx, current
        if current:
            text = _normalize_line(" ".join(current))
            if text:
                chunks.append(Chunk(section=section, chunk_index=idx, text=text, source="upstage_parse"))
                idx += 1
        current = []

    for raw in (lines or []):
        line = _normalize_line(raw)
        if not line:
            continue

        if _looks_like_heading(line):
            flush()
            section = line.strip().strip(":").lower()
            continue

        m = BULLET_START_RE.match(line)
        if m:
            flush()
            current = [m.group(1).strip()]
            continue

        m2 = NUMBERED_START_RE.match(line)
        if m2:
            flush()
            current = [m2.group(2).strip()]
            continue

        if current:
            current.append(line)
        else:
            current = [line]
            flush()

    flush()
    return chunks


def chunk_from_parse_json(parse_json: Any) -> List[Chunk]:
    lines = extract_lines_from_parse(parse_json) or []
    return chunk_resume_lines(lines)
