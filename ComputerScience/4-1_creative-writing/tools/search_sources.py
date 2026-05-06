#!/usr/bin/env python3
"""Search course sources with local lexical retrieval.

The goal is not to replace semantic RAG. It gives Codex and the student a fast,
offline way to find page-level evidence before writing or revising assignments.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = ROOT / "knowledge" / "source_text" / "pages"
LOCAL_NOTES = [
    ROOT / "중간고사_창의적글쓰기_정리.md",
    ROOT / "pdf" / "퀴즈.md",
]


@dataclass
class Chunk:
    label: str
    path: str
    title: str
    priority: str
    page: int | None
    text: str


@dataclass
class Hit:
    score: float
    label: str
    path: str
    title: str
    priority: str
    page: int | None
    snippet: str


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[0-9A-Za-z가-힣]+", text)]


def parse_page_file(path: Path) -> Chunk:
    raw = path.read_text(encoding="utf-8", errors="replace")
    header, _, body = raw.partition("---\n")
    fields: dict[str, str] = {}
    for line in header.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    title = fields.get("TITLE", path.parent.name)
    page_raw = fields.get("PAGE")
    page = int(page_raw) if page_raw and page_raw.isdigit() else None
    source_id = fields.get("SOURCE_ID", path.parent.name)
    label = f"{source_id} p.{page}" if page is not None else source_id
    priority = "source"
    if "26-1창의적글쓰기_강의자료_중핵교양" in source_id:
        priority = "primary"
    elif "중간고사" in source_id and "정리" in source_id:
        priority = "derived-summary"
    elif "퀴즈" in source_id:
        priority = "assignment-example"
    return Chunk(
        label=label,
        path=str(path.relative_to(ROOT)),
        title=title,
        priority=priority,
        page=page,
        text=body.strip(),
    )


def split_markdown(path: Path, chunk_size: int = 1800, overlap: int = 250) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    rel = str(path.relative_to(ROOT))
    chunks: list[Chunk] = []
    start = 0
    index = 1
    while start < len(raw):
        end = min(start + chunk_size, len(raw))
        text = raw[start:end].strip()
        if text:
            chunks.append(
                Chunk(
                    label=f"{rel} chunk {index}",
                    path=rel,
                    title=path.stem,
                    priority="local-note",
                    page=None,
                    text=text,
                )
            )
            index += 1
        if end == len(raw):
            break
        start = max(0, end - overlap)
    return chunks


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    if PAGES_DIR.exists():
        for path in sorted(PAGES_DIR.glob("*/page-*.txt")):
            chunks.append(parse_page_file(path))
    for note in LOCAL_NOTES:
        if note.exists():
            chunks.extend(split_markdown(note))
    return chunks


def score_chunk(query: str, query_tokens: list[str], chunk: Chunk) -> float:
    haystack = f"{chunk.title}\n{chunk.text}".lower()
    score = 0.0
    query_lower = query.lower().strip()
    if query_lower and query_lower in haystack:
        score += 12.0 + haystack.count(query_lower) * 2.0
    matched_tokens = 0
    for token in query_tokens:
        if len(token) < 2:
            continue
        occurrences = haystack.count(token)
        if occurrences:
            matched_tokens += 1
            score += 1.0 + min(occurrences, 4) * 0.45
            if token in chunk.title.lower():
                score += 2.0
    meaningful_tokens = [token for token in query_tokens if len(token) >= 2]
    if meaningful_tokens and matched_tokens == len(set(meaningful_tokens)):
        score += 10.0
    elif matched_tokens >= 2:
        score += 3.0
    if chunk.page is not None:
        score += 0.2
    if chunk.priority == "primary":
        score += 1.5
    return score


def make_snippet(text: str, query_tokens: list[str], width: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""
    lower = compact.lower()
    positions = [lower.find(token) for token in query_tokens if len(token) >= 2 and lower.find(token) >= 0]
    pos = min(positions) if positions else 0
    start = max(0, pos - width // 2)
    end = min(len(compact), start + width)
    snippet = compact[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(compact):
        snippet = snippet + "..."
    return snippet


def search(query: str, limit: int, primary_only: bool = False, pdf_only: bool = False) -> list[Hit]:
    query_tokens = tokenize(query)
    hits: list[Hit] = []
    for chunk in load_chunks():
        if primary_only and chunk.priority != "primary":
            continue
        if pdf_only and chunk.page is None:
            continue
        score = score_chunk(query, query_tokens, chunk)
        if score <= 0:
            continue
        hits.append(
            Hit(
                score=round(score, 2),
                label=chunk.label,
                path=chunk.path,
                title=chunk.title,
                priority=chunk.priority,
                page=chunk.page,
                snippet=make_snippet(chunk.text, query_tokens),
            )
        )
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:limit]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Search course PDF and note sources.")
    parser.add_argument("query", help="검색어 또는 질문")
    parser.add_argument("-n", "--limit", type=int, default=8, help="표시할 결과 수")
    parser.add_argument("--primary-only", action="store_true", help="원본 강의자료 PDF만 검색")
    parser.add_argument("--pdf-only", action="store_true", help="PDF에서 추출한 페이지 텍스트만 검색")
    parser.add_argument("--json", action="store_true", help="JSON으로 출력")
    args = parser.parse_args(argv)

    hits = search(args.query, args.limit, primary_only=args.primary_only, pdf_only=args.pdf_only)
    if args.json:
        print(json.dumps([asdict(hit) for hit in hits], ensure_ascii=False, indent=2))
        return 0

    if not hits:
        print("No matches. Try broader Korean keywords from the lecture slides.")
        return 1

    for index, hit in enumerate(hits, start=1):
        page = f" p.{hit.page}" if hit.page is not None else ""
        print(f"{index}. score={hit.score} | {hit.priority} | {hit.title}{page} | {hit.path}")
        print(f"   {hit.snippet}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
