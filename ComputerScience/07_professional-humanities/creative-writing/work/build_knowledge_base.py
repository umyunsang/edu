#!/usr/bin/env python3
"""Build a page-level text knowledge base from the course PDFs.

This script intentionally avoids external Python dependencies. It relies on
Poppler's `pdfinfo` and `pdftotext`, which are already available on this
machine.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "knowledge" / "source_text"
PAGES_DIR = OUT_DIR / "pages"
SOURCE_MAP = ROOT / "knowledge" / "source_map.md"
MANIFEST = OUT_DIR / "manifest.json"


@dataclass
class SourceRecord:
    source_id: str
    title: str
    relative_path: str
    pages: int
    combined_text: str
    page_dir: str
    priority: str


def run_command(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            args,
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Missing required command: {args[0]}. Install Poppler first."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Command failed: {' '.join(args)}\nSTDERR:\n{exc.stderr}"
        ) from exc
    return completed.stdout


def pdf_pages(path: Path) -> int:
    info = run_command(["pdfinfo", str(path)])
    match = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE)
    if not match:
        raise SystemExit(f"Could not detect page count for {path}")
    return int(match.group(1))


def slugify(stem: str, existing: set[str]) -> str:
    normalized = unicodedata.normalize("NFC", stem)
    slug = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", normalized).strip("_")
    if not slug:
        slug = "source"
    if slug not in existing:
        existing.add(slug)
        return slug
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
    unique = f"{slug}_{digest}"
    existing.add(unique)
    return unique


def source_priority(path: Path) -> str:
    name = unicodedata.normalize("NFC", path.name)
    if "강의자료" in name:
        return "primary"
    if "정리" in name:
        return "derived-summary"
    if "퀴즈" in name:
        return "assignment-example"
    return "source"


def extract_page(path: Path, page: int) -> str:
    return run_command(["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(path), "-"])


def find_pdfs() -> list[Path]:
    pdfs = [p for p in ROOT.rglob("*.pdf") if "knowledge" not in p.parts]
    return sorted(pdfs, key=lambda p: str(p.relative_to(ROOT)))


def build() -> list[SourceRecord]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    records: list[SourceRecord] = []
    used_slugs: set[str] = set()

    for pdf in find_pdfs():
        source_id = slugify(pdf.stem, used_slugs)
        pages = pdf_pages(pdf)
        page_dir = PAGES_DIR / source_id
        page_dir.mkdir(parents=True, exist_ok=True)
        combined_chunks: list[str] = []

        for page in range(1, pages + 1):
            text = extract_page(pdf, page).rstrip()
            page_text = (
                f"SOURCE_ID: {source_id}\n"
                f"TITLE: {pdf.stem}\n"
                f"PDF_PATH: {pdf.relative_to(ROOT)}\n"
                f"PAGE: {page}\n"
                "---\n"
                f"{text}\n"
            )
            page_file = page_dir / f"page-{page:03d}.txt"
            page_file.write_text(page_text, encoding="utf-8")
            combined_chunks.append(page_text)

        combined_path = OUT_DIR / f"{source_id}.txt"
        combined_path.write_text("\n\n".join(combined_chunks), encoding="utf-8")

        records.append(
            SourceRecord(
                source_id=source_id,
                title=pdf.stem,
                relative_path=str(pdf.relative_to(ROOT)),
                pages=pages,
                combined_text=str(combined_path.relative_to(ROOT)),
                page_dir=str(page_dir.relative_to(ROOT)),
                priority=source_priority(pdf),
            )
        )

    MANIFEST.write_text(
        json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    source_rows = [
        "# Source Map",
        "",
        "이 파일은 `tools/build_knowledge_base.py`로 생성한 PDF 근거 인덱스의 요약입니다.",
        "",
        "| 우선순위 | source_id | 원본 | 페이지 | 추출 텍스트 |",
        "|---|---|---|---:|---|",
    ]
    for record in records:
        source_rows.append(
            f"| {record.priority} | `{record.source_id}` | `{record.relative_path}` | "
            f"{record.pages} | `{record.combined_text}` |"
        )
    source_rows.extend(
        [
            "",
            "## 우선순위 규칙",
            "",
            "- `primary`: 원본 강의자료 PDF. 과제 근거로 가장 먼저 확인합니다.",
            "- `derived-summary`: 사용자가 정리한 파생 자료. 원본 PDF와 대조해 보조 근거로 사용합니다.",
            "- `assignment-example`: 기존 실습/퀴즈 예시. 형식 추론에 사용합니다.",
        ]
    )
    SOURCE_MAP.write_text("\n".join(source_rows) + "\n", encoding="utf-8")
    return records


def main() -> int:
    records = build()
    print(f"Built {len(records)} PDF sources into {OUT_DIR.relative_to(ROOT)}")
    for record in records:
        print(f"- {record.source_id}: {record.pages} pages ({record.priority})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
