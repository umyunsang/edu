#!/usr/bin/env python3
"""Build a local SQLite FTS index for AIOSS PDFs, markdown notes, and samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run_text(cmd: list[str], cwd: Path, timeout: int = 120) -> str:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_week(path: Path) -> str:
    match = re.search(r"week(\d+)", path.name, re.IGNORECASE)
    return match.group(1) if match else ""


def chunk_text(text: str, max_chars: int = 2600, overlap: int = 250) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


def pdf_records(path: Path, root: Path) -> list[dict[str, str]]:
    text = run_text(["pdftotext", "-layout", str(path), "-"], cwd=root) if shutil.which("pdftotext") else ""
    chunks = chunk_text(text)
    if not chunks:
        chunks = [f"[needs_ocr] {path.name} could not be extracted with pdftotext."]
    records = []
    for index, chunk in enumerate(chunks, start=1):
        records.append(
            {
                "source_type": "pdf",
                "path": str(path.relative_to(root)),
                "title": path.stem,
                "week": infer_week(path),
                "chunk": str(index),
                "text": chunk,
                "metadata": json.dumps({"needs_ocr": chunk.startswith("[needs_ocr]")}, ensure_ascii=False),
            }
        )
    return records


def text_records(path: Path, root: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    records = []
    for index, chunk in enumerate(chunk_text(text), start=1):
        records.append(
            {
                "source_type": path.suffix.lstrip(".") or "text",
                "path": str(path.relative_to(root)),
                "title": path.stem,
                "week": infer_week(path),
                "chunk": str(index),
                "text": chunk,
                "metadata": json.dumps({}, ensure_ascii=False),
            }
        )
    return records


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS docs")
    conn.execute(
        """
        CREATE VIRTUAL TABLE docs USING fts5(
            source_type,
            path,
            title,
            week,
            chunk,
            text,
            hash UNINDEXED,
            metadata UNINDEXED
        )
        """
    )
    return conn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="AIOSS folder root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    index_dir = root / ".aioss-rag" / "index"
    conn = connect(index_dir / "fts.sqlite")
    records: list[dict[str, str]] = []

    for pdf in sorted(root.glob("week*.pdf")):
        records.extend(pdf_records(pdf, root))
    for path in sorted((root / "md").glob("*.md")):
        records.extend(text_records(path, root))
    for path in sorted((root / "sample").rglob("*")):
        if path.is_file() and path.suffix in {".md", ".py", ".yml", ".yaml"}:
            records.extend(text_records(path, root))

    for record in records:
        source_path = root / record["path"]
        conn.execute(
            "INSERT INTO docs(source_type, path, title, week, chunk, text, hash, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record["source_type"],
                record["path"],
                record["title"],
                record["week"],
                record["chunk"],
                record["text"],
                sha256(source_path) if source_path.exists() else "",
                record["metadata"],
            ),
        )
    conn.commit()
    conn.close()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": ".",
        "database": str((index_dir / "fts.sqlite").relative_to(root)),
        "records": len(records),
        "method": "SQLite FTS5 lexical index with PDF layout extraction and overlap chunks",
        "upgrade_path": [
            "Docling layout parsing",
            "hybrid dense plus sparse retrieval",
            "cross-encoder or ColBERT-style reranking",
            "RAGAS or TruLens answer-level evaluation",
        ],
    }
    manifest_path = root / ".aioss-rag" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Indexed {len(records)} chunks")
    print(f"database: {index_dir / 'fts.sqlite'}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
