#!/usr/bin/env python3
"""Query the local AIOSS SQLite FTS index with cited snippets."""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sanitize_query(query: str) -> str:
    terms = re.findall(r"[\w가-힣]+", query)
    return " OR ".join(terms[:12]) or query


def snippet(text: str, size: int = 360) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:size] + ("..." if len(compact) > size else "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    db_path = root / ".aioss-rag" / "index" / "fts.sqlite"
    if not db_path.exists():
        print(f"Missing index: {db_path}")
        print("Run: python3 tools/aioss_eval/build_rag_index.py --root .")
        return 2

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    fts_query = sanitize_query(args.query)
    rows = conn.execute(
        """
        SELECT path, source_type, title, week, chunk, text, bm25(docs) AS score
        FROM docs
        WHERE docs MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (fts_query, args.limit),
    ).fetchall()
    conn.close()

    if not rows:
        print("No results.")
        return 1

    for index, row in enumerate(rows, start=1):
        week = f"week {row['week']}, " if row["week"] else ""
        print(f"[{index}] {row['path']} ({week}chunk {row['chunk']}, score {row['score']:.4f})")
        print(f"    {snippet(row['text'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
