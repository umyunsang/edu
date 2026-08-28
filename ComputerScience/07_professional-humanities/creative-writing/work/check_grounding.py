#!/usr/bin/env python3
"""Lightweight citation/grounding check for assignment drafts."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


CITATION_RE = re.compile(r"\[[^\]]*(?:p\.|쪽|페이지)\s*\d+[^\]]*\]")


def paragraphs(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check whether draft paragraphs cite source pages.")
    parser.add_argument("draft", type=Path, help="검사할 Markdown 초안")
    args = parser.parse_args(argv)

    text = args.draft.read_text(encoding="utf-8", errors="replace")
    blocks = paragraphs(text)
    citation_count = len(CITATION_RE.findall(text))
    risky: list[tuple[int, str]] = []

    for index, block in enumerate(blocks, start=1):
        plain = re.sub(r"^#+\s*", "", block)
        if len(plain) < 80:
            continue
        if not CITATION_RE.search(block):
            risky.append((index, re.sub(r"\s+", " ", plain)[:120]))

    print(f"Citations found: {citation_count}")
    if not risky:
        print("No long uncited paragraphs found.")
        return 0

    print("Long paragraphs without page citation:")
    for index, preview in risky:
        print(f"- paragraph {index}: {preview}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
