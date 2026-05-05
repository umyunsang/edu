#!/usr/bin/env python3
"""Phase 13: identify truly orphan notes (no inbound or outbound wikilinks).

Counts ALL link types: [[X]], ![[X]] embeds, and inline-field [[X]] (up::, central:: etc).
A note is orphan iff: in_count + out_count == 0.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import iter_vault_notes  # noqa: E402

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
EXCLUDE = (".obsidian", ".git", ".claude", ".pytest_cache", "scripts", "docs", "_templates")
LINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")


def build_index() -> dict[str, str]:
    """basename (no .md, lowercase) → relative path. Manual stem to avoid pathlib
    confusion on '1. xxx.md' style names."""
    idx: dict[str, str] = {}
    for p in iter_vault_notes(VAULT, exclude=EXCLUDE):
        name = p.name
        stem_str = name[:-3] if name.endswith(".md") else name
        idx.setdefault(stem_str.lower(), str(p.relative_to(VAULT)))
    return idx


def extract_links(text: str) -> set[str]:
    """Return unique LOWERCASE basenames referenced in the body."""
    out: set[str] = set()
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        for m in LINK_RE.finditer(line):
            target = m.group(1).strip()
            if not target:
                continue
            if target.startswith("http") or target.startswith(("'", '"', "$", "{")):
                continue
            base = target.rsplit("/", 1)[-1]
            stem = (base[:-3] if base.endswith(".md") else base).lower()
            out.add(stem)
    return out


def main():
    idx = build_index()
    notes = list(iter_vault_notes(VAULT, exclude=EXCLUDE))
    def note_key(p: Path) -> str:
        n = p.name
        return (n[:-3] if n.endswith(".md") else n).lower()

    out_links: dict[str, set[str]] = {}
    in_count: dict[str, int] = {note_key(p): 0 for p in notes}

    for p in notes:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        links = extract_links(text)
        out_links[note_key(p)] = links
        for link in links:
            if link in in_count:
                in_count[link] += 1

    orphans: list[Path] = []
    weak: list[tuple[Path, int, int]] = []
    for p in notes:
        key = note_key(p)
        out_n = len(out_links.get(key, set()))
        in_n = in_count.get(key, 0)
        if out_n == 0 and in_n == 0:
            orphans.append(p)
        elif (out_n + in_n) <= 1:
            weak.append((p, in_n, out_n))

    print(f"Total notes: {len(notes)}")
    print(f"Truly orphan (in=0, out=0): {len(orphans)}")
    print(f"Weakly connected (in+out <= 1): {len(weak)}")
    print()
    print("=== ORPHANS ===")
    for p in orphans:
        print(f"  {p.relative_to(VAULT)}")
    print()
    print("=== WEAK (sample 30) ===")
    for p, i, o in weak[:30]:
        print(f"  in={i} out={o}  {p.relative_to(VAULT)}")
    if len(weak) > 30:
        print(f"  ... and {len(weak)-30} more")

    summary = {"total": len(notes), "orphan": len(orphans), "weak": len(weak)}
    Path("/tmp/orphan-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    Path("/tmp/orphans.txt").write_text("\n".join(str(p.relative_to(VAULT)) for p in orphans), encoding="utf-8")


if __name__ == "__main__":
    main()
