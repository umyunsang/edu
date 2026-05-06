#!/usr/bin/env python3
"""Phase 12: inject `up:: [[Domain MOC]]` inline field into every non-MOC note.

Resolution order:
  1. If folder→tag map yields a known cs/* / math/* / skill/* tag, use the matching domain MOC.
  2. If filename contains 'MOC' or path is in MOCs/, skip (MOC notes already have up::).
  3. Default fallback: Home MOC.

The inline field is inserted as the first non-blank line of the body, AFTER frontmatter.
If `up::` already exists, only update its value if it currently points to a different MOC
(idempotent — re-running the script doesn't duplicate).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import iter_vault_notes, read_note, write_note  # noqa: E402

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
MAP_PATH = Path(__file__).resolve().parent / "folder_tag_map.json"
EXCLUDE = (".obsidian", ".git", ".claude", ".pytest_cache", "scripts", "docs", "_templates")

# Tag → MOC name resolver (priority order — first match wins)
TAG_TO_MOC = [
    ("cs/dl",         "Deep Learning MOC"),
    ("cs/cv",         "Computer Vision MOC"),
    ("cs/llm",        "LLM & NLP MOC"),
    ("cs/nlp",        "LLM & NLP MOC"),
    ("cs/ml",         "Machine Learning MOC"),
    ("cs/algorithms", "Algorithms MOC"),
    ("skill/docker",  "Cloud & Containers MOC"),
    ("cs/devops",     "Cloud & Containers MOC"),
    ("cs/distributed","Systems MOC"),
    ("cs/systems",    "Systems MOC"),
    ("cs/db",         "Database MOC"),
    ("cs/security",   "Security MOC"),
    ("cs/open-source","AI Open Source MOC"),
    ("cs/ai",         "AI Open Source MOC"),
    ("cs/se",         "Software Engineering MOC"),
    ("math/linalg",   "Math Foundations MOC"),
    ("math/calculus", "Math Foundations MOC"),
    ("math/probability","Math Foundations MOC"),
    ("math/statistics","Math Foundations MOC"),
    ("math/discrete", "Math Foundations MOC"),
    ("meta/cert",     "Certifications MOC"),
    ("meta/portfolio","Portfolio MOC"),
    ("meta/question", "Open Questions MOC"),
]


def load_map():
    return json.loads(MAP_PATH.read_text(encoding="utf-8"))


def folder_tags(rel: Path, m: dict) -> list[str]:
    parts = rel.parts
    tags: list[str] = []
    if len(parts) >= 2 and parts[0] == "ComputerScience":
        folder = parts[1]
        for rule in m.get("patterns", []):
            if fnmatch.fnmatch(folder, rule["match"]):
                tags = list(rule.get("tags", []))
                break
    elif parts and parts[0] in {"certifications", "LGAimer"}:
        rel_str = str(rel).replace("\\", "/")
        for rule in m.get("patterns", []):
            if fnmatch.fnmatch(rel_str, rule["match"]):
                tags = list(rule.get("tags", []))
                break
    return tags


def resolve_moc(rel: Path, fm: dict, fmap: dict) -> str:
    # Fast path: existing tags
    note_tags = set(fm.get("tags") or [])
    for tag, moc in TAG_TO_MOC:
        if tag in note_tags:
            return moc
    # Fallback: folder→tag mapping
    for tag in folder_tags(rel, fmap):
        for ttag, moc in TAG_TO_MOC:
            if tag == ttag:
                return moc
    return "Home MOC"


UP_LINE_RE = re.compile(r"^up::\s*(\[\[[^\]]+\]\])\s*$", re.MULTILINE)


def inject_up(body: str, moc_name: str) -> tuple[str, bool]:
    """Insert or update `up:: [[<moc_name>]]` as the first inline-field block of body.
    Returns (new_body, changed)."""
    target = f"up:: [[{moc_name}]]"
    m = UP_LINE_RE.search(body)
    if m:
        if m.group(0).strip() == target:
            return body, False
        new_body = UP_LINE_RE.sub(target, body, count=1)
        return new_body, True
    # Insert at top, before first heading or content
    lines = body.split("\n")
    insert_at = 0
    for i, l in enumerate(lines):
        if l.strip() == "":
            insert_at = i + 1
            continue
        # If first non-blank starts with `central::`, `children::`, etc., insert before it
        if re.match(r"^(central|children|parents|siblings|friends)::", l):
            insert_at = i
            break
        insert_at = i
        break
    lines.insert(insert_at, target)
    if insert_at > 0 and lines[insert_at - 1].strip() != "":
        lines.insert(insert_at, "")
    if insert_at + 1 < len(lines) and lines[insert_at + 1].strip() != "":
        lines.insert(insert_at + 1, "")
    return "\n".join(lines), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    fmap = load_map()
    n_modified = n_skipped_moc = n_skipped_already = 0
    moc_distribution: dict[str, int] = {}

    for p in iter_vault_notes(VAULT, exclude=EXCLUDE):
        rel = p.relative_to(VAULT)
        try:
            fm, body = read_note(p)
        except Exception:
            continue
        # Skip MOCs themselves
        if fm.get("type") == "MOC":
            n_skipped_moc += 1
            continue
        if rel.parts[0] == "MOCs":
            n_skipped_moc += 1
            continue

        moc = resolve_moc(rel, fm, fmap)
        new_body, changed = inject_up(body, moc)
        moc_distribution[moc] = moc_distribution.get(moc, 0) + 1

        if not changed:
            n_skipped_already += 1
            continue

        if args.dry_run:
            print(f"DRY: {rel}  →  up:: [[{moc}]]")
        else:
            write_note(p, fm, new_body)
        n_modified += 1
        if args.limit and n_modified >= args.limit:
            break

    print(f"\nSummary: modified={n_modified}, skipped_moc={n_skipped_moc}, skipped_unchanged={n_skipped_already}")
    print(f"\nMOC distribution:")
    for moc, n in sorted(moc_distribution.items(), key=lambda x: -x[1]):
        print(f"  {n:4}  {moc}")


if __name__ == "__main__":
    main()
