#!/usr/bin/env python3
"""Phase 5: inject domain tags from folder→tag map."""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import iter_vault_notes, read_note, write_note  # noqa: E402

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
MAP_PATH = Path(__file__).resolve().parent / "folder_tag_map.json"


def load_map():
    return json.loads(MAP_PATH.read_text(encoding="utf-8"))


def tags_for_path(rel: Path, m: dict) -> list[str]:
    tags: set[str] = set()
    rel_str = str(rel).replace("\\", "/")

    for rule in m.get("filename_rules", []):
        if fnmatch.fnmatch(rel.name, rule["match"]):
            tags.update(rule.get("tags", []))

    for rule in m.get("subdir_rules", []):
        if fnmatch.fnmatch(rel_str, rule["match"]):
            tags.update(rule.get("tags", []))

    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "ComputerScience":
        folder = parts[1]
        for rule in m.get("patterns", []):
            if fnmatch.fnmatch(folder, rule["match"]):
                tags.update(rule.get("tags", []))
                break
    elif parts and parts[0] in {"certifications", "LGAimer"}:
        for rule in m.get("patterns", []):
            if fnmatch.fnmatch(rel_str, rule["match"]):
                tags.update(rule.get("tags", []))
                break

    return sorted(tags)


def process(path: Path, m: dict, dry_run: bool) -> tuple[int, str]:
    rel = path.relative_to(VAULT)
    new_tags = tags_for_path(rel, m)
    if not new_tags:
        return 0, ""
    fm, body = read_note(path)
    existing = set(fm.get("tags") or [])
    union = sorted(existing | set(new_tags))
    if union == sorted(existing):
        return 0, ""
    fm["tags"] = union
    if not dry_run:
        write_note(path, fm, body)
    added = sorted(set(new_tags) - existing)
    return len(added), f"{rel} +{added}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    m = load_map()
    total_added = total_files = 0
    for p in iter_vault_notes(VAULT, exclude=(".obsidian", ".git", ".claude", ".pytest_cache", "scripts", "docs", "_templates", "MOCs")):
        added, msg = process(p, m, args.dry_run)
        if added:
            total_added += added
            total_files += 1
            print(msg)
        if args.limit and total_files >= args.limit:
            break

    print(f"\nSummary: files={total_files}, tags_added={total_added}, dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
