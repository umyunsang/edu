#!/usr/bin/env python3
"""Phase 7: rename notes with name collisions (excluding README.md → handled in Phase 8)."""
from __future__ import annotations

import argparse
import collections
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import iter_vault_notes  # noqa: E402

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
PRESERVE = {"README.md"}
EXCLUDE = (".obsidian", ".git", ".claude", ".pytest_cache", "scripts", "docs", "_templates", "MOCs")


def find_collisions() -> dict[str, list[Path]]:
    by_name: dict[str, list[Path]] = collections.defaultdict(list)
    for p in iter_vault_notes(VAULT, exclude=EXCLUDE):
        by_name[p.name].append(p)
    return {name: paths for name, paths in by_name.items() if len(paths) > 1 and name not in PRESERVE}


def derive_new_name(p: Path) -> str:
    rel = p.relative_to(VAULT)
    parts = rel.parts
    course = ""
    if parts[0] == "ComputerScience" and len(parts) >= 2:
        course = parts[1]
        course = re.sub(r"^\d-\d_", "", course)
        course = re.sub(r"^elective_", "", course)
    if not course:
        course = parts[0].lower()
    # Sub-folder context if collision still possible
    if len(parts) >= 4:
        sub = parts[-2]
        return f"{course}__{sub}__{p.name}"
    return f"{course}__{p.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    collisions = find_collisions()
    print(f"Collision groups: {len(collisions)}")
    rename_plan: list[tuple[Path, Path]] = []
    for name, paths in collisions.items():
        for p in paths:
            new_name = derive_new_name(p)
            new_path = p.parent / new_name
            i = 1
            while new_path.exists() and new_path.resolve() != p.resolve():
                stem = new_path.stem
                ext = new_path.suffix
                new_path = p.parent / f"{stem}_{i}{ext}"
                i += 1
            rename_plan.append((p, new_path))
            print(f"  {p.relative_to(VAULT)} -> {new_path.relative_to(VAULT)}")

    if args.dry_run:
        return

    name_map: dict[str, str] = {}
    for old, new in rename_plan:
        if old.resolve() != new.resolve():
            shutil.move(str(old), str(new))
            name_map[old.stem] = new.stem

    pattern = re.compile(r"\[\[([^\]|#]+?)(\|[^\]]*)?\]\]")
    updated = 0
    for note in iter_vault_notes(VAULT, exclude=EXCLUDE):
        text = note.read_text(encoding="utf-8")
        original = text

        def fix(m):
            target = m.group(1).strip()
            if target.startswith("!"):
                return m.group(0)
            base = Path(target).name
            stem = Path(base).stem if base.endswith(".md") else base
            if stem in name_map:
                new_target = target.replace(stem, name_map[stem])
                return f"[[{new_target}{m.group(2) or ''}]]"
            return m.group(0)

        text = pattern.sub(fix, text)
        if text != original:
            note.write_text(text, encoding="utf-8")
            updated += 1

    print(f"\nRenamed {len(rename_plan)} files, updated wikilinks in {updated} notes.")


if __name__ == "__main__":
    main()
