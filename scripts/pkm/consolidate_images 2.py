#!/usr/bin/env python3
"""Phase 6: consolidate scattered images into /image/ with course prefix."""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import iter_vault_notes  # noqa: E402

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
IMAGE_DIR = VAULT / "image"
EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
EXCLUDE_PARTS = {".obsidian", ".git", ".claude", ".pytest_cache", "scripts", "docs", "_templates"}


def find_scattered_images() -> list[Path]:
    out: list[Path] = []
    for p in VAULT.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            continue
        if any(part in EXCLUDE_PARTS for part in p.relative_to(VAULT).parts):
            continue
        try:
            if p.parent.resolve() == IMAGE_DIR.resolve():
                continue
            if (IMAGE_DIR / "_archive").resolve() in p.resolve().parents:
                continue
        except Exception:
            pass
        out.append(p)
    return sorted(out)


def derive_new_name(p: Path) -> str:
    rel = p.relative_to(VAULT)
    parts = rel.parts
    course = "misc"
    if parts[0] == "ComputerScience" and len(parts) >= 2:
        course = parts[1]
    return f"{course}__{p.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    IMAGE_DIR.mkdir(exist_ok=True)
    moves: list[tuple[Path, Path, str, str]] = []
    for p in find_scattered_images():
        new_name = derive_new_name(p)
        new_path = IMAGE_DIR / new_name
        i = 1
        while new_path.exists() and new_path.resolve() != p.resolve():
            stem = Path(new_name).stem
            ext = Path(new_name).suffix
            new_path = IMAGE_DIR / f"{stem}_{i}{ext}"
            i += 1
        moves.append((p, new_path, p.name, new_path.name))

    print(f"Found {len(moves)} scattered images.")
    for old, new, _, _ in moves[:10]:
        print(f"  {old.relative_to(VAULT)} -> image/{new.name}")
    if len(moves) > 10:
        print(f"  ... and {len(moves) - 10} more")

    if args.dry_run:
        return

    for old, new, _, _ in moves:
        if old.resolve() != new.resolve():
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(new))

    name_map = {old_name: new_name for _, _, old_name, new_name in moves}
    pattern_wiki = re.compile(
        r"!\[\[([^\]|#]+?\.(?:png|jpg|jpeg|gif|webp|svg))(\|[^\]]*)?\]\]",
        re.IGNORECASE,
    )
    pattern_md = re.compile(
        r"!\[([^\]]*)\]\(([^)]+?\.(?:png|jpg|jpeg|gif|webp|svg))\)",
        re.IGNORECASE,
    )

    updated_notes = 0
    for note in iter_vault_notes(VAULT, exclude=EXCLUDE_PARTS):
        text = note.read_text(encoding="utf-8")
        original = text

        def fix_wiki(m):
            full = m.group(1)
            base = Path(full).name
            if base in name_map:
                return f"![[{name_map[base]}{m.group(2) or ''}]]"
            return m.group(0)

        text = pattern_wiki.sub(fix_wiki, text)

        def fix_md(m):
            base = Path(m.group(2)).name
            if base in name_map:
                return f"![{m.group(1)}](image/{name_map[base]})"
            return m.group(0)

        text = pattern_md.sub(fix_md, text)

        if text != original:
            note.write_text(text, encoding="utf-8")
            updated_notes += 1

    for d in sorted(VAULT.rglob("images"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            try:
                d.rmdir()
            except OSError:
                pass

    print(f"\nMoved {len(moves)} images, updated wikilinks in {updated_notes} notes.")


if __name__ == "__main__":
    main()
