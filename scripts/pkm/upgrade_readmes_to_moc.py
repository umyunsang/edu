#!/usr/bin/env python3
"""Phase 8: upgrade course README.md files to MOC type with Dataview block."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import read_note, write_note  # noqa: E402

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
EXCLUDE = (".obsidian", ".git", ".claude", ".pytest_cache", "scripts", "docs", "_templates")

DATAVIEW_BLOCK = '''
## All notes in this course (auto)
```dataview
TABLE status, file.mtime as updated
FROM "{folder}"
WHERE type != "MOC"
SORT file.mtime DESC
LIMIT 50
```
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    readmes = []
    for p in VAULT.rglob("README.md"):
        if any(part in EXCLUDE for part in p.relative_to(VAULT).parts):
            continue
        readmes.append(p)

    print(f"Found {len(readmes)} README files.")
    n = 0
    for p in readmes:
        rel = p.relative_to(VAULT)
        folder = str(rel.parent).replace("\\", "/")
        if folder == "." or folder == "":
            # Root README — keep as portfolio entry, not MOC
            continue
        fm, body = read_note(p)
        fm["type"] = "MOC"
        fm["status"] = "evergreen"
        existing_tags = set(fm.get("tags") or [])
        existing_tags.add("type/MOC")
        existing_tags.discard("type/lecture")
        fm["tags"] = sorted(existing_tags)

        if "```dataview" not in body:
            body = body.rstrip() + "\n" + DATAVIEW_BLOCK.format(folder=folder)

        if args.dry_run:
            print(f"DRY: {rel} -> type=MOC, +type/MOC, +dataview")
        else:
            write_note(p, fm, body)
            print(f"OK:  {rel}")
        n += 1

    print(f"\nUpgraded {n} READMEs.")


if __name__ == "__main__":
    main()
