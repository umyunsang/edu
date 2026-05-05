#!/usr/bin/env python3
"""Phase 17: revert README files from `type: MOC` to `type: index`.

README files are folder entry documentation, not conceptual parents.
- Reset type from MOC → index
- Remove `type/MOC` tag
- Keep up:: link to domain MOC
- Keep Dataview block as informational
- Remove auto-injected `## Linked Notes (auto)` listing (since README isn't a true parent)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import iter_vault_notes, read_note, write_note  # noqa: E402

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
EXCLUDE = (".obsidian", ".git", ".claude", ".pytest_cache", "scripts", "docs", "_templates")
LINKED_NOTES_RE = re.compile(
    r"^##\s+Linked Notes \(auto\)\s*\n.*?(?=\n##\s|\Z)",
    re.DOTALL | re.MULTILINE,
)


def main():
    n_reverted = 0
    moc_dir = VAULT / "MOCs"
    domain_mocs = sorted(moc_dir.glob("*.md")) if moc_dir.exists() else []

    # Step 1: Revert README files
    for p in iter_vault_notes(VAULT, exclude=EXCLUDE):
        if p.name != "README.md":
            continue
        # Skip MOCs/ folder (those are real MOCs)
        if p.parent.name == "MOCs":
            continue

        fm, body = read_note(p)
        if fm.get("type") != "MOC":
            continue  # already not a MOC

        fm["type"] = "index"
        fm["status"] = "evergreen"  # README usually stable
        tags = set(fm.get("tags") or [])
        tags.discard("type/MOC")
        tags.add("type/index")
        fm["tags"] = sorted(tags)
        write_note(p, fm, body)
        print(f"reverted: {p.relative_to(VAULT)}")
        n_reverted += 1

    # Step 2: Strip README references from "## Linked Notes (auto)" sections in domain MOCs
    n_mocs_cleaned = 0
    for moc in domain_mocs:
        fm, body = read_note(moc)
        m = LINKED_NOTES_RE.search(body)
        if not m:
            continue
        section = m.group(0)
        # Filter out lines like "- [[README]]" — there shouldn't be any since README stem is "README"
        # But also filter "- [[<course>__README]]" patterns from collisions
        new_lines = []
        for line in section.splitlines():
            if line.strip().startswith("- [[") and line.rstrip().endswith("]]"):
                inner = line.strip()[4:-2]
                if inner == "README" or inner.endswith("__README"):
                    continue
            new_lines.append(line)
        new_section = "\n".join(new_lines)
        if new_section != section:
            new_body = body.replace(section, new_section)
            write_note(moc, fm, new_body)
            n_mocs_cleaned += 1
            print(f"cleaned MOC: {moc.relative_to(VAULT)}")

    print(f"\nReverted {n_reverted} READMEs, cleaned {n_mocs_cleaned} domain MOCs.")


if __name__ == "__main__":
    main()
