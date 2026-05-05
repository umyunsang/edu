#!/usr/bin/env python3
"""Phase 18: slim README files to context cards.

Per community research (LYT, Matuschak, kepano):
- README is GitHub convention, not Obsidian primitive
- README should NOT compete with MOCs as graph hub
- README role: 1-3 sentence folder context + up:: link to course/domain MOC

Strategy:
- Strip auto-injected Dataview blocks (those duplicate domain MOC's coverage)
- Strip auto-injected `## Linked Notes` sections (READMEs aren't parents)
- Preserve user-curated body content (intro paragraphs, manual lists, etc.)
- Ensure up:: link points to domain MOC
- Set type=index, status=evergreen, tags include type/index (not type/MOC)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import iter_vault_notes, read_note, write_note  # noqa: E402

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
EXCLUDE = (".obsidian", ".git", ".claude", ".pytest_cache", "scripts", "docs", "_templates")

# Pattern: ## All notes in this course (auto)\n```dataview\n...\n```\n
DATAVIEW_AUTO_RE = re.compile(
    r"\n##\s+All notes in this course \(auto\)\s*\n```dataview\s*\n.*?\n```\s*\n*",
    re.DOTALL,
)
# Phase 14a auto-injected section
LINKED_NOTES_RE = re.compile(
    r"\n##\s+Linked Notes \(auto\)\s*\n.*?(?=\n##\s|\Z)",
    re.DOTALL,
)


def main():
    n = 0
    for p in iter_vault_notes(VAULT, exclude=EXCLUDE):
        if p.name != "README.md":
            continue
        if p.parent.name == "MOCs":
            continue

        fm, body = read_note(p)
        original = body

        # Strip auto-injected blocks
        body = DATAVIEW_AUTO_RE.sub("\n", body)
        body = LINKED_NOTES_RE.sub("\n", body)
        body = re.sub(r"\n{3,}", "\n\n", body).rstrip() + "\n"

        # Ensure type=index (not MOC)
        fm["type"] = "index"
        fm["status"] = "evergreen"
        tags = set(fm.get("tags") or [])
        tags.discard("type/MOC")
        tags.add("type/index")
        fm["tags"] = sorted(tags)

        if body != original or fm != fm:  # always write to ensure consistency
            write_note(p, fm, body)
            print(f"slimmed: {p.relative_to(VAULT)}")
            n += 1

    print(f"\nSlimmed {n} README files.")


if __name__ == "__main__":
    main()
