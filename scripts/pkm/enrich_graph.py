#!/usr/bin/env python3
"""Phase 14a: enrich graph with bidirectional MOC links and folder-sibling links.

Strategy:
  1. For each domain MOC, append a `## Linked Notes` section listing every note that
     has `up:: [[<this MOC>]]`. Real [[wikilink]] format → creates incoming graph edges.
  2. For each note, set/update `siblings::` inline field listing other notes in the
     same immediate parent folder (max 8). Creates dense intra-folder clusters.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import DEFAULT_EXCLUDE, iter_vault_notes, read_note, write_note  # noqa: E402

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
EXCLUDE = (*DEFAULT_EXCLUDE, "_templates")
UP_RE = re.compile(r"^up::\s*\[\[([^\]]+)\]\]\s*$", re.MULTILINE)
SIBLINGS_RE = re.compile(r"^siblings::\s*.*$", re.MULTILINE)
SECTION_HEADER = "## Linked Notes (auto)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-mocs", action="store_true", help="Skip MOC enrichment step")
    ap.add_argument("--skip-siblings", action="store_true", help="Skip siblings step")
    args = ap.parse_args()

    notes = list(iter_vault_notes(VAULT, exclude=EXCLUDE))

    # === Pass 1: collect up:: → MOC mapping ===
    moc_children: dict[str, list[Path]] = defaultdict(list)
    for p in notes:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        m = UP_RE.search(text)
        if m:
            moc_name = m.group(1).strip()
            moc_children[moc_name].append(p)

    # === Step 1: Update each domain MOC with linked notes section ===
    n_mocs_updated = 0
    if not args.skip_mocs:
        moc_dir = VAULT / "MOCs"
        for moc_name, children in moc_children.items():
            moc_path = moc_dir / f"{moc_name}.md"
            if not moc_path.exists():
                continue
            fm, body = read_note(moc_path)

            # Build the section content
            children_sorted = sorted(children, key=lambda p: p.stem.lower())
            lines = [SECTION_HEADER, "", f"> {len(children_sorted)} notes link up to this MOC.", ""]
            for c in children_sorted:
                lines.append(f"- [[{c.stem}]]")
            section = "\n".join(lines)

            # Replace existing section if present, else append before Dataview block
            section_re = re.compile(
                rf"^{re.escape(SECTION_HEADER)}\s*\n.*?(?=\n##\s|\Z)",
                re.DOTALL | re.MULTILINE,
            )
            if section_re.search(body):
                new_body = section_re.sub(section + "\n", body)
            else:
                # Insert before "## All notes (auto)" Dataview section if exists
                dv_re = re.compile(r"^##\s+All\s+.*?\(auto\)", re.MULTILINE)
                m = dv_re.search(body)
                if m:
                    new_body = body[:m.start()] + section + "\n\n" + body[m.start():]
                else:
                    new_body = body.rstrip() + "\n\n" + section + "\n"

            if new_body != body:
                if args.dry_run:
                    print(f"DRY MOC: {moc_path.relative_to(VAULT)}  +{len(children_sorted)} note links")
                else:
                    write_note(moc_path, fm, new_body)
                n_mocs_updated += 1

    # === Step 2: Inject siblings:: inline field per note ===
    # Fallback: if a note is alone in its parent folder, look at grandparent for siblings.
    n_notes_with_siblings = 0
    if not args.skip_siblings:
        by_folder: dict[Path, list[Path]] = defaultdict(list)
        for p in notes:
            if p.parent.name in {"MOCs"}:
                continue
            if p.name == "README.md":
                continue
            by_folder[p.parent].append(p)

        # Helper: pick siblings, falling back up the directory tree
        def pick_siblings(p: Path) -> list[Path]:
            cur = p.parent
            for _ in range(4):  # walk up to 4 levels
                members = [m for m in by_folder.get(cur, []) if m.resolve() != p.resolve()]
                if len(members) >= 1:
                    return sorted(members, key=lambda m: m.stem.lower())[:8]
                # Aggregate from siblings of cur (grandparent's children that are folders)
                if cur.parent and cur.parent != VAULT:
                    grand_members: list[Path] = []
                    for sub_folder, sub_members in by_folder.items():
                        try:
                            if sub_folder.parent.resolve() == cur.parent.resolve() and sub_folder != cur:
                                grand_members.extend(sub_members)
                        except Exception:
                            continue
                    if grand_members:
                        return sorted(grand_members, key=lambda m: m.stem.lower())[:8]
                    cur = cur.parent
                else:
                    break
            return []

        for p in notes:
            if p.parent.name == "MOCs":
                continue
            if p.name == "README.md":
                continue
            others = pick_siblings(p)
            if not others:
                continue
            sibling_links = ", ".join(f"[[{m.stem}]]" for m in others)
            target_line = f"siblings:: {sibling_links}"

            fm, body = read_note(p)
            if SIBLINGS_RE.search(body):
                new_body = SIBLINGS_RE.sub(target_line, body, count=1)
            else:
                m_up = UP_RE.search(body)
                if m_up:
                    idx = m_up.end()
                    new_body = body[:idx] + "\n" + target_line + body[idx:]
                else:
                    new_body = target_line + "\n\n" + body

            if new_body != body:
                if args.dry_run:
                    print(f"DRY SIB: {p.relative_to(VAULT)}  +{len(others)} siblings")
                else:
                    write_note(p, fm, new_body)
                n_notes_with_siblings += 1

    print(f"\nSummary:")
    print(f"  MOCs updated: {n_mocs_updated}")
    print(f"  Notes with siblings:: : {n_notes_with_siblings}")


if __name__ == "__main__":
    main()
