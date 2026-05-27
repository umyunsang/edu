#!/usr/bin/env python3
"""Phase 9: scan and report broken wikilinks across the vault."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import DEFAULT_EXCLUDE, iter_vault_notes  # noqa: E402

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
EXCLUDE = (*DEFAULT_EXCLUDE, "_templates")
WIKI_PAT = re.compile(r"(?<!\!)\[\[([^\]|#]+?)(?:#[^\]|]*)?(\|[^\]]*)?\]\]")


def build_index() -> tuple[set[str], set[str]]:
    """Index by exact filename basename (NFC-normalized, no .md stripped)."""
    import os
    import unicodedata
    stems: set[str] = set()
    paths: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(VAULT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE]
        rel_dir = Path(unicodedata.normalize("NFC", dirpath)).relative_to(VAULT)
        for fn in filenames:
            name = unicodedata.normalize("NFC", fn)
            rel = rel_dir / name
            if name.endswith(".md"):
                stems.add(name[:-3])
            else:
                stems.add(name)
            stems.add(name)
            paths.add(str(rel).replace("\\", "/"))
            if name.endswith(".md"):
                paths.add(str(rel)[:-3].replace("\\", "/"))
    return stems, paths


def main():
    stems, paths = build_index()
    broken: list[tuple[Path, str, int]] = []
    for p in iter_vault_notes(VAULT, exclude=EXCLUDE):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        in_code = False
        for line_no, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            for m in WIKI_PAT.finditer(line):
                target = m.group(1).strip()
                if not target:
                    continue
                if target.startswith("http") or target.startswith("attachment:"):
                    continue
                # Skip Python/JS-looking code patterns
                if target.startswith(("'", '"', "$", "{", "[")):
                    continue
                if any(c in target for c in ("**", "$")):
                    continue
                if "," in target and any(c in target for c in ("'", '"')):
                    continue
                if target.replace(".", "").replace(",", "").replace(" ", "").isdigit():
                    continue
                import unicodedata as _ud
                target_nfc = _ud.normalize("NFC", target)
                base = target_nfc.rsplit("/", 1)[-1]
                stem_candidate = base[:-3] if base.endswith(".md") else base
                if stem_candidate in stems or base in stems:
                    continue
                norm = target_nfc.lstrip("/").removesuffix(".md")
                if norm in paths:
                    continue
                broken.append((p, target, line_no))

    print(f"Broken wikilinks: {len(broken)}")
    for src, tgt, ln in broken[:80]:
        print(f"  {src.relative_to(VAULT)}:{ln}  →  [[{tgt}]]")
    if len(broken) > 80:
        print(f"  ... and {len(broken)-80} more")


if __name__ == "__main__":
    main()
