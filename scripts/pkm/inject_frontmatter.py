#!/usr/bin/env python3
"""Phase 3+4 frontmatter migration."""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import iter_vault_notes, merge_frontmatter, read_note, write_note  # noqa: E402

TODAY = dt.date.today().isoformat()
VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")


def git_first_commit_date(path: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "log", "--diff-filter=A", "--follow", "--format=%ai", "--", str(path.relative_to(VAULT))],
            cwd=VAULT, text=True, stderr=subprocess.DEVNULL,
        )
        lines = [l for l in out.splitlines() if l.strip()]
        if lines:
            return lines[-1].split(" ")[0]
    except subprocess.CalledProcessError:
        pass
    return None


def infer_semester(rel: Path) -> str:
    parts = rel.parts
    if not parts:
        return "extracurricular"
    top = parts[0]
    if top == "ComputerScience" and len(parts) > 1:
        sub = parts[1]
        m = re.match(r"^(\d-\d)_", sub)
        if m:
            return m.group(1)
        if sub.startswith("elective"):
            return "elective"
        return "extracurricular"
    if top == "certifications":
        return "cert"
    if top == "LGAimer":
        return "extracurricular"
    if top == "MOCs":
        return "all"
    return "extracurricular"


def infer_course(rel: Path) -> str:
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "ComputerScience":
        sub = parts[1]
        m = re.match(r"^(?:\d-\d|elective)_(.+)$", sub)
        if m:
            return m.group(1)
        return sub
    if parts[0] == "certifications":
        return "certification"
    if parts[0] == "LGAimer":
        return "lgaimer"
    if parts[0] == "MOCs":
        return "cross-curriculum"
    return "uncategorized"


def infer_type(rel: Path, has_existing_type: str | None = None) -> str:
    if has_existing_type:
        return has_existing_type
    name = rel.name
    if name == "README.md":
        return "index"  # README is folder entry doc, not conceptual parent
    if "MOC" in name:
        return "MOC"
    if rel.parts and rel.parts[0] == "MOCs":
        return "MOC"
    if any(seg in {"과제", "프로젝트", "실습"} for seg in rel.parts):
        return "project"
    if any(seg in {"papers", "reading", "교재"} for seg in rel.parts):
        return "literature"
    return "lecture"


def build_new_fields(path: Path) -> dict:
    rel = path.relative_to(VAULT)
    created = git_first_commit_date(path) or TODAY
    return {
        "type": infer_type(rel),
        "status": "seedling",
        "semester": infer_semester(rel),
        "course": infer_course(rel),
        "created": created,
        "updated": TODAY,
        "source": "",
    }


def process(path: Path, dry_run: bool) -> tuple[bool, str]:
    fm, body = read_note(path)
    new_fields = build_new_fields(path)
    rel = path.relative_to(VAULT)
    if not fm:
        m = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        title = m.group(1).strip() if m else path.stem
        full = {
            "title": title,
            "date": new_fields["created"],
            "aliases": [],
            "tags": [f"type/{new_fields['type']}"],
            **new_fields,
        }
        if dry_run:
            return True, f"PHASE3 inject {rel}"
        write_note(path, full, body)
        return True, f"PHASE3 wrote {rel}"
    else:
        existing_type = fm.get("type")
        if not existing_type:
            for tag in (fm.get("tags") or []):
                if isinstance(tag, str) and tag.startswith("type/"):
                    existing_type = tag.split("/", 1)[1]
                    break
        if existing_type:
            new_fields["type"] = existing_type
        if fm.get("status"):
            new_fields["status"] = fm["status"]
        if fm.get("date") and not fm.get("created"):
            new_fields["created"] = str(fm["date"])
        merged = merge_frontmatter(fm, new_fields, protected=("title", "date", "aliases"))
        existing_tags = set(merged.get("tags") or [])
        existing_tags.add(f"type/{new_fields['type']}")
        merged["tags"] = sorted(existing_tags)
        if dry_run:
            added = set(merged) - set(fm)
            return True, f"PHASE4 merge {rel} → +{added}"
        write_note(path, merged, body)
        return True, f"PHASE4 wrote {rel}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--phase", choices=["3", "4", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    notes = list(iter_vault_notes(VAULT, exclude=(".obsidian", ".git", ".claude", ".pytest_cache", "scripts", "docs", "_templates")))
    n_phase3 = n_phase4 = 0
    bad = []
    for p in notes:
        try:
            fm, _ = read_note(p)
        except Exception as e:
            bad.append((p, str(e).splitlines()[0]))
            continue
        if not fm and args.phase in ("3", "both"):
            ok, msg = process(p, args.dry_run)
            if ok:
                n_phase3 += 1
                print(msg)
        elif fm and args.phase in ("4", "both"):
            need_keys = {"type", "status", "semester", "course", "created", "updated", "source"}
            if need_keys.issubset(fm.keys()):
                continue
            ok, msg = process(p, args.dry_run)
            if ok:
                n_phase4 += 1
                print(msg)
        if args.limit and (n_phase3 + n_phase4) >= args.limit:
            break

    print(f"\nSummary: phase3={n_phase3}, phase4={n_phase4}, dry_run={args.dry_run}")
    if bad:
        print(f"\n{len(bad)} files with parse errors (skipped — need manual fix):")
        for p, err in bad:
            print(f"  {p.relative_to(VAULT)}: {err}")


if __name__ == "__main__":
    main()
