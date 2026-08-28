#!/usr/bin/env python3
"""
restructure_courses.py — 모든 과목 폴더를 동일한 배치로 재편한다.

  <과목>/
  ├── README.md      진입점 (기존 `00. * 인덱스.md` 를 승격)
  ├── notes/         ③ 쓴 것 — 정리문서 (.md, 평면)
  ├── sources/       ① 받은 것 — 강의 슬라이드·시험지 (불변)
  ├── work/          ② 만든 것 — 실습 코드·노트북·과제
  └── .ok/           (그대로)

폴더는 "가공 단계"로 나눈다. 종류는 확장자가, 순서는 지식그래프가 담당한다.
해당하는 파일이 없으면 그 폴더는 만들지 않는다.

이동 후 모든 마크다운 링크를 새 경로 기준으로 다시 계산한다.
숨김 폴더(.omo, .aioss-eval, .ok ...)와 assets/ 는 건드리지 않는다.

사용법:
  python3 scripts/restructure_courses.py <과목경로 ...> [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SOURCE_EXT = {".pdf", ".pptx", ".ppt", ".docx", ".doc", ".hwp", ".hwpx", ".key"}
WORK_EXT = {".ipynb", ".py", ".c", ".cpp", ".cc", ".h", ".hpp", ".java", ".sql",
            ".js", ".ts", ".html", ".css", ".sh", ".r", ".m", ".cu", ".asm"}
KEEP_DIRS = {"assets", ".ok", "notes", "sources", "work"}
INDEX_RE = re.compile(r"^0*0[.\s].*인덱스.*\.md$")

# [텍스트](<경로>) 또는 [텍스트](경로) — 외부 URL 은 건드리지 않는다
LINK_RE = re.compile(r"(!?\[[^\]]*\]\()(<)?([^)<>]+?)(>)?(\))")


def classify(p: Path) -> str | None:
    """파일이 어느 폴더로 가야 하는지."""
    ext = p.suffix.lower()
    if ext in {".md", ".mdx"}:
        return "notes"
    if ext in SOURCE_EXT:
        return "sources"
    if ext in WORK_EXT:
        return "work"
    return None  # 분류 불가 → 제자리 유지


def plan_course(course: Path) -> tuple[dict[Path, Path], Path | None]:
    """이동 계획 (원본 → 대상) 과 README 승격 대상을 만든다."""
    moves: dict[Path, Path] = {}
    readme_src: Path | None = None

    for p in sorted(course.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(course)
        # 숨김 경로·이미 자리 잡은 폴더는 제외
        if any(part.startswith(".") for part in rel.parts):
            continue
        if rel.parts[0] in KEEP_DIRS:
            continue
        if rel.name == "README.md":
            continue

        # 과목 인덱스는 README 로 승격
        if len(rel.parts) == 1 and INDEX_RE.match(rel.name):
            readme_src = p
            moves[p] = course / "README.md"
            continue

        bucket = classify(p)
        if bucket is None:
            continue
        dest = course / bucket / rel.name
        n = 1
        while dest in moves.values() or (dest.exists() and dest != p):
            dest = course / bucket / f"{Path(rel.name).stem}~{n}{p.suffix}"
            n += 1
        if dest != p:
            moves[p] = dest
    return moves, readme_src


def rewrite_links(root: Path, moved: dict[Path, Path], dry: bool) -> int:
    """이동으로 깨질 마크다운 링크를 새 경로로 고쳐 쓴다."""
    changed = 0
    for md in root.rglob("*.md"):
        if any(part.startswith(".") for part in md.relative_to(root).parts):
            continue
        # 이 문서 자신이 옮겨졌다면 새 위치를 기준으로 계산한다
        md_new = moved.get(md, md)
        text = md.read_text(encoding="utf-8")

        def repl(m: re.Match) -> str:
            head, lt, target, gt, tail = m.groups()
            if re.match(r"^(https?:|mailto:|#)", target):
                return m.group(0)
            old_abs = (md.parent / target.split("#", 1)[0]).resolve()
            new_abs = moved.get(old_abs, old_abs if old_abs.exists() else None)
            if new_abs is None:
                return m.group(0)
            rel = os.path.relpath(new_abs, md_new.parent)
            if not rel.startswith("."):
                rel = "./" + rel
            anchor = "#" + target.split("#", 1)[1] if "#" in target else ""
            return f"{head}<{rel}{anchor}>{tail}"

        new = LINK_RE.sub(repl, text)
        if new != text:
            changed += 1
            if not dry:
                md.write_text(new, encoding="utf-8")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("courses", nargs="+", type=Path)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    all_moves: dict[Path, Path] = {}
    per_course: dict[str, dict] = {}

    for c in args.courses:
        course = c.resolve()
        if not course.is_dir():
            print(f"건너뜀 (폴더 아님): {c}", file=sys.stderr)
            continue
        moves, readme = plan_course(course)
        all_moves.update(moves)
        buckets = defaultdict(int)
        for src, dest in moves.items():
            buckets[dest.parent.name if dest.name != "README.md" else "README"] += 1
        per_course[str(course.relative_to(root))] = dict(buckets)

    for name, buckets in per_course.items():
        line = "  ".join(f"{k} {v}" for k, v in sorted(buckets.items()))
        print(f"{name}\n    {line or '(변경 없음)'}")

    print(f"\n{'[dry-run] ' if args.dry_run else ''}이동 {len(all_moves)}건")

    # 링크를 먼저 고친다. 파일이 아직 옛 위치에 있어야 `md.parent` 기준으로
    # 기존 링크를 해석할 수 있고, 새 경로는 all_moves 로 계산한다.
    n = rewrite_links(root, all_moves, args.dry_run)
    print(f"{'[dry-run] ' if args.dry_run else ''}링크 재작성: {n}개 문서")

    if not args.dry_run and all_moves:
        for src, dest in all_moves.items():
            dest.parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(["git", "mv", "--sparse", str(src), str(dest)],
                               cwd=root, capture_output=True, text=True)
            if r.returncode != 0:  # 미추적 파일은 git mv 가 거부한다
                src.rename(dest)
        for d in sorted(root.rglob("*"), key=lambda p: -len(p.parts)):
            if d.is_dir() and not any(d.iterdir()) and not d.name.startswith("."):
                d.rmdir()  # 이동으로 비워진 폴더 정리


if __name__ == "__main__":
    main()
