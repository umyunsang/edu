#!/usr/bin/env python3
"""
generate_course_readme.py — 과목 README(L1 진입점)를 실제 파일에서 생성한다.

내용을 지어내지 않는다. 세 곳에서만 읽는다.
  - <과목>/.ok/frontmatter.yml   과목 제목·설명
  - <과목>/notes/*.md            각 문서의 frontmatter (title, description)
  - <과목>/{sources,work}/       원본·실습 파일 목록

학습 경로는 notes 의 번호 순서에서 유도한다 (번호가 곧 강의 진도).
기존 README 는 덮어쓰지 않는다 (--force 로만).

사용법:
  python3 scripts/generate_course_readme.py <과목경로 ...> [--force] [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML 필요: pip install pyyaml", file=sys.stderr)
    raise SystemExit(2)

FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)
NUM_RE = re.compile(r"^(\d+)[.\s]")
SOURCE_EXT = {".pdf", ".pptx", ".ppt", ".docx", ".hwp", ".hwpx"}


def read_fm(p: Path) -> dict:
    try:
        m = FM_RE.match(p.read_text(encoding="utf-8"))
        return yaml.safe_load(m.group(1)) or {} if m else {}
    except Exception:
        return {}


def esc(s: str) -> str:
    """표 셀 안에서 파이프를 이스케이프한다."""
    return str(s).replace("|", "\\|").strip()


def mermaid_path(notes: list[tuple[str, Path, dict]]) -> str:
    """번호가 붙은 노트를 순서대로 이어 학습 경로를 만든다."""
    numbered: list[str] = []
    for title, p, fm in notes:
        if NUM_RE.match(p.name):
            numbered.append(fm.get("title") or p.stem)
    if len(numbered) < 2:
        return ""
    ids = []
    lines = ["```mermaid", "flowchart LR"]
    for i, title in enumerate(numbered[:12]):
        nid = f"N{i}"
        ids.append(nid)
        short = title if len(title) <= 22 else title[:21] + "…"
        lines.append(f'    {nid}["{short}"]')
    for a, b in zip(ids, ids[1:]):
        lines.append(f"    {a} --> {b}")
    if len(numbered) > 12:
        lines.append(f'    {ids[-1]} --> MORE["… 외 {len(numbered) - 12}편"]')
    lines.append("```")
    return "\n".join(lines)


def build(course: Path, root: Path) -> str | None:
    notes_dir = course / "notes"
    if not notes_dir.is_dir():
        return None

    meta = {}
    fmy = course / ".ok" / "frontmatter.yml"
    if fmy.is_file():
        try:
            meta = yaml.safe_load(fmy.read_text(encoding="utf-8")) or {}
        except Exception:
            meta = {}

    slug = course.name
    title = meta.get("title") or slug
    desc = (meta.get("description") or "").split("\n")[0].strip()

    notes = []
    for p in sorted(notes_dir.glob("*.md")):
        fm = read_fm(p)
        notes.append((fm.get("title", p.stem), p, fm))
    notes.sort(key=lambda x: x[1].name)

    out: list[str] = []
    out.append("---")
    out.append(yaml.safe_dump({
        "title": title,
        "description": desc or f"{title} 과목의 진입점. 정리문서·원본 자료·실습을 잇는다.",
        "type": "course-index",
        "tags": meta.get("tags", ["course"]),
        "course": slug,
        "semester": meta.get("semester", ""),
        "status": "draft",
        "created": str(date.today()),
        "updated": str(date.today()),
    }, allow_unicode=True, sort_keys=False).strip())
    out.append("---")
    out.append("")

    if desc:
        out += ["> [!abstract] 이 과목은", f"> {desc}", ""]

    path = mermaid_path(notes)
    if path:
        out += ["## 학습 경로", "",
                "번호는 강의 진도 순이다. 앞 문서를 읽었다는 전제로 다음 문서가 쓰인다.",
                "", path, ""]

    if notes:
        out += ["## 정리문서", "", f"모두 `notes/` 에 있다. 총 {len(notes)}편.", "",
                "| 문서 | 다루는 내용 |", "| :-- | :-- |"]
        for t, p, fm in notes:
            d = esc(fm.get("description", "")) or "—"
            out.append(f"| [{esc(t)}](<./notes/{p.name}>) | {d} |")
        out.append("")

    src = sorted([p for p in (course / "sources").glob("*")
                  if p.is_file() and p.suffix.lower() in SOURCE_EXT]) \
        if (course / "sources").is_dir() else []
    if src:
        out += ["## 원본 자료", "",
                f"교수가 배포한 자료다. `sources/` 에 있고 수정하지 않는다. 총 {len(src)}건.", ""]
        for p in src[:40]:
            out.append(f"- `{p.name}`")
        if len(src) > 40:
            out.append(f"- … 외 {len(src) - 40}건")
        out.append("")

    work = sorted([p for p in (course / "work").rglob("*") if p.is_file()]) \
        if (course / "work").is_dir() else []
    if work:
        kinds: dict[str, int] = {}
        for p in work:
            kinds[p.suffix.lower() or "(확장자 없음)"] = kinds.get(p.suffix.lower() or "(확장자 없음)", 0) + 1
        top = sorted(kinds.items(), key=lambda x: -x[1])[:6]
        out += ["## 실습", "",
                f"직접 만든 코드와 산출물이다. `work/` 에 있고 총 {len(work)}건.", "",
                "| 종류 | 개수 |", "| :-- | --: |"]
        out += [f"| `{k}` | {v} |" for k, v in top]
        out.append("")

    assets = list((course / "assets").glob("*")) if (course / "assets").is_dir() else []
    if assets:
        out += ["## 슬라이드 이미지", "",
                f"정리문서가 근거로 인라인 임베드하는 강의 슬라이드다. `assets/` 에 {len(assets)}장.", ""]

    out += ["## 관련 과목", "",
            "> [!note] 아직 비어 있다",
            "> 다른 과목과의 관계는 지식그래프 4단계에서 관계 타입"
            "(`prerequisite` · `elaborates` · `contrasts` · `applies` · `evidences`)과"
            " 함께 채운다. 근거 없이 미리 이어두지 않는다.", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("courses", nargs="+", type=Path)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    made = skipped = 0
    for c in args.courses:
        course = c.resolve()
        if not course.is_dir():
            continue
        readme = course / "README.md"
        if readme.exists() and not args.force:
            skipped += 1
            continue
        text = build(course, root)
        if text is None:
            skipped += 1
            continue
        made += 1
        if not args.dry_run:
            readme.write_text(text, encoding="utf-8")
        print(f"  {'[dry] ' if args.dry_run else ''}{course.relative_to(root)}/README.md")

    print(f"\n생성 {made}편 · 건너뜀 {skipped}편")


if __name__ == "__main__":
    main()
