#!/usr/bin/env python3
"""
strip_obsidian.py — 옵시디언 전용 문법을 OK 네이티브 마크다운으로 변환한다.

세 가지를 처리한다.
  1. Dataview 인라인 필드 (`up:: [[...]]`)  → 줄 삭제 (화이트리스트에 있는 필드만)
  2. 위키링크 (`[[경로|표시]]`)              → `[표시](<상대경로.md>)`, 대상이 없으면 평문
  3. 계층 태그 (`- type/lecture`)            → `- lecture` (프론트매터 tags 안에서만)

안전장치
  - ``` / ~~~ 코드블록 안은 절대 건드리지 않는다 (C++ `std::`, `thrust::` 보호)
  - 제거 필드는 화이트리스트. 목록에 없는 `xxx::` 는 그대로 둔다
  - 위키링크 대상이 실재할 때만 링크로 만든다. 없으면 표시 텍스트만 남긴다

사용법:
  python3 scripts/strip_obsidian.py [경로 ...] [--dry-run] [--report FILE]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# 삭제 대상 Dataview 인라인 필드. 여기 없는 `xxx::` 는 손대지 않는다.
# (std:: / thrust:: 같은 C++ 네임스페이스가 자동으로 보호된다)
DATAVIEW_FIELDS = {
    "kg_profile", "kg_evidence", "kg_query_mode", "kg_parent", "kg_concepts",
    "kg_skeleton", "kg_source", "kg_related",
    "up", "related", "module", "domain", "stage", "bridge", "prerequisites",
    "next", "prev", "graph", "graph_design", "method", "schema",
}

FENCE_RE = re.compile(r"^\s*(```|~~~)")
DATAVIEW_RE = re.compile(r"^([a-z_][a-z0-9_]*)::\s")
EMBED_RE = re.compile(r"!\[\[([^\[\]]+?)\]\]")      # ![[...]] — 반드시 먼저 처리
WIKILINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")
FM_TAG_RE = re.compile(r"^(\s*-\s+)([a-z_]+)/(.+?)\s*$")

IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".avif"}

# OKF 는 status 를 draft|stable|deprecated 로 제한한다. 옛 값들을 매핑한다.
STATUS_MAP = {
    "seedling": "draft", "budding": "draft", "sprout": "draft", "wip": "draft",
    "evergreen": "stable", "active": "stable", "mature": "stable", "done": "stable",
    "archived": "deprecated", "obsolete": "deprecated",
}
FM_STATUS_RE = re.compile(r"^(\s*status:\s*)['\"]?([a-zA-Z]+)['\"]?\s*$")


def relative(dest: Path, src_dir: Path) -> str:
    """./ 접두어를 붙인 상대 경로 (OK 관례)."""
    rel = os.path.relpath(dest, src_dir)
    return rel if rel.startswith(".") else "./" + rel


def build_index(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """문서 인덱스를 만든다. (전체경로 → Path, stem → [Path])"""
    by_path: dict[str, Path] = {}
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for p in root.rglob("*.md"):
        if any(part.startswith(".") for part in p.parts):
            continue
        rel = p.relative_to(root)
        by_path[str(rel.with_suffix(""))] = p
        by_stem[p.stem].append(p)
    return by_path, dict(by_stem)


ASSET_EXT = IMG_EXT | {".pdf", ".pptx", ".xlsx", ".docx", ".ipynb", ".py", ".csv"}


def build_asset_index(root: Path) -> dict[str, list[Path]]:
    """임베드 대상(이미지·PDF 등)을 파일명으로 찾을 수 있게 색인한다."""
    idx: dict[str, list[Path]] = defaultdict(list)
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in ASSET_EXT:
            continue
        if any(part.startswith(".") for part in p.parts):
            continue
        idx[p.name].append(p)
    return dict(idx)


def convert_embeds(line: str, src: Path, root: Path, assets: dict,
                   stats: dict) -> str:
    """![[파일]] 옵시디언 임베드 → 마크다운 이미지/링크."""

    def repl(m: re.Match) -> str:
        inner = m.group(1)
        target_raw, display = (inner.split("|", 1) + [""])[:2]
        anchor = ""
        if "#" in target_raw:
            target_raw, anchor = target_raw.split("#", 1)
        target_raw = target_raw.strip()
        name = target_raw.rsplit("/", 1)[-1]
        ext = Path(name).suffix.lower()

        # 1) 문서 기준 상대 경로에 실재하는가  2) 파일명으로 유일하게 찾히는가
        cand = (src.parent / target_raw).resolve()
        if not cand.is_file():
            hits = assets.get(name, [])
            cand = hits[0] if len(hits) == 1 else None

        if cand is None:
            # 커밋 305a1fa5 에서 image/ 1,722개가 의도적으로 제거됐다.
            # 링크로 남기면 no-such-file 이 영구히 쌓이므로, 자리 정보만 주석으로 보존한다.
            stats["embed_unresolved"] += 1
            return f"<!-- 원본 이미지 없음: {name} -->"
        rel = relative(cand, src.parent)

        label = display.strip() or Path(name).stem
        if ext in IMG_EXT:
            stats["embed_img"] += 1
            return f"![{label}](<{rel}>)"
        if anchor.startswith("page="):
            label += f" p.{anchor[5:]}"
        stats["embed_link"] += 1
        return f"[{label}](<{rel}>)"

    return EMBED_RE.sub(repl, line)


def resolve_target(raw: str, by_path: dict, by_stem: dict) -> Path | None:
    """위키링크 대상 문자열을 실제 파일로 해석한다."""
    target = raw.split("#", 1)[0].strip()
    if not target:
        return None
    if target in by_path:
        return by_path[target]
    if target.endswith(".md") and target[:-3] in by_path:
        return by_path[target[:-3]]
    stem = target.rsplit("/", 1)[-1]
    hits = by_stem.get(stem, [])
    if len(hits) == 1:  # 동명이인이 없을 때만 신뢰한다
        return hits[0]
    return None


def convert_wikilinks(line: str, src: Path, by_path: dict, by_stem: dict,
                      stats: dict) -> str:
    def repl(m: re.Match) -> str:
        inner = m.group(1)
        if "|" in inner:
            target_raw, display = inner.split("|", 1)
        else:
            target_raw, display = inner, inner.rsplit("/", 1)[-1]
        display = display.strip()
        dest = resolve_target(target_raw, by_path, by_stem)
        if dest is None:
            stats["wikilink_plain"] += 1
            return display  # 대상 부재 → 평문으로 강등 (dead link 생성 방지)
        stats["wikilink_link"] += 1
        return f"[{display}](<{relative(dest, src.parent)}>)"

    return WIKILINK_RE.sub(repl, line)


def process(path: Path, root: Path, by_path: dict, by_stem: dict,
            assets: dict, stats: dict) -> str | None:
    original = path.read_text(encoding="utf-8")
    lines = original.split("\n")
    out: list[str] = []

    in_fence = False
    in_fm = False
    fm_done = False
    seen_tags: set[str] = set()

    for i, line in enumerate(lines):
        # --- 프론트매터 경계 ---
        if i == 0 and line.strip() == "---":
            in_fm, out = True, out + [line]
            continue
        if in_fm and line.strip() == "---":
            in_fm, fm_done = False, True
            out.append(line)
            continue

        # --- 코드블록 경계 (프론트매터 밖에서만) ---
        if not in_fm and FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)  # 코드는 원본 그대로
            continue

        # --- 프론트매터: status 정규화 + 계층 태그 평면화 ---
        if in_fm:
            ms = FM_STATUS_RE.match(line)
            if ms and ms.group(2) in STATUS_MAP:
                stats["status_norm"] += 1
                out.append(f"{ms.group(1)}{STATUS_MAP[ms.group(2)]}")
                continue
            m = FM_TAG_RE.match(line)
            if m:
                flat = m.group(3).strip()
                if flat in seen_tags:
                    stats["tag_dedup"] += 1
                    continue  # 중복 태그는 줄째로 버린다
                seen_tags.add(flat)
                stats["tag_flat"] += 1
                out.append(f"{m.group(1)}{flat}")
                continue
            out.append(line)
            continue

        # --- 본문: Dataview 인라인 필드 삭제 ---
        m = DATAVIEW_RE.match(line)
        if m and m.group(1) in DATAVIEW_FIELDS:
            stats["dataview"] += 1
            continue

        # --- 본문: 임베드를 먼저, 그다음 일반 위키링크 ---
        if "[[" in line:
            line = convert_embeds(line, path, root, assets, stats)
            line = convert_wikilinks(line, path, by_path, by_stem, stats)

        out.append(line)

    # Dataview 필드를 지운 자리에 빈 줄이 3개 이상 남는 것을 정리
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not fm_done:
        stats["no_frontmatter"] += 1
    return text if text != original else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=["."], type=Path)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    root = args.root.resolve()
    by_path, by_stem = build_index(root)
    assets = build_asset_index(root)
    print(f"인덱스: 문서 {len(by_path)}개 · 자산 "
          f"{sum(len(v) for v in assets.values())}개", file=sys.stderr)

    targets: list[Path] = []
    for p in args.paths:
        p = Path(p)
        if p.is_file() and p.suffix == ".md":
            targets.append(p.resolve())
        elif p.is_dir():
            targets += [q for q in p.resolve().rglob("*.md")
                        if not any(s.startswith(".") for s in q.parts)]

    stats = defaultdict(int)
    changed: list[str] = []
    for t in sorted(set(targets)):
        new = process(t, root, by_path, by_stem, assets, stats)
        if new is None:
            continue
        changed.append(str(t.relative_to(root)))
        if not args.dry_run:
            t.write_text(new, encoding="utf-8")

    print(f"\n{'[dry-run] ' if args.dry_run else ''}변경 파일: {len(changed)}개")
    for k in sorted(stats):
        print(f"  {k:20s} {stats[k]}")

    if args.report:
        args.report.write_text("\n".join(changed) + "\n", encoding="utf-8")
        print(f"목록 → {args.report}")


if __name__ == "__main__":
    main()
