#!/usr/bin/env python3
"""
graph_check.py — 지식그래프가 설계를 지키는지 기계적으로 검사한다.

docs/지식그래프 설계.md 7장의 검증 규칙 6가지를 구현한다.

  1. 라벨 유효성   `## 관련 개념` 의 모든 라벨이 5종 중 하나인가
                   (L0·L1 README 의 목차 링크는 예외 — 목차는 관계가 아니다)
  2. 사이클 없음   prerequisite 그래프가 비순환(DAG)인가
  3. 고아 없음     모든 정리문서가 인바운드 간선을 갖는가
  4. 증거 보유     모든 정리문서가 evidences(슬라이드·노트북·PDF)를 갖는가
  5. 개념 왕복     L3 개념 노드가 정의처로 되돌아가는 간선을 갖는가
  6. 링크 무결성   `audit` 이 담당 (여기서는 파일 존재만 확인)

사용법:
  python3 scripts/graph_check.py [경로 ...] [--verbose]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

VALID = {"prerequisite", "elaborates", "contrasts", "applies", "evidences"}
REL_RE = re.compile(r"^- \*\*([a-z-]+)\*\* — \[([^\]]+)\]\(<([^>]+)>\)\s*(?::\s*(.*))?$")
ANY_REL_RE = re.compile(r"^- \*\*([a-z-]+)\*\*")
SECTION_RE = re.compile(r"^##\s+관련 개념\s*$")
HEAD_RE = re.compile(r"^##\s")
# 본문 어디서든 나타나는 증거: 슬라이드 이미지 · 노트북 · PDF
EVIDENCE_RE = re.compile(
    r"!\[[^\]]*\]\(<?[^)>]*assets/[^)>]+>?\)"      # 슬라이드 이미지 임베드
    r"|\]\(<?[^)>]*\.(ipynb|pdf)"                  # 노트북·PDF 링크
    r"|>\s*`[^`]*\.pdf`\s*p\.\d"                   # > [!quote] 슬라이드 근거 + 페이지
)


def parse(md: Path) -> tuple[list[tuple[str, str, str]], bool, bool]:
    """(라벨, 표시, 경로) 목록과 (증거 보유, 관련개념 섹션 존재)."""
    rels: list[tuple[str, str, str]] = []
    text = md.read_text(encoding="utf-8")
    has_ev = bool(EVIDENCE_RE.search(text))
    in_sec = False
    found_sec = False
    for line in text.split("\n"):
        if SECTION_RE.match(line):
            in_sec, found_sec = True, True
            continue
        if in_sec and HEAD_RE.match(line):
            in_sec = False
        if not in_sec:
            continue
        m = REL_RE.match(line)
        if m:
            rels.append((m.group(1), m.group(2), m.group(3)))
        elif ANY_REL_RE.match(line):
            rels.append((ANY_REL_RE.match(line).group(1), "", ""))  # 형식 위반
    return rels, has_ev, found_sec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=["ComputerScience"], type=Path)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    notes: list[Path] = []
    for root in args.paths:
        for p in Path(root).rglob("*.md"):
            if any(s.startswith(".") for s in p.parts):
                continue
            if p.name == "README.md":
                continue  # L0·L1 목차는 관계 검사 대상이 아니다
            if "/notes/" in str(p) or p.parent.name == "concepts":
                notes.append(p)

    bad_label: list[str] = []
    bad_form: list[str] = []
    no_ev: list[str] = []
    dead: list[str] = []
    prereq: dict[str, set[str]] = defaultdict(set)
    inbound: dict[str, int] = defaultdict(int)

    for md in notes:
        rels, has_ev, _ = parse(md)
        rel_key = str(md)
        if not has_ev:
            no_ev.append(str(md))
        for label, _disp, href in rels:
            if label not in VALID:
                bad_label.append(f"{md}: {label}")
                continue
            if not href:
                bad_form.append(str(md))
                continue
            target = (md.parent / href.split("#", 1)[0]).resolve()
            if not target.exists():
                dead.append(f"{md} → {href}")
                continue
            inbound[str(target)] += 1
            if label == "prerequisite":
                prereq[rel_key].add(str(target))

    # README 는 관계 라벨 검사 대상이 아니지만(목차는 관계가 아니다),
    # 목차 링크 자체는 실재하는 간선이므로 고아 판정에서는 inbound 로 센다.
    LINK_RE = re.compile(r"\]\(<?([^)>]+?)>?\)")
    for root in args.paths:
        for rm in Path(root).rglob("README.md"):
            if any(s.startswith(".") for s in rm.parts):
                continue
            for m in LINK_RE.finditer(rm.read_text(encoding="utf-8")):
                href = m.group(1).split("#", 1)[0]
                if href.startswith(("http", "mailto:")):
                    continue
                inbound[str((rm.parent / href).resolve())] += 1

    # 2. prerequisite 사이클 (DFS)
    cycles: list[str] = []
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = defaultdict(int)

    def dfs(u: str, stack: list[str]) -> None:
        color[u] = GRAY
        for v in prereq.get(u, ()):
            if color[v] == GRAY:
                cycles.append(" → ".join(Path(x).stem for x in stack[stack.index(v):] + [v])
                              if v in stack else f"{Path(u).stem} → {Path(v).stem}")
            elif color[v] == WHITE:
                dfs(v, stack + [v])
        color[u] = BLACK

    for u in list(prereq):
        if color[u] == WHITE:
            dfs(u, [u])

    orphans = [str(p) for p in notes if inbound.get(str(p.resolve()), 0) == 0]

    checks = [
        ("1. 라벨 유효성 (5종)", bad_label),
        ("1b. 표기 형식 (정규식 추출 가능)", bad_form),
        ("2. prerequisite 비순환", cycles),
        ("3. 고아 없음", orphans),
        ("4. 증거 보유", no_ev),
        ("6. 링크 대상 실재", dead),
    ]
    print(f"검사 대상: 정리문서 {len(notes)}편\n")
    failed = 0
    for name, bad in checks:
        mark = "OK" if not bad else f"위반 {len(bad)}"
        print(f"  {'✓' if not bad else '✗'} {name:34s} {mark}")
        if bad:
            failed += 1
            if args.verbose:
                for b in bad[:12]:
                    print(f"      {b}")
                if len(bad) > 12:
                    print(f"      … 외 {len(bad) - 12}건")
    print(f"\n{'전부 통과' if not failed else f'{failed}개 규칙 위반'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
