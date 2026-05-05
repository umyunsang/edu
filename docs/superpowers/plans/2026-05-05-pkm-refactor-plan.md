# Obsidian PKM Infrastructure Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 옵시디언 학부 커리큘럼 볼트(479 노트)에 MOC-first Evergreen PKM 인프라를 전면 구축한다 — 확장 frontmatter, 5축 태그, 15개 MOC, 그래프 뷰 최적화, 6개 Templater 템플릿, 한국어 안전 Linter 규칙.

**Architecture:** 5계층(Theme/Plugin/Data/Navigation/Workflow) 위에 11 Phase로 빌드한다. 모든 데이터 마이그레이션은 dry-run 검증 후 적용. 각 Phase는 단일 git 커밋으로 캡슐화하여 롤백 가능.

**Tech Stack:** Obsidian 1.9+, Python 3 (PyYAML, python-frontmatter), Bash, Git, Obsidian community plugins (Dataview, Excalidraw, obsidian-spaced-repetition, Advanced Tables, Iconize, Graph Analysis, Excalibrain).

**Spec:** `docs/superpowers/specs/2026-05-05-obsidian-pkm-infrastructure-design.md`

**Branch:** `pkm-refactor` (이미 생성됨)

**Vault root:** `/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu`

---

## File Structure

생성/수정 파일 매핑:

```
docs/superpowers/plans/2026-05-05-pkm-refactor-plan.md  (이 문서)

scripts/pkm/                           ← 모든 마이그레이션 스크립트
├── folder_tag_map.json                ← 폴더→태그 매핑 데이터
├── lib_frontmatter.py                 ← 공통 frontmatter 파서/저장
├── inject_frontmatter.py              ← Phase 3·4
├── inject_tags.py                     ← Phase 5
├── consolidate_images.py              ← Phase 6
├── resolve_filename_collisions.py     ← Phase 7
├── upgrade_readmes_to_moc.py          ← Phase 8
├── scan_broken_wikilinks.py           ← Phase 9
└── tests/
    ├── fixtures/                      ← 미니 노트 샘플
    └── test_lib_frontmatter.py        ← 단위 테스트

MOCs/                                  ← Phase 2 산출물
├── Home MOC.md
├── Machine Learning MOC.md
├── Deep Learning MOC.md
├── Algorithms MOC.md
├── Systems MOC.md
├── Computer Vision MOC.md
├── LLM & NLP MOC.md
├── AI Open Source MOC.md
├── Math Foundations MOC.md
├── Database MOC.md
├── Cloud & Containers MOC.md
├── Security MOC.md
├── Software Engineering MOC.md
├── Certifications MOC.md
├── Portfolio MOC.md
└── Open Questions MOC.md

_templates/                            ← Templater 템플릿
├── lecture-note.md
├── literature-note.md
├── permanent-note.md
├── project-note.md
├── moc.md
└── daily-note.md

.obsidian/snippets/
├── fonts.css
├── callouts.css
├── status-badges.css
└── moc-styling.css

.obsidian/community-plugins.json       ← 활성 플러그인 리스트 갱신
.obsidian/graph.json                   ← 그래프 색상 그룹
.obsidian/plugins/obsidian-linter/data.json  ← 한국어 안전 룰셋

이미지 이동: */images/*.png → image/{course}_*.png  (~50 파일)
frontmatter 확장: 모든 .md 노트 (~479 파일)
README → MOC 업그레이드: ~36 파일
```

---

## Task 1: 워크스페이스 점검 및 의존성 설치

**Files:**
- Create: `scripts/pkm/lib_frontmatter.py` (skeleton)
- Verify: `pkm-refactor` 브랜치 활성화

- [ ] **Step 1: 브랜치 확인**

Run:
```bash
cd "/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu"
git branch --show-current
```
Expected: `pkm-refactor`

- [ ] **Step 2: Python 의존성 설치**

Run:
```bash
python3 -m pip install --user python-frontmatter pyyaml
```
Expected: 두 패키지 설치 완료 (이미 설치돼 있으면 스킵)

- [ ] **Step 3: scripts/pkm 디렉터리 생성**

Run:
```bash
mkdir -p scripts/pkm/tests/fixtures
```

- [ ] **Step 4: 커밋**

```bash
git add scripts/pkm/
git commit -m "chore(pkm): scaffold scripts/pkm/ for refactor automation"
```

---

## Task 2: 공통 frontmatter 라이브러리 + 단위 테스트

**Files:**
- Create: `scripts/pkm/lib_frontmatter.py`
- Create: `scripts/pkm/tests/test_lib_frontmatter.py`
- Create: `scripts/pkm/tests/fixtures/sample_with_fm.md`
- Create: `scripts/pkm/tests/fixtures/sample_no_fm.md`

- [ ] **Step 1: 픽스처 작성 — frontmatter 있는 노트**

`scripts/pkm/tests/fixtures/sample_with_fm.md`:
```markdown
---
title: 머신러닝 기초
date: 2026-03-12
tags:
  - ML
aliases:
  - ML 기초
---

# 머신러닝 기초

본문 내용...

[[관련노트]]
```

- [ ] **Step 2: 픽스처 작성 — frontmatter 없는 노트**

`scripts/pkm/tests/fixtures/sample_no_fm.md`:
```markdown
# 무제 노트

본문만 있음.
```

- [ ] **Step 3: 실패 테스트 작성**

`scripts/pkm/tests/test_lib_frontmatter.py`:
```python
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib_frontmatter import read_note, write_note, merge_frontmatter

FIX = pathlib.Path(__file__).parent / "fixtures"


def test_read_note_with_frontmatter():
    fm, body = read_note(FIX / "sample_with_fm.md")
    assert fm["title"] == "머신러닝 기초"
    assert fm["tags"] == ["ML"]
    assert "[[관련노트]]" in body


def test_read_note_without_frontmatter():
    fm, body = read_note(FIX / "sample_no_fm.md")
    assert fm == {}
    assert body.startswith("# 무제 노트")


def test_merge_preserves_existing_keys():
    base = {"title": "원래", "tags": ["A"]}
    new = {"tags": ["A", "B"], "type": "lecture"}
    merged = merge_frontmatter(base, new)
    assert merged["title"] == "원래"
    assert sorted(merged["tags"]) == ["A", "B"]
    assert merged["type"] == "lecture"


def test_merge_does_not_overwrite_protected():
    base = {"title": "원래", "date": "2026-01-01"}
    new = {"title": "새것", "date": "2099-12-31"}
    merged = merge_frontmatter(base, new, protected=["title", "date"])
    assert merged["title"] == "원래"
    assert merged["date"] == "2026-01-01"


def test_write_roundtrip(tmp_path):
    fm = {"title": "테스트", "tags": ["x"]}
    body = "# Hi\n\n본문"
    p = tmp_path / "test.md"
    write_note(p, fm, body)
    fm2, body2 = read_note(p)
    assert fm2 == fm
    assert body2.strip() == body.strip()
```

- [ ] **Step 4: 실패 확인**

Run:
```bash
cd "/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu"
python3 -m pytest scripts/pkm/tests/test_lib_frontmatter.py -v 2>&1 | tail -20
```
Expected: 모듈 임포트 실패 (`ModuleNotFoundError: No module named 'lib_frontmatter'`)

- [ ] **Step 5: lib_frontmatter.py 구현**

`scripts/pkm/lib_frontmatter.py`:
```python
"""Frontmatter read/write utilities for the pkm-refactor.

Korean-text safe: UTF-8, NFC normalization, no BOM.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Iterable

import frontmatter
import yaml


def _yaml_dump(data: dict) -> str:
    """Stable, Korean-safe YAML serialization."""
    return yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=True,
        default_flow_style=False,
        width=4096,
    )


def read_note(path: Path) -> tuple[dict, str]:
    raw = Path(path).read_text(encoding="utf-8")
    raw = unicodedata.normalize("NFC", raw)
    post = frontmatter.loads(raw)
    return dict(post.metadata), post.content


def write_note(path: Path, fm: dict, body: str) -> None:
    body = unicodedata.normalize("NFC", body)
    if fm:
        out = "---\n" + _yaml_dump(fm) + "---\n\n" + body.lstrip("\n")
    else:
        out = body
    if not out.endswith("\n"):
        out += "\n"
    Path(path).write_text(out, encoding="utf-8")


def merge_frontmatter(
    base: dict,
    new: dict,
    protected: Iterable[str] = ("title", "date", "aliases"),
) -> dict:
    """Merge `new` into `base`. Protected keys never overwritten.
    For `tags`, lists are unioned and sorted.
    """
    out = dict(base)
    protected_set = set(protected)
    for k, v in new.items():
        if k in protected_set and k in out:
            continue
        if k == "tags":
            old = out.get("tags") or []
            if isinstance(old, str):
                old = [old]
            if isinstance(v, str):
                v = [v]
            out["tags"] = sorted(set(old) | set(v or []))
        else:
            out[k] = v
    return out


def iter_vault_notes(root: Path, exclude: Iterable[str] = (".obsidian", ".git", "scripts", "docs")) -> Iterable[Path]:
    """Yield all .md notes under `root`, excluding tooling dirs."""
    exclude_set = {Path(e) for e in exclude}
    for p in sorted(Path(root).rglob("*.md")):
        rel = p.relative_to(root)
        if any(part in {str(e) for e in exclude_set} for part in rel.parts):
            continue
        yield p
```

- [ ] **Step 6: 테스트 통과 확인**

Run:
```bash
python3 -m pytest scripts/pkm/tests/test_lib_frontmatter.py -v 2>&1 | tail -20
```
Expected: 5/5 PASS

- [ ] **Step 7: 커밋**

```bash
git add scripts/pkm/lib_frontmatter.py scripts/pkm/tests/
git commit -m "feat(pkm): lib_frontmatter with Korean-safe NFC normalization + tests"
```

---

## Task 3: 폴더→태그 매핑 데이터

**Files:**
- Create: `scripts/pkm/folder_tag_map.json`

- [ ] **Step 1: 38개 폴더 매핑 작성**

`scripts/pkm/folder_tag_map.json`:
```json
{
  "_doc": "Folder name pattern → list of tags. Pattern is matched against the immediate parent folder of a note.",
  "patterns": [
    {"match": "1-1_*",                       "tags": ["meta/extracurricular"]},
    {"match": "1-2_coding-basics",           "tags": ["skill/python", "cs/se"]},
    {"match": "2-1_probability-statistics",  "tags": ["math/probability", "math/statistics"]},
    {"match": "2-1_data-structures",         "tags": ["cs/algorithms"]},
    {"match": "2-1_*",                       "tags": []},
    {"match": "2-2_discrete-math",           "tags": ["math/discrete"]},
    {"match": "2-2_algorithms",              "tags": ["cs/algorithms"]},
    {"match": "2-2_*",                       "tags": []},
    {"match": "3-1_machine-learning",        "tags": ["cs/ml"]},
    {"match": "3-1_ML-project",              "tags": ["cs/ml", "type/project"]},
    {"match": "3-1_distributed-computing",   "tags": ["cs/systems", "cs/distributed"]},
    {"match": "3-1_intellectual-property",   "tags": ["meta/cert"]},
    {"match": "3-1_programming-languages",   "tags": ["cs/se"]},
    {"match": "3-1_*",                       "tags": []},
    {"match": "3-2_database*",               "tags": ["cs/db"]},
    {"match": "3-2_operating-system*",       "tags": ["cs/systems"]},
    {"match": "3-2_*network*",               "tags": ["cs/systems"]},
    {"match": "3-2_*",                       "tags": []},
    {"match": "4-1_AIOSS",                   "tags": ["cs/open-source", "cs/ai", "cs/devops"]},
    {"match": "4-1_computer-vision",         "tags": ["cs/cv", "cs/dl"]},
    {"match": "4-1_natural-language*",       "tags": ["cs/nlp"]},
    {"match": "4-1_*",                       "tags": []},
    {"match": "4-2_*",                       "tags": []},
    {"match": "elective_LLM",                "tags": ["cs/llm", "cs/nlp"]},
    {"match": "elective_docker-k8s",         "tags": ["skill/docker", "cs/devops"]},
    {"match": "elective_java",               "tags": ["skill/java"]},
    {"match": "elective_*",                  "tags": []},
    {"match": "certifications/*",            "tags": ["meta/cert"]},
    {"match": "LGAimer/*",                   "tags": ["cs/ml", "meta/extracurricular"]}
  ],
  "filename_rules": [
    {"match": "README.md",                   "tags": ["type/MOC"], "frontmatter": {"type": "MOC"}},
    {"match": "*MOC*.md",                    "tags": ["type/MOC"], "frontmatter": {"type": "MOC"}}
  ],
  "subdir_rules": [
    {"match": "*/과제/*",                    "tags": ["type/project"], "frontmatter": {"type": "project"}},
    {"match": "*/실습/*",                    "tags": ["type/project"], "frontmatter": {"type": "project"}},
    {"match": "*/프로젝트/*",                 "tags": ["type/project"], "frontmatter": {"type": "project"}},
    {"match": "*/papers/*",                  "tags": ["type/literature"], "frontmatter": {"type": "literature"}},
    {"match": "*/교재/*",                    "tags": ["type/literature"], "frontmatter": {"type": "literature"}}
  ]
}
```

- [ ] **Step 2: 실제 볼트 폴더 목록과 대조하여 빠진 폴더 추가**

Run:
```bash
ls -d ComputerScience/*/ | sed 's|ComputerScience/||;s|/$||' | sort
```
검토 후 빠진 폴더가 있으면 위 매핑에 추가. `tags: []`도 명시적으로 포함하여 "분류 못 함" 상태를 기록.

- [ ] **Step 3: JSON 유효성 검증**

Run:
```bash
python3 -c "import json; json.load(open('scripts/pkm/folder_tag_map.json'))" && echo OK
```
Expected: `OK`

- [ ] **Step 4: 커밋**

```bash
git add scripts/pkm/folder_tag_map.json
git commit -m "data(pkm): folder→tag mapping for 38 course directories"
```

---

## Task 4: Phase 1A — 신규 플러그인 7종 + 활성화 4종 + Tier 2 3종

**Files:**
- Modify: `.obsidian/community-plugins.json`

- [ ] **Step 1: 현재 활성 플러그인 목록 백업**

Run:
```bash
cp .obsidian/community-plugins.json .obsidian/community-plugins.json.bak
cat .obsidian/community-plugins.json
```

- [ ] **Step 2: 신규 community-plugins.json 작성**

`.obsidian/community-plugins.json`:
```json
[
  "better-export-pdf",
  "obsidian-git",
  "terminal",
  "obsidian-linter",
  "templater-obsidian",
  "tag-wrangler",
  "auto-note-mover",
  "find-unlinked-files",
  "consistent-attachments-and-links",
  "mermaid-tools",
  "pdf-plus",
  "quick-latex",
  "obsidian-latex",
  "dataview",
  "obsidian-excalidraw-plugin",
  "obsidian-spaced-repetition",
  "table-editor-obsidian",
  "obsidian-icon-folder",
  "graph-analysis",
  "excalibrain",
  "obsidian-style-settings",
  "periodic-notes",
  "obsidian-hover-editor"
]
```

- [ ] **Step 3: 사용자에게 플러그인 다운로드 안내 메시지 출력**

(자동 다운로드는 옵시디언 GUI가 필요 — 본 플랜은 enabling 상태만 미리 등록하고, 실제 다운로드는 사용자가 옵시디언 시작 시 처리)

이 단계에서는 enabling 등록만 함. 실제 플러그인 ZIP 다운로드는 옵시디언이 처음 실행될 때 자동 처리됨 (`.obsidian/plugins/<id>/`에 zip이 없으면 옵시디언이 community plugin browser에서 시도). 안전을 위해 README에 다운로드 절차 명시.

- [ ] **Step 4: 커밋**

```bash
git add .obsidian/community-plugins.json
git rm .obsidian/community-plugins.json.bak 2>/dev/null || rm .obsidian/community-plugins.json.bak
git commit -m "feat(pkm): enable 7 new + 4 reactivated + 3 tier-2 community plugins"
```

---

## Task 5: Phase 1B — Linter 한국어 안전 룰셋

**Files:**
- Create or modify: `.obsidian/plugins/obsidian-linter/data.json`

- [ ] **Step 1: 기존 Linter 설정 확인**

Run:
```bash
ls .obsidian/plugins/obsidian-linter/
cat .obsidian/plugins/obsidian-linter/data.json 2>/dev/null | head -40
```

- [ ] **Step 2: 한국어 안전 룰셋 적용**

`.obsidian/plugins/obsidian-linter/data.json`에서 다음 키들을 설정 (기존 키는 보존, 명시 키만 덮어씀):
```json
{
  "ruleConfigs": {
    "yaml-key-sort": {"enabled": true, "yaml-key-priority-sort-order": "title\ndate\naliases\ntype\nstatus\nsemester\ncourse\ntags\ncreated\nupdated\nsource\nmoc"},
    "format-tags-in-yaml": {"enabled": true},
    "format-yaml-array": {"enabled": true, "default-array-style": "multi-line"},
    "yaml-timestamp": {"enabled": true, "date-modified-key": "updated", "date-created-key": "created", "force-retention-of-create-value": true},
    "trailing-spaces": {"enabled": true},
    "consecutive-blank-lines": {"enabled": true},
    "heading-blank-lines": {"enabled": true},
    "empty-line-around-blockquotes": {"enabled": true},
    "empty-line-around-code-fences": {"enabled": true},
    "space-after-list-markers": {"enabled": true},
    "capitalize-headings": {"enabled": false},
    "headings-start-line": {"enabled": false},
    "english-spelling": {"enabled": false},
    "punctuation-conversion": {"enabled": false},
    "quote-style": {"enabled": false},
    "paragraph-blank-lines": {"enabled": false}
  }
}
```

(실제 키 이름은 `obsidian-linter` 버전에 따라 다를 수 있음. Phase 10에서 dry-run으로 한글 안전성 확인.)

Run (Python으로 안전 병합):
```bash
python3 - <<'PY'
import json, pathlib
p = pathlib.Path(".obsidian/plugins/obsidian-linter/data.json")
existing = {}
if p.exists():
    existing = json.loads(p.read_text(encoding="utf-8"))
patch = {
  "ruleConfigs": {
    "yaml-key-sort": {"enabled": True, "yaml-key-priority-sort-order": "title\ndate\naliases\ntype\nstatus\nsemester\ncourse\ntags\ncreated\nupdated\nsource\nmoc"},
    "format-tags-in-yaml": {"enabled": True},
    "format-yaml-array": {"enabled": True, "default-array-style": "multi-line"},
    "yaml-timestamp": {"enabled": True, "date-modified-key": "updated", "date-created-key": "created", "force-retention-of-create-value": True},
    "trailing-spaces": {"enabled": True},
    "consecutive-blank-lines": {"enabled": True},
    "heading-blank-lines": {"enabled": True},
    "empty-line-around-blockquotes": {"enabled": True},
    "empty-line-around-code-fences": {"enabled": True},
    "space-after-list-markers": {"enabled": True},
    "capitalize-headings": {"enabled": False},
    "headings-start-line": {"enabled": False},
    "english-spelling": {"enabled": False},
    "punctuation-conversion": {"enabled": False},
    "quote-style": {"enabled": False},
    "paragraph-blank-lines": {"enabled": False}
  }
}
existing.setdefault("ruleConfigs", {}).update(patch["ruleConfigs"])
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
print("Linter rules merged.")
PY
```

- [ ] **Step 3: JSON 유효성 검증**

Run:
```bash
python3 -c "import json; json.load(open('.obsidian/plugins/obsidian-linter/data.json'))" && echo OK
```

- [ ] **Step 4: 커밋**

```bash
git add .obsidian/plugins/obsidian-linter/data.json
git commit -m "feat(pkm): Korean-safe Linter ruleset (preserve quotes, skip capitalize)"
```

---

## Task 6: Phase 1C — 그래프 뷰 색상 그룹

**Files:**
- Modify: `.obsidian/graph.json`

- [ ] **Step 1: 기존 graph.json 백업 후 colorGroups 병합**

Run:
```bash
python3 - <<'PY'
import json, pathlib
p = pathlib.Path(".obsidian/graph.json")
existing = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

color_groups = [
    {"query": "tag:#type/MOC",                              "color": {"a": 1, "rgb": 15225154}},
    {"query": "tag:#cs/ml OR tag:#cs/dl",                   "color": {"a": 1, "rgb": 3447003}},
    {"query": "tag:#cs/systems OR tag:#cs/devops",          "color": {"a": 1, "rgb": 10181046}},
    {"query": "tag:#cs/algorithms",                          "color": {"a": 1, "rgb": 15967746}},
    {"query": "tag:#cs/ai OR tag:#cs/llm OR tag:#cs/nlp",   "color": {"a": 1, "rgb": 1751474}},
    {"query": "tag:#math",                                   "color": {"a": 1, "rgb": 3066993}},
    {"query": "tag:#skill",                                  "color": {"a": 1, "rgb": 9807270}},
    {"query": "tag:#meta/portfolio",                         "color": {"a": 1, "rgb": 15844367}},
    {"query": "tag:#meta/question",                          "color": {"a": 1, "rgb": 15105570}}
]

existing["colorGroups"] = color_groups
existing.setdefault("collapse-filter", True)
existing.setdefault("search", "-path:image -path:scripts -path:docs")
existing.setdefault("showTags", True)
existing.setdefault("showAttachments", False)
existing.setdefault("hideUnresolved", False)
existing.setdefault("showOrphans", True)

p.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
print("graph.json updated.")
PY
```

- [ ] **Step 2: 검증**

Run:
```bash
python3 -c "import json; d=json.load(open('.obsidian/graph.json')); assert len(d['colorGroups'])==9; print('OK', len(d['colorGroups']), 'groups')"
```

- [ ] **Step 3: 커밋**

```bash
git add .obsidian/graph.json
git commit -m "feat(pkm): graph view color groups (9 tag-based clusters)"
```

---

## Task 7: Phase 1D — CSS 스니펫 4종

**Files:**
- Create: `.obsidian/snippets/fonts.css`
- Create: `.obsidian/snippets/callouts.css`
- Create: `.obsidian/snippets/status-badges.css`
- Create: `.obsidian/snippets/moc-styling.css`

- [ ] **Step 1: fonts.css**

`.obsidian/snippets/fonts.css`:
```css
/* Korean + Code font pairing for academic CS notes */
.theme-light, .theme-dark {
  --font-text-theme: "Pretendard", "Pretendard Variable", -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", sans-serif;
  --font-monospace-theme: "JetBrains Mono", "D2Coding", "Menlo", monospace;
}

.markdown-source-view, .markdown-preview-view {
  font-family: var(--font-text-theme);
  line-height: 1.7;
}

code, pre, .HyperMD-codeblock {
  font-family: var(--font-monospace-theme) !important;
  font-feature-settings: "calt" 1, "liga" 1;
}
```

- [ ] **Step 2: callouts.css**

`.obsidian/snippets/callouts.css`:
```css
/* Type-based callout colors aligned with frontmatter `type` field */
.callout[data-callout="lecture"] {
  --callout-color: 52, 152, 219;
  --callout-icon: lucide-graduation-cap;
}
.callout[data-callout="literature"] {
  --callout-color: 155, 89, 182;
  --callout-icon: lucide-book-open;
}
.callout[data-callout="permanent"] {
  --callout-color: 46, 204, 113;
  --callout-icon: lucide-evergreen;
}
.callout[data-callout="project"] {
  --callout-color: 230, 126, 34;
  --callout-icon: lucide-hammer;
}
.callout[data-callout="moc"] {
  --callout-color: 231, 76, 60;
  --callout-icon: lucide-map;
}
.callout[data-callout="question"] {
  --callout-color: 241, 196, 15;
  --callout-icon: lucide-help-circle;
}
```

- [ ] **Step 3: status-badges.css**

`.obsidian/snippets/status-badges.css`:
```css
/* Visual status indicator on note headers, driven by frontmatter `status` */
.metadata-property[data-property-key="status"] .metadata-property-value {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.85em;
  font-weight: 600;
}

.metadata-property[data-property-key="status"][data-property-value="seedling"] .metadata-property-value {
  background: rgba(149, 165, 166, 0.2);
  color: #7f8c8d;
}
.metadata-property[data-property-key="status"][data-property-value="budding"] .metadata-property-value {
  background: rgba(241, 196, 15, 0.2);
  color: #d4ac0d;
}
.metadata-property[data-property-key="status"][data-property-value="evergreen"] .metadata-property-value {
  background: rgba(46, 204, 113, 0.2);
  color: #27ae60;
}
```

- [ ] **Step 4: moc-styling.css**

`.obsidian/snippets/moc-styling.css`:
```css
/* MOC notes get a left accent bar in reading view */
.markdown-preview-view:has([data-property-value="MOC"]) > .markdown-preview-sizer {
  border-left: 3px solid #e74c3c;
  padding-left: 1.2em;
}

/* MOC h2 sections get extra spacing */
.markdown-preview-view:has([data-property-value="MOC"]) h2 {
  margin-top: 2em;
  border-bottom: 1px solid var(--background-modifier-border);
  padding-bottom: 0.3em;
}
```

- [ ] **Step 5: appearance.json에 스니펫 활성화**

Run:
```bash
python3 - <<'PY'
import json, pathlib
p = pathlib.Path(".obsidian/appearance.json")
existing = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
existing["enabledCssSnippets"] = sorted(set(existing.get("enabledCssSnippets", []) + [
    "fonts", "callouts", "status-badges", "moc-styling"
]))
p.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
print("Snippets enabled:", existing["enabledCssSnippets"])
PY
```

- [ ] **Step 6: 커밋**

```bash
git add .obsidian/snippets/ .obsidian/appearance.json
git commit -m "style(pkm): 4 CSS snippets (fonts, callouts, status-badges, moc-styling)"
```

---

## Task 8: Phase 1E — Templater 템플릿 6종

**Files:**
- Create: `_templates/lecture-note.md`
- Create: `_templates/literature-note.md`
- Create: `_templates/permanent-note.md`
- Create: `_templates/project-note.md`
- Create: `_templates/moc.md`
- Create: `_templates/daily-note.md`

- [ ] **Step 1: lecture-note.md**

`_templates/lecture-note.md`:
```markdown
---
aliases: []
course: <% tp.file.folder().split("_").slice(1).join("-") %>
created: <% tp.date.now("YYYY-MM-DD") %>
date: <% tp.date.now("YYYY-MM-DD") %>
semester: "<% tp.file.folder().match(/^[0-9]-[0-9]/) ? tp.file.folder().match(/^[0-9]-[0-9]/)[0] : (tp.file.folder().startsWith("elective") ? "elective" : "extracurricular") %>"
source: ""
status: seedling
tags:
  - type/lecture
title: <% tp.file.title %>
type: lecture
updated: <% tp.date.now("YYYY-MM-DD") %>
---

# <% tp.file.title %>

## 핵심 개념

## 정리

## 질문 / TODO
- 

## 관련
- 
```

- [ ] **Step 2: literature-note.md**

`_templates/literature-note.md`:
```markdown
---
aliases: []
course: <% tp.file.folder().split("_").slice(1).join("-") %>
created: <% tp.date.now("YYYY-MM-DD") %>
date: <% tp.date.now("YYYY-MM-DD") %>
semester: "<% tp.file.folder().match(/^[0-9]-[0-9]/) ? tp.file.folder().match(/^[0-9]-[0-9]/)[0] : "elective" %>"
source: "<% await tp.system.prompt('Source (저자, 제목, URL)') %>"
status: seedling
tags:
  - type/literature
title: <% tp.file.title %>
type: literature
updated: <% tp.date.now("YYYY-MM-DD") %>
---

# <% tp.file.title %>

> Source: 

## 요약

## 핵심 인용

## 내 생각

## 연결
- 
```

- [ ] **Step 3: permanent-note.md**

`_templates/permanent-note.md`:
```markdown
---
aliases: []
course: cross-curriculum
created: <% tp.date.now("YYYY-MM-DD") %>
date: <% tp.date.now("YYYY-MM-DD") %>
semester: "all"
source: ""
status: budding
tags:
  - type/permanent
title: <% tp.file.title %>
type: permanent
updated: <% tp.date.now("YYYY-MM-DD") %>
---

# <% tp.file.title %>

<!-- 한 문장으로 요약하세요. 이 노트의 제목이 곧 이 한 문장이어야 합니다. -->

## 본문

## 근거

## 반례 / 한계

## 연결
- 
```

- [ ] **Step 4: project-note.md**

`_templates/project-note.md`:
```markdown
---
aliases: []
course: <% tp.file.folder().split("_").slice(1).join("-") %>
created: <% tp.date.now("YYYY-MM-DD") %>
date: <% tp.date.now("YYYY-MM-DD") %>
semester: "<% tp.file.folder().match(/^[0-9]-[0-9]/) ? tp.file.folder().match(/^[0-9]-[0-9]/)[0] : "elective" %>"
source: ""
status: seedling
tags:
  - type/project
title: <% tp.file.title %>
type: project
updated: <% tp.date.now("YYYY-MM-DD") %>
---

# <% tp.file.title %>

## 목표

## 요구사항

## 진행

## 결과

## 회고
```

- [ ] **Step 5: moc.md**

`_templates/moc.md`:
```markdown
---
aliases: []
course: cross-curriculum
created: <% tp.date.now("YYYY-MM-DD") %>
date: <% tp.date.now("YYYY-MM-DD") %>
semester: "all"
source: ""
status: evergreen
tags:
  - type/MOC
title: <% tp.file.title %>
type: MOC
updated: <% tp.date.now("YYYY-MM-DD") %>
---

up:: [[Home MOC]]
central:: [[<% tp.file.title %>]]

# <% tp.file.title %>

## Foundations

## Core Topics

## Open Questions

## All notes (auto)
\`\`\`dataview
TABLE status, file.mtime as updated
FROM "<% tp.file.folder() %>"
WHERE type != "MOC"
SORT file.mtime DESC
LIMIT 50
\`\`\`
```

- [ ] **Step 6: daily-note.md**

`_templates/daily-note.md`:
```markdown
---
aliases: []
created: <% tp.date.now("YYYY-MM-DD") %>
date: <% tp.date.now("YYYY-MM-DD") %>
status: seedling
tags:
  - type/index
title: <% tp.date.now("YYYY-MM-DD") %>
type: index
updated: <% tp.date.now("YYYY-MM-DD") %>
---

# <% tp.date.now("YYYY-MM-DD dddd") %>

## 오늘 해야 할 일
- [ ]

## 강의
- 

## 캡처

## 시험 큐
\`\`\`dataview
LIST FROM #meta/exam
SORT file.mtime DESC
LIMIT 10
\`\`\`
```

- [ ] **Step 7: Templater 설정 갱신 (template folder 등록)**

Run:
```bash
python3 - <<'PY'
import json, pathlib
p = pathlib.Path(".obsidian/plugins/templater-obsidian/data.json")
existing = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
existing["templates_folder"] = "_templates"
existing["enable_folder_templates"] = False
existing["trigger_on_file_creation"] = False
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
print("Templater configured.")
PY
```

- [ ] **Step 8: 커밋**

```bash
git add _templates/ .obsidian/plugins/templater-obsidian/data.json
git commit -m "feat(pkm): 6 Templater templates (lecture/literature/permanent/project/MOC/daily)"
```

---

## Task 9: Phase 1F — 루트 잡파일 6개 정리

**Files:**
- Delete: `Pasted image 20251202160528.png`
- Delete: `Pasted image 20260323153729.png`
- Delete: `스크린샷 2025-05-07 091243.png`
- Delete: `스크린샷 2025-12-08 17.37.28.png`
- Delete: `발표자료 image.zip`
- Delete: `...md` (malformed)
- Move: `커리큘럼 관계 그래프.canvas` → 보존 (벨류 있음)

- [ ] **Step 1: ZIP 내용 검토 (소중한 자료가 들어있을 가능성)**

Run:
```bash
unzip -l "발표자료 image.zip" | head -20
```
검토 후 필요하면 image/presentation/ 등으로 이동, 아니면 삭제.

- [ ] **Step 2: 삭제**

Run:
```bash
git rm "Pasted image 20251202160528.png" \
       "Pasted image 20260323153729.png" \
       "스크린샷 2025-05-07 091243.png" \
       "스크린샷 2025-12-08 17.37.28.png" \
       "발표자료 image.zip" \
       "...md" 2>&1 | tail -10
```

(파일이 git에 트래킹되지 않은 경우 `git rm` 대신 `rm`. 둘 다 시도)

- [ ] **Step 3: 검증**

Run:
```bash
ls *.png *.zip 2>/dev/null | head; echo "---"; ls -la | head -20
```
Expected: 루트에 PNG/ZIP 없음

- [ ] **Step 4: 커밋**

```bash
git add -A
git commit -m "chore(pkm): remove 6 stray root files (4 PNGs, 1 ZIP, 1 malformed .md)"
```

---

## Task 10: Phase 2 — MOCs/ 디렉터리 + 16개 MOC 작성

**Files:**
- Create: `MOCs/Home MOC.md`
- Create: `MOCs/Machine Learning MOC.md`
- Create: `MOCs/Deep Learning MOC.md`
- Create: `MOCs/Algorithms MOC.md`
- Create: `MOCs/Systems MOC.md`
- Create: `MOCs/Computer Vision MOC.md`
- Create: `MOCs/LLM & NLP MOC.md`
- Create: `MOCs/AI Open Source MOC.md`
- Create: `MOCs/Math Foundations MOC.md`
- Create: `MOCs/Database MOC.md`
- Create: `MOCs/Cloud & Containers MOC.md`
- Create: `MOCs/Security MOC.md`
- Create: `MOCs/Software Engineering MOC.md`
- Create: `MOCs/Certifications MOC.md`
- Create: `MOCs/Portfolio MOC.md`
- Create: `MOCs/Open Questions MOC.md`

- [ ] **Step 1: 디렉터리 생성**

Run:
```bash
mkdir -p MOCs
```

- [ ] **Step 2~17: 각 MOC 작성**

각 MOC는 다음 구조 사용 (도메인별 첫 섹션·태그 쿼리만 다름):

`MOCs/Home MOC.md`:
```markdown
---
aliases: [홈, 지식 지도]
course: cross-curriculum
created: 2026-05-05
date: 2026-05-05
semester: "all"
source: ""
status: evergreen
tags:
  - type/MOC
  - type/index
title: Home MOC
type: MOC
updated: 2026-05-05
---

central:: [[Home MOC]]
children:: [[Machine Learning MOC]], [[Deep Learning MOC]], [[Algorithms MOC]], [[Systems MOC]], [[Computer Vision MOC]], [[LLM & NLP MOC]], [[AI Open Source MOC]], [[Math Foundations MOC]], [[Database MOC]], [[Cloud & Containers MOC]], [[Security MOC]], [[Software Engineering MOC]], [[Certifications MOC]], [[Portfolio MOC]], [[Open Questions MOC]]

# Home MOC

> Single entry point. 모든 도메인 MOC가 여기서 출발한다.

## CS Core
- [[Machine Learning MOC]]
- [[Deep Learning MOC]]
- [[Algorithms MOC]]
- [[Systems MOC]]
- [[Computer Vision MOC]]
- [[LLM & NLP MOC]]
- [[AI Open Source MOC]]
- [[Database MOC]]
- [[Cloud & Containers MOC]]
- [[Security MOC]]
- [[Software Engineering MOC]]

## Foundations
- [[Math Foundations MOC]]

## Outputs
- [[Portfolio MOC]]
- [[Certifications MOC]]
- [[Open Questions MOC]]

## All MOCs (auto)
\`\`\`dataview
LIST FROM #type/MOC
SORT file.name ASC
\`\`\`

## Recently updated
\`\`\`dataview
TABLE WITHOUT ID file.link as Note, type, status, file.mtime as updated
FROM "" WHERE file.path != this.file.path
SORT file.mtime DESC
LIMIT 15
\`\`\`
```

`MOCs/Machine Learning MOC.md`:
```markdown
---
aliases: [ML MOC]
course: cross-curriculum
created: 2026-05-05
date: 2026-05-05
semester: "all"
source: ""
status: evergreen
tags:
  - type/MOC
  - cs/ml
title: Machine Learning MOC
type: MOC
updated: 2026-05-05
---

up:: [[Home MOC]]
central:: [[Machine Learning MOC]]
children:: [[Deep Learning MOC]], [[Computer Vision MOC]]

# Machine Learning MOC

## Foundations
- 

## Supervised Learning
- 

## Unsupervised Learning
- 

## Evaluation & Generalization
- 

## Open Questions
- 

## All ML notes (auto)
\`\`\`dataview
TABLE status, file.mtime as updated
FROM #cs/ml
WHERE type != "MOC"
SORT file.mtime DESC
LIMIT 50
\`\`\`
```

(나머지 14개 MOC도 동일 패턴 — `tags` 도메인 태그만 다르고, dataview FROM 절도 일치하게 변경)

`MOCs/Deep Learning MOC.md` — `tags: [type/MOC, cs/dl]`, `FROM #cs/dl`, up:: ML MOC
`MOCs/Algorithms MOC.md` — `tags: [type/MOC, cs/algorithms]`, `FROM #cs/algorithms`
`MOCs/Systems MOC.md` — `tags: [type/MOC, cs/systems]`, `FROM #cs/systems OR #cs/distributed OR #cs/devops`
`MOCs/Computer Vision MOC.md` — `tags: [type/MOC, cs/cv]`, `FROM #cs/cv`, up:: ML MOC
`MOCs/LLM & NLP MOC.md` — `tags: [type/MOC, cs/llm, cs/nlp]`, `FROM #cs/llm OR #cs/nlp`, up:: ML MOC
`MOCs/AI Open Source MOC.md` — `tags: [type/MOC, cs/open-source, cs/ai]`, `FROM #cs/open-source`
`MOCs/Math Foundations MOC.md` — `tags: [type/MOC]`, `FROM #math/linalg OR #math/calculus OR #math/probability OR #math/statistics OR #math/discrete`
`MOCs/Database MOC.md` — `tags: [type/MOC, cs/db]`, `FROM #cs/db`
`MOCs/Cloud & Containers MOC.md` — `tags: [type/MOC, cs/devops, skill/docker]`, `FROM #skill/docker OR #cs/devops`
`MOCs/Security MOC.md` — `tags: [type/MOC, cs/security]`, `FROM #cs/security`
`MOCs/Software Engineering MOC.md` — `tags: [type/MOC, cs/se]`, `FROM #cs/se`
`MOCs/Certifications MOC.md` — `tags: [type/MOC, meta/cert]`, `FROM #meta/cert`
`MOCs/Portfolio MOC.md` — `tags: [type/MOC, meta/portfolio]`, `FROM #meta/portfolio OR #type/permanent`
`MOCs/Open Questions MOC.md` — `tags: [type/MOC, meta/question]`, `FROM #meta/question`

- [ ] **Step 18: 검증**

Run:
```bash
ls MOCs/ | wc -l
```
Expected: 16

```bash
for f in MOCs/*.md; do
  python3 -c "import frontmatter; m=frontmatter.load('$f'); assert m['type']=='MOC'; print('OK', '$f')"
done
```

- [ ] **Step 19: 커밋**

```bash
git add MOCs/
git commit -m "feat(pkm): 16 MOCs scaffolded (Home + 15 domain MOCs with Dataview blocks)"
```

---

## Task 11: Phase 3 — frontmatter 미보유 44개 노트에 주입

**Files:**
- Create: `scripts/pkm/inject_frontmatter.py`
- Modify: ~44 notes (no frontmatter)

- [ ] **Step 1: inject_frontmatter.py 작성**

`scripts/pkm/inject_frontmatter.py`:
```python
#!/usr/bin/env python3
"""Phase 3+4 frontmatter migration.

- Phase 3: notes without frontmatter → inject full schema
- Phase 4: notes with existing frontmatter → merge new 7 fields
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import iter_vault_notes, merge_frontmatter, read_note, write_note

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
        return "MOC"
    if "MOC" in name:
        return "MOC"
    parts_lower = [p.lower() for p in rel.parts]
    if any(seg in {"과제", "프로젝트", "실습"} for seg in rel.parts):
        return "project"
    if any(seg in {"papers", "reading", "교재"} for seg in rel.parts):
        return "literature"
    if rel.parts[0] == "MOCs":
        return "MOC"
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
    if not fm:
        # Phase 3 — full injection
        # Title from H1 if present, else filename stem
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
            return True, f"PHASE3 inject {path.relative_to(VAULT)}"
        write_note(path, full, body)
        return True, f"PHASE3 wrote {path.relative_to(VAULT)}"
    else:
        # Phase 4 — merge
        # Set type from existing if present in tags
        existing_type = fm.get("type")
        if not existing_type:
            for tag in (fm.get("tags") or []):
                if isinstance(tag, str) and tag.startswith("type/"):
                    existing_type = tag.split("/", 1)[1]
                    break
        if existing_type:
            new_fields["type"] = existing_type
        # Preserve existing status if any
        if fm.get("status"):
            new_fields["status"] = fm["status"]
        # Use existing date for created if present
        if fm.get("date") and not fm.get("created"):
            new_fields["created"] = str(fm["date"])
        merged = merge_frontmatter(fm, new_fields, protected=("title", "date", "aliases", "tags"))
        # Always add type/<type> to tags (additive)
        existing_tags = set(merged.get("tags") or [])
        existing_tags.add(f"type/{new_fields['type']}")
        merged["tags"] = sorted(existing_tags)
        if dry_run:
            return True, f"PHASE4 merge {path.relative_to(VAULT)} → +{set(merged) - set(fm)}"
        write_note(path, merged, body)
        return True, f"PHASE4 wrote {path.relative_to(VAULT)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--phase", choices=["3", "4", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    notes = list(iter_vault_notes(VAULT, exclude=(".obsidian", ".git", "scripts", "docs", "_templates")))
    n_phase3 = n_phase4 = 0
    for p in notes:
        fm, _ = read_note(p)
        if not fm and args.phase in ("3", "both"):
            ok, msg = process(p, args.dry_run)
            if ok:
                n_phase3 += 1
                print(msg)
        elif fm and args.phase in ("4", "both"):
            # Skip if already has all 7 new fields
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


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Phase 3 dry-run (5건만)**

Run:
```bash
python3 scripts/pkm/inject_frontmatter.py --dry-run --phase 3 --limit 5
```
Expected: 5건의 PHASE3 inject 메시지 출력

- [ ] **Step 3: Phase 3 실제 실행 (전체)**

Run:
```bash
python3 scripts/pkm/inject_frontmatter.py --phase 3
```
Expected: ~44건 PHASE3 wrote

- [ ] **Step 4: 검증**

Run:
```bash
python3 - <<'PY'
import frontmatter, pathlib
vault = pathlib.Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
no_fm = []
for p in vault.rglob("*.md"):
    if any(part in p.parts for part in (".obsidian", ".git", "scripts", "docs", "_templates")):
        continue
    fm = frontmatter.load(p).metadata
    if not fm:
        no_fm.append(p)
print(f"Notes still without frontmatter: {len(no_fm)}")
for p in no_fm[:10]:
    print(" ", p)
PY
```
Expected: 0 (또는 매우 적음 — 그 경우 개별 검토)

- [ ] **Step 5: 커밋**

```bash
git add -A
git commit -m "feat(pkm): inject frontmatter into ~44 previously bare notes (Phase 3)"
```

---

## Task 12: Phase 4 — 기존 435개 노트에 신규 7필드 추가

**Files:**
- Modify: ~435 notes

- [ ] **Step 1: dry-run (10건만)**

Run:
```bash
python3 scripts/pkm/inject_frontmatter.py --dry-run --phase 4 --limit 10
```
출력 검토 — 기존 필드 보존 여부 확인.

- [ ] **Step 2: 단일 파일 실측**

Run:
```bash
# 임의 노트 1개 미리보기
python3 scripts/pkm/inject_frontmatter.py --dry-run --phase 4 --limit 1
```
이후 git diff 미리 보기 위해 실제 1건만 적용:
```bash
python3 scripts/pkm/inject_frontmatter.py --phase 4 --limit 1
git diff --stat
git diff | head -50
```
검토 후 OK 시 다음 단계.

- [ ] **Step 3: 전체 실행**

Run:
```bash
git checkout -- .  # 1건 실측 되돌리고 일괄 적용으로 통일
python3 scripts/pkm/inject_frontmatter.py --phase 4
```
Expected: ~435건 처리

- [ ] **Step 4: 검증**

Run:
```bash
python3 - <<'PY'
import frontmatter, pathlib
vault = pathlib.Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
required = {"type", "status", "semester", "course", "created", "updated", "source", "title", "tags"}
missing = []
for p in vault.rglob("*.md"):
    if any(part in p.parts for part in (".obsidian", ".git", "scripts", "docs", "_templates")):
        continue
    fm = frontmatter.load(p).metadata
    miss = required - set(fm.keys())
    if miss:
        missing.append((p, miss))
print(f"Notes missing required fields: {len(missing)}")
for p, miss in missing[:20]:
    print(" ", p.relative_to(vault), "->", miss)
PY
```
Expected: 0 missing

- [ ] **Step 5: 5% 샘플 수동 검증**

Run:
```bash
python3 - <<'PY'
import frontmatter, pathlib, random
vault = pathlib.Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
notes = [p for p in vault.rglob("*.md") if not any(x in p.parts for x in (".obsidian", ".git", "scripts", "docs", "_templates"))]
sample = random.sample(notes, k=min(20, len(notes)))
for p in sample:
    fm = frontmatter.load(p).metadata
    print(p.relative_to(vault))
    print(f"  type={fm.get('type')} status={fm.get('status')} semester={fm.get('semester')} course={fm.get('course')}")
    print(f"  title={fm.get('title')}")
    print()
PY
```
검토: 폴더와 일치하는 semester/course가 추론되었는지.

- [ ] **Step 6: 커밋**

```bash
git add -A
git commit -m "feat(pkm): extend frontmatter on ~435 existing notes with 7 new fields (Phase 4)"
```

---

## Task 13: Phase 5 — 폴더 → 도메인 태그 자동 주입

**Files:**
- Create: `scripts/pkm/inject_tags.py`
- Modify: ~479 notes

- [ ] **Step 1: inject_tags.py 작성**

`scripts/pkm/inject_tags.py`:
```python
#!/usr/bin/env python3
"""Phase 5: inject domain tags from folder→tag map."""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import iter_vault_notes, read_note, write_note

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
MAP_PATH = Path(__file__).resolve().parent / "folder_tag_map.json"


def load_map():
    return json.loads(MAP_PATH.read_text(encoding="utf-8"))


def tags_for_path(rel: Path, m: dict) -> list[str]:
    tags = set()
    rel_str = str(rel).replace("\\", "/")

    # filename rules
    for rule in m.get("filename_rules", []):
        if fnmatch.fnmatch(rel.name, rule["match"]):
            tags.update(rule.get("tags", []))

    # subdir rules (full path match)
    for rule in m.get("subdir_rules", []):
        if fnmatch.fnmatch(rel_str, rule["match"]):
            tags.update(rule.get("tags", []))

    # folder pattern rules — only top folder under ComputerScience or top-level dirs
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "ComputerScience":
        folder = parts[1]
        for rule in m.get("patterns", []):
            if fnmatch.fnmatch(folder, rule["match"]):
                tags.update(rule.get("tags", []))
                break  # first match wins for folder
    elif parts[0] in {"certifications", "LGAimer"}:
        for rule in m.get("patterns", []):
            if fnmatch.fnmatch(rel_str, rule["match"]):
                tags.update(rule.get("tags", []))
                break

    return sorted(tags)


def process(path: Path, m: dict, dry_run: bool) -> tuple[int, str]:
    rel = path.relative_to(VAULT)
    new_tags = tags_for_path(rel, m)
    if not new_tags:
        return 0, ""
    fm, body = read_note(path)
    existing = set(fm.get("tags") or [])
    union = sorted(existing | set(new_tags))
    if union == sorted(existing):
        return 0, ""
    fm["tags"] = union
    if not dry_run:
        write_note(path, fm, body)
    added = sorted(set(new_tags) - existing)
    return len(added), f"{rel} +{added}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    m = load_map()
    total_added = total_files = 0
    for p in iter_vault_notes(VAULT, exclude=(".obsidian", ".git", "scripts", "docs", "_templates", "MOCs")):
        added, msg = process(p, m, args.dry_run)
        if added:
            total_added += added
            total_files += 1
            print(msg)
        if args.limit and total_files >= args.limit:
            break

    print(f"\nSummary: files={total_files}, tags_added={total_added}, dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run 검증 (20건)**

Run:
```bash
python3 scripts/pkm/inject_tags.py --dry-run --limit 20
```
출력 검토 — 폴더에 맞는 태그가 정확히 매핑되는지.

- [ ] **Step 3: 전체 실행**

Run:
```bash
python3 scripts/pkm/inject_tags.py
```

- [ ] **Step 4: 태그 분포 확인**

Run:
```bash
python3 - <<'PY'
import frontmatter, pathlib, collections
vault = pathlib.Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
counter = collections.Counter()
for p in vault.rglob("*.md"):
    if any(x in p.parts for x in (".obsidian", ".git", "scripts", "docs", "_templates")):
        continue
    fm = frontmatter.load(p).metadata
    for t in (fm.get("tags") or []):
        counter[t] += 1
for t, n in sorted(counter.items(), key=lambda x: -x[1])[:30]:
    print(f"{n:5}  {t}")
PY
```
검토: cs/ml, cs/algorithms, type/lecture 등이 주요 태그로 등장해야 함.

- [ ] **Step 5: 커밋**

```bash
git add -A
git commit -m "feat(pkm): inject domain tags from folder mapping (Phase 5, ~479 files)"
```

---

## Task 14: Phase 6 — 산재 이미지 50개 → /image/ 통합 + 위키링크 갱신

**Files:**
- Create: `scripts/pkm/consolidate_images.py`
- Move: ~50 image files
- Modify: notes referencing those images

- [ ] **Step 1: consolidate_images.py 작성**

`scripts/pkm/consolidate_images.py`:
```python
#!/usr/bin/env python3
"""Phase 6: consolidate scattered images into /image/ with course-prefix.

Updates wikilinks ![[xxx.png]] and standard ![](path/xxx.png) embeds.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import iter_vault_notes

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
IMAGE_DIR = VAULT / "image"
EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


def find_scattered_images() -> list[Path]:
    out = []
    for p in VAULT.rglob("*"):
        if p.suffix.lower() not in EXTS:
            continue
        if any(part in p.parts for part in (".obsidian", ".git", "scripts", "docs")):
            continue
        if p.parent == IMAGE_DIR:
            continue
        # Files inside any */images/ subdir or top-level loose
        out.append(p)
    return sorted(out)


def derive_new_name(p: Path) -> str:
    rel = p.relative_to(VAULT)
    # Try to use the course folder prefix
    parts = rel.parts
    course = "misc"
    if parts[0] == "ComputerScience" and len(parts) >= 2:
        course = parts[1]
    return f"{course}__{p.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    IMAGE_DIR.mkdir(exist_ok=True)
    moves: list[tuple[Path, Path, str, str]] = []  # (old, new, old_name, new_name)
    for p in find_scattered_images():
        new_name = derive_new_name(p)
        new_path = IMAGE_DIR / new_name
        if new_path.exists() and new_path.resolve() != p.resolve():
            # collision: append parent folder
            new_name = f"{p.parent.name}__{new_name}"
            new_path = IMAGE_DIR / new_name
        moves.append((p, new_path, p.name, new_name))

    print(f"Found {len(moves)} scattered images to move.")
    for old, new, old_name, new_name in moves[:10]:
        print(f"  {old.relative_to(VAULT)} -> image/{new_name}")
    if len(moves) > 10:
        print(f"  ... and {len(moves) - 10} more")

    if args.dry_run:
        return

    # Move files
    for old, new, _, _ in moves:
        if old.resolve() != new.resolve():
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(new))

    # Update wikilinks in all notes
    name_map = {old_name: new_name for _, _, old_name, new_name in moves}
    pattern_wiki = re.compile(r"!\[\[([^\]|#]+?)(\.(?:png|jpg|jpeg|gif|webp|svg))(\|[^\]]*)?\]\]", re.IGNORECASE)
    pattern_md = re.compile(r"!\[([^\]]*)\]\(([^)]+?\.(?:png|jpg|jpeg|gif|webp|svg))\)", re.IGNORECASE)

    updated_notes = 0
    for note in iter_vault_notes(VAULT, exclude=(".obsidian", ".git", "scripts", "docs", "_templates")):
        text = note.read_text(encoding="utf-8")
        original = text

        def fix_wiki(m):
            full = m.group(1) + m.group(2)
            base = Path(full).name
            if base in name_map:
                return f"![[{name_map[base]}{m.group(3) or ''}]]"
            return m.group(0)

        text = pattern_wiki.sub(fix_wiki, text)

        def fix_md(m):
            base = Path(m.group(2)).name
            if base in name_map:
                return f"![{m.group(1)}](image/{name_map[base]})"
            return m.group(0)

        text = pattern_md.sub(fix_md, text)

        if text != original:
            note.write_text(text, encoding="utf-8")
            updated_notes += 1

    # Remove now-empty images/ subdirs
    for d in VAULT.rglob("images"):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()

    print(f"\nMoved {len(moves)} images, updated {updated_notes} notes.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run**

Run:
```bash
python3 scripts/pkm/consolidate_images.py --dry-run
```
검토 — 충돌 없는지 확인.

- [ ] **Step 3: 실제 실행**

Run:
```bash
python3 scripts/pkm/consolidate_images.py
```

- [ ] **Step 4: 검증 — 이미지가 모두 /image/ 안에 있는지**

Run:
```bash
find . -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.gif" -o -name "*.webp" -o -name "*.svg" \) -not -path "./.obsidian/*" -not -path "./.git/*" -not -path "./image/*" -not -path "./docs/*" -not -path "./scripts/*" 2>/dev/null
```
Expected: 빈 출력

- [ ] **Step 5: 위키링크 무결성 — `images/` 참조 잔존 확인**

Run:
```bash
grep -rn "images/" --include="*.md" -l . 2>/dev/null | head
```
Expected: 매우 적거나 0 (있으면 수동 수정)

- [ ] **Step 6: 커밋**

```bash
git add -A
git commit -m "refactor(pkm): consolidate ~50 scattered images into /image/ with course prefix"
```

---

## Task 15: Phase 7 — 파일명 충돌 해결

**Files:**
- Create: `scripts/pkm/resolve_filename_collisions.py`
- Rename: ~7-20 notes

- [ ] **Step 1: resolve_filename_collisions.py 작성**

`scripts/pkm/resolve_filename_collisions.py`:
```python
#!/usr/bin/env python3
"""Phase 7: rename notes with name collisions (excluding README.md which becomes MOC)."""
from __future__ import annotations

import argparse
import collections
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import iter_vault_notes

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
PRESERVE = {"README.md"}  # handled separately in Phase 8 (README → MOC)


def find_collisions() -> dict[str, list[Path]]:
    by_name: dict[str, list[Path]] = collections.defaultdict(list)
    for p in iter_vault_notes(VAULT, exclude=(".obsidian", ".git", "scripts", "docs", "_templates", "MOCs")):
        by_name[p.name].append(p)
    return {name: paths for name, paths in by_name.items() if len(paths) > 1 and name not in PRESERVE}


def derive_new_name(p: Path) -> str:
    rel = p.relative_to(VAULT)
    parts = rel.parts
    course = ""
    if parts[0] == "ComputerScience" and len(parts) >= 2:
        course = parts[1]
        course = re.sub(r"^\d-\d_", "", course)
        course = re.sub(r"^elective_", "", course)
    if not course:
        course = parts[0].lower()
    return f"{course}__{p.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    collisions = find_collisions()
    print(f"Collision groups: {len(collisions)}")
    rename_plan: list[tuple[Path, Path, str, str]] = []
    for name, paths in collisions.items():
        for p in paths:
            new_name = derive_new_name(p)
            new_path = p.parent / new_name
            rename_plan.append((p, new_path, name, new_name))
            print(f"  {p.relative_to(VAULT)} -> {new_path.relative_to(VAULT)}")

    if args.dry_run:
        return

    # 1) Rename files
    name_map: dict[str, str] = {}  # old_basename → new_basename
    for old, new, old_name, new_name in rename_plan:
        if old.resolve() != new.resolve():
            shutil.move(str(old), str(new))
            name_map[old.stem] = new.stem

    # 2) Update wikilinks (assumes no folder-qualified links to these basenames)
    pattern = re.compile(r"\[\[([^\]|#]+?)(\|[^\]]*)?\]\]")
    updated = 0
    for note in iter_vault_notes(VAULT, exclude=(".obsidian", ".git", "scripts", "docs", "_templates")):
        text = note.read_text(encoding="utf-8")
        original = text

        def fix(m):
            target = m.group(1).strip()
            base = Path(target).name
            stem = Path(base).stem if base.endswith(".md") else base
            if stem in name_map:
                # Reconstruct link path with renamed stem
                new_target = target.replace(stem, name_map[stem])
                return f"[[{new_target}{m.group(2) or ''}]]"
            return m.group(0)

        text = pattern.sub(fix, text)
        if text != original:
            note.write_text(text, encoding="utf-8")
            updated += 1

    print(f"\nRenamed {len(rename_plan)} files, updated wikilinks in {updated} notes.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run**

Run:
```bash
python3 scripts/pkm/resolve_filename_collisions.py --dry-run
```
검토.

- [ ] **Step 3: 실행**

Run:
```bash
python3 scripts/pkm/resolve_filename_collisions.py
```

- [ ] **Step 4: 검증**

Run:
```bash
python3 -c "
import collections, pathlib
v = pathlib.Path('/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu')
c = collections.Counter()
for p in v.rglob('*.md'):
    if any(x in p.parts for x in ('.obsidian','.git','scripts','docs','_templates')):
        continue
    if p.name == 'README.md':
        continue
    c[p.name] += 1
dups = {k:v for k,v in c.items() if v>1}
print('Remaining collisions (excl. README):', len(dups))
for k,v in dups.items():
    print(' ',k,v)
"
```
Expected: 0

- [ ] **Step 5: 커밋**

```bash
git add -A
git commit -m "refactor(pkm): resolve filename collisions with course-prefix rename (Phase 7)"
```

---

## Task 16: Phase 8 — 36개 README.md → MOC 업그레이드

**Files:**
- Create: `scripts/pkm/upgrade_readmes_to_moc.py`
- Modify: ~36 README.md files

- [ ] **Step 1: upgrade_readmes_to_moc.py 작성**

`scripts/pkm/upgrade_readmes_to_moc.py`:
```python
#!/usr/bin/env python3
"""Phase 8: upgrade course README.md files to MOC type with Dataview block."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import read_note, write_note

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")

DATAVIEW_BLOCK = """
## All notes in this course (auto)
```dataview
TABLE status, file.mtime as updated
FROM "{folder}"
WHERE type != "MOC"
SORT file.mtime DESC
LIMIT 50
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    readmes = list(VAULT.rglob("README.md"))
    readmes = [p for p in readmes if not any(x in p.parts for x in (".obsidian", ".git", "scripts", "docs", "_templates"))]
    print(f"Found {len(readmes)} README files.")

    for p in readmes:
        rel = p.relative_to(VAULT)
        folder = str(rel.parent).replace("\\", "/")
        fm, body = read_note(p)
        # Set type to MOC
        fm["type"] = "MOC"
        fm["status"] = "evergreen"
        existing_tags = set(fm.get("tags") or [])
        existing_tags.add("type/MOC")
        # Drop the auto-injected type/lecture if present
        existing_tags.discard("type/lecture")
        fm["tags"] = sorted(existing_tags)

        # Append Dataview block if not already present
        if "```dataview" not in body:
            body = body.rstrip() + "\n" + DATAVIEW_BLOCK.format(folder=folder)

        if args.dry_run:
            print(f"DRY: {rel} -> type=MOC, tags+=type/MOC, dataview={'```dataview' in body}")
        else:
            write_note(p, fm, body)
            print(f"OK:  {rel}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run**

Run:
```bash
python3 scripts/pkm/upgrade_readmes_to_moc.py --dry-run | head -40
```

- [ ] **Step 3: 실행**

Run:
```bash
python3 scripts/pkm/upgrade_readmes_to_moc.py
```

- [ ] **Step 4: 검증**

Run:
```bash
python3 - <<'PY'
import frontmatter, pathlib
v = pathlib.Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
n = 0
for p in v.rglob("README.md"):
    if any(x in p.parts for x in (".obsidian", ".git", "scripts", "docs", "_templates")):
        continue
    fm = frontmatter.load(p).metadata
    body = p.read_text(encoding="utf-8")
    assert fm.get("type") == "MOC", p
    assert "type/MOC" in fm.get("tags", [])
    assert "```dataview" in body, p
    n += 1
print(f"All {n} READMEs upgraded to MOC.")
PY
```

- [ ] **Step 5: 커밋**

```bash
git add -A
git commit -m "feat(pkm): upgrade ~36 course READMEs to MOC type with Dataview blocks (Phase 8)"
```

---

## Task 17: Phase 9 — 깨진 위키링크 스캔 + 수정

**Files:**
- Create: `scripts/pkm/scan_broken_wikilinks.py`

- [ ] **Step 1: scan_broken_wikilinks.py 작성**

`scripts/pkm/scan_broken_wikilinks.py`:
```python
#!/usr/bin/env python3
"""Phase 9: scan and report broken wikilinks across the vault."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import iter_vault_notes

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
WIKI_PAT = re.compile(r"\[\[([^\]|#]+?)(?:#[^\]|]*)?(\|[^\]]*)?\]\]")


def build_index() -> dict[str, list[Path]]:
    """Map basename (without .md) → list of paths having that name."""
    idx: dict[str, list[Path]] = {}
    for p in iter_vault_notes(VAULT, exclude=(".obsidian", ".git", "scripts", "docs", "_templates")):
        idx.setdefault(p.stem, []).append(p)
    return idx


def main():
    idx = build_index()
    broken: list[tuple[Path, str, int]] = []
    for p in iter_vault_notes(VAULT, exclude=(".obsidian", ".git", "scripts", "docs", "_templates")):
        text = p.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            for m in WIKI_PAT.finditer(line):
                target = m.group(1).strip()
                # Skip image embeds
                if line[max(0, m.start()-1):m.start()] == "!":
                    continue
                stem = Path(target).stem
                base = Path(target).name
                if stem in idx or base in {pp.name for pp in [q for paths in idx.values() for q in paths]}:
                    continue
                # Strict path check: if target contains "/", verify the actual path
                if "/" in target:
                    candidate = VAULT / (target + (".md" if not target.endswith(".md") else ""))
                    if candidate.exists():
                        continue
                broken.append((p, target, line_no))

    print(f"Broken wikilinks: {len(broken)}")
    for src, tgt, ln in broken[:50]:
        print(f"  {src.relative_to(VAULT)}:{ln}  →  [[{tgt}]]")
    if len(broken) > 50:
        print(f"  ... and {len(broken)-50} more")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 스캔 실행**

Run:
```bash
python3 scripts/pkm/scan_broken_wikilinks.py > /tmp/broken-wikilinks.txt 2>&1
head -60 /tmp/broken-wikilinks.txt
wc -l /tmp/broken-wikilinks.txt
```

- [ ] **Step 3: 깨진 링크 수동 분류**

`/tmp/broken-wikilinks.txt`를 읽고 다음 카테고리로 분류:
- (a) 단순 오타 (e.g., 한글 자모 분리) → 수동 fix
- (b) 의도적 stub (아직 안 만든 노트) → 그대로 두거나 stub 파일 생성
- (c) 잘못된 경로 prefix (e.g., `[[ComputerScience/3-1_machine-learning/X]]` → `[[X]]`로 단순화) → sed 일괄

- [ ] **Step 4: 분류 결과에 따라 일괄 수정**

(b) stub 파일 자동 생성 (옵션):
```bash
python3 - <<'PY'
import re, pathlib
broken_log = pathlib.Path("/tmp/broken-wikilinks.txt").read_text(encoding="utf-8")
targets = sorted(set(re.findall(r"→  \[\[([^\]]+)\]\]", broken_log)))
print(f"Unique broken targets: {len(targets)}")
for t in targets[:30]:
    print(" ", t)
PY
```
검토 후 수동 또는 스크립트로 수정.

- [ ] **Step 5: 재스캔**

Run:
```bash
python3 scripts/pkm/scan_broken_wikilinks.py | head -10
```
Expected: 깨진 링크 수 감소

- [ ] **Step 6: 커밋**

```bash
git add -A
git commit -m "fix(pkm): repair broken wikilinks identified by scan (Phase 9)"
```

---

## Task 18: Phase 10 — Linter 전체 적용 + 검증

**Files:**
- Modify: ~479 notes (Linter 자동 정리)

- [ ] **Step 1: 옵시디언 GUI에서 "Lint all files in vault" 실행 안내**

Linter 플러그인은 옵시디언 안에서 실행해야 함. 사용자가 옵시디언 시작 후 다음:
- Command Palette → `Linter: Lint all files in the vault`

자동화 대안: 별도 노드 스크립트 없이 yaml-key-sort만 Python으로 강제 적용:

Run:
```bash
python3 - <<'PY'
import frontmatter, pathlib
from scripts.pkm.lib_frontmatter import read_note, write_note  # 폴더 컨텍스트
import sys; sys.path.insert(0, "scripts/pkm")
from lib_frontmatter import iter_vault_notes, read_note, write_note
v = pathlib.Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
n = 0
for p in iter_vault_notes(v, exclude=(".obsidian",".git","scripts","docs","_templates")):
    fm, body = read_note(p)
    if fm:
        write_note(p, fm, body)  # write_note는 sort_keys=True로 정렬
        n += 1
print(f"Re-serialized {n} notes with sorted YAML keys.")
PY
```

- [ ] **Step 2: 한글 안전성 sanity check**

Run:
```bash
python3 - <<'PY'
import pathlib, random
v = pathlib.Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
notes = [p for p in v.rglob("*.md") if not any(x in p.parts for x in (".obsidian",".git","scripts","docs","_templates"))]
sample = random.sample(notes, k=30)
fail = 0
for p in sample:
    try:
        text = p.read_text(encoding="utf-8")
        text.encode("utf-8")  # round trip
    except Exception as e:
        print("FAIL", p, e); fail += 1
print(f"30 random notes UTF-8 sane: {30-fail}/30")
PY
```
Expected: 30/30

- [ ] **Step 3: 커밋**

```bash
git add -A
git commit -m "chore(pkm): YAML key sort + UTF-8 verification (Phase 10)"
```

---

## Task 19: Phase 11 — 최종 검증

**Files:** 변경 없음 (검증만)

- [ ] **Step 1: 통합 검증 스크립트**

Run:
```bash
python3 - <<'PY'
import frontmatter, pathlib, collections, json
v = pathlib.Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")

required = {"title","date","aliases","tags","type","status","semester","course","created","updated","source"}
notes = [p for p in v.rglob("*.md") if not any(x in p.parts for x in (".obsidian",".git","scripts","docs","_templates"))]

stats = {"total": len(notes), "complete_fm": 0, "missing": collections.Counter(), "type_counts": collections.Counter(), "status_counts": collections.Counter()}

for p in notes:
    fm = frontmatter.load(p).metadata
    miss = required - set(fm.keys())
    if not miss:
        stats["complete_fm"] += 1
    for k in miss:
        stats["missing"][k] += 1
    if fm.get("type"):
        stats["type_counts"][fm["type"]] += 1
    if fm.get("status"):
        stats["status_counts"][fm["status"]] += 1

print(json.dumps({
    "total_notes": stats["total"],
    "complete_frontmatter": stats["complete_fm"],
    "completion_rate": f"{stats['complete_fm']/stats['total']*100:.1f}%",
    "missing_field_counts": dict(stats["missing"]),
    "type_distribution": dict(stats["type_counts"]),
    "status_distribution": dict(stats["status_counts"]),
}, ensure_ascii=False, indent=2))
PY
```
Expected: completion_rate ≥ 99%, 모든 type 분포가 합리적

- [ ] **Step 2: 그래프 시각화 가능성 사전 점검**

Run:
```bash
# MOCs 디렉터리의 모든 MOC가 type/MOC 태그를 가지는지
python3 - <<'PY'
import frontmatter, pathlib
v = pathlib.Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
mocs = list((v/"MOCs").glob("*.md")) + list(v.rglob("README.md"))
mocs = [p for p in mocs if not any(x in p.parts for x in (".obsidian",".git","scripts","docs","_templates"))]
ok = 0
for p in mocs:
    fm = frontmatter.load(p).metadata
    if fm.get("type") == "MOC" and "type/MOC" in (fm.get("tags") or []):
        ok += 1
    else:
        print(f"NON-MOC: {p.relative_to(v)} type={fm.get('type')} tags={fm.get('tags')}")
print(f"\nMOC integrity: {ok}/{len(mocs)}")
PY
```

- [ ] **Step 3: 깨진 링크 재스캔**

Run:
```bash
python3 scripts/pkm/scan_broken_wikilinks.py | head -5
wc -l < <(python3 scripts/pkm/scan_broken_wikilinks.py)
```

- [ ] **Step 4: git log 요약**

Run:
```bash
git log --oneline pkm-refactor ^main | head -30
git diff main..pkm-refactor --stat | tail -5
```

- [ ] **Step 5: 최종 검증 보고서 작성**

`docs/superpowers/specs/2026-05-05-pkm-refactor-completion-report.md` 작성:
- 영향 파일 수
- frontmatter completion rate
- 태그 분포
- type/status 분포
- 깨진 링크 잔존 수
- 다음 단계 제안 (사용자 점진 큐레이션 작업)

- [ ] **Step 6: 최종 커밋**

```bash
git add docs/superpowers/specs/2026-05-05-pkm-refactor-completion-report.md
git commit -m "docs(pkm): completion report — refactor finished, ready for review"
```

---

## Self-Review

이 plan이 spec의 요구를 모두 커버하는지 점검:

- §1 아키텍처 5계층: Task 4(plugin), Task 7(theme/CSS), Task 11~13(data), Task 10(navigation), Task 8(workflow templates) → 커버
- §2 플러그인 결정: Task 4 → 커버
- §3 테마/CSS: Task 7 → 커버
- §4 frontmatter 스키마: Task 11~12 → 커버
- §5 태그 분류: Task 3(map), Task 13(injection) → 커버
- §6 MOC: Task 10(15 MOCs), Task 16(README→MOC) → 커버
- §7 워크플로우 (Templater/Linter/Graph/SR/CC skill): Task 5(Linter), Task 6(Graph), Task 8(Templater) → 커버
  - **Spaced repetition 워크플로우 별도 task 없음** — 플러그인 활성화(Task 4)만으로 충분, 카드 작성은 사용자 점진 작업
  - **CC 스킬 통합 패턴 별도 task 없음** — spec §7.5는 문서일 뿐 자동화 대상 아님
- §8 11 Phase: Task 4~9(P1), Task 10(P2), Task 11(P3), Task 12(P4), Task 13(P5), Task 14(P6), Task 15(P7), Task 16(P8), Task 17(P9), Task 18(P10), Task 19(P11) → 커버
- §9 안전장치: 각 Task 끝에 commit, dry-run 우선, UTF-8 검증 포함
- §11 검증 기준: Task 19 → 커버

**플레이스홀더 검사:** Task 10 Step 2~17은 14개 도메인 MOC 본문을 풀텍스트로 명시하지 않았음 — Step 17까지 풀어 쓰는 대신 첫 두 개(Home, ML)를 풀로 보이고 나머지는 동일 패턴 + 도메인 차이만 명시했음. 실행 시 명확하므로 placeholder 아님.

**일관성 검사:** Task 11(`build_new_fields`)에서 `type` 추론이 README→MOC, MOCs/→MOC을 처리. Task 16에서 README→MOC 업그레이드 시 `type/lecture` 태그 제거 후 `type/MOC` 추가 — 정합됨. Task 12의 Phase 4 머지에서 기존 type을 보존하는 분기와 정합됨.

---

## 변경 이력

- 2026-05-05: 초안 작성 (사용자 자율 진행 권한 부여, 즉시 실행 모드)
