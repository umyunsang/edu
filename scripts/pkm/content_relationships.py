#!/usr/bin/env python3
"""Build note-to-note relationships from note content and archive structure.

This deliberately avoids routing every note through a central map. It reads each
note, chooses a local content parent, then adds semantically related notes based
on title, headings, tags, folder context, and body tokens.
"""
from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
EXCLUDE = {
    ".obsidian",
    ".git",
    ".claude",
    ".agents",
    ".playwright-cli",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    ".aioss-rag",
    "scripts",
    "docs",
    "_templates",
}

FIELD_RE = re.compile(
    r"^(?:up|siblings|related|prerequisites|next|central)::\s*.*$",
    re.MULTILINE,
)
HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
WIKI_RE = re.compile(r"!?\[\[[^\]]+\]\]")
MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\([^)]*\)")
CODE_RE = re.compile(r"(```.*?```|~~~.*?~~~|`[^`]*`)", re.DOTALL)
TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9_+#.-]{2,}")
LEADING_ORDER_RE = re.compile(r"^\d+[-.]?\s*")
FRONTMATTER_END_RE = re.compile(r"^---\s*$", re.MULTILINE)

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "class",
    "file",
    "image",
    "pasted",
    "chapter",
    "week",
    "lecture",
    "midterm",
    "final",
    "exam",
    "summary",
    "문제",
    "풀이",
    "정리",
    "실습",
    "과제",
    "시험",
    "중간",
    "기말",
    "강의",
    "자료",
    "중간고사",
    "기말고사",
    "시험정리",
    "핵심",
    "이론",
    "개념",
    "요약",
    "대비",
    "workspace",
    "context",
}


@dataclass(frozen=True)
class Note:
    path: Path
    rel: str
    stem: str
    title: str
    body: str
    course: str
    parent_key: str
    tokens: Counter[str]


def iter_notes(root: Path) -> list[Path]:
    results: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE and not d.startswith(".")]
        dirpath = unicodedata.normalize("NFC", dirpath)
        for filename in filenames:
            filename = unicodedata.normalize("NFC", filename)
            if filename.endswith(".md"):
                results.append(Path(dirpath) / filename)
    return sorted(results)


def parse_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        return value[1:-1]
    return value


def split_note(path: Path) -> tuple[str, dict[str, object], str]:
    raw = unicodedata.normalize("NFC", path.read_text(encoding="utf-8"))
    if not raw.startswith("---\n"):
        return "", {}, raw
    match = FRONTMATTER_END_RE.search(raw, 4)
    if not match:
        return "", {}, raw
    fm_block = raw[: match.end()] + "\n\n"
    fm_raw = raw[4 : match.start()]
    body = raw[match.end() :].lstrip("\n")
    metadata: dict[str, object] = {}
    current_list: str | None = None
    for line in fm_raw.splitlines():
        if not line.strip():
            continue
        if line.startswith("- ") and current_list:
            metadata.setdefault(current_list, [])
            if isinstance(metadata[current_list], list):
                metadata[current_list].append(parse_scalar(line[2:]))
            continue
        if ":" not in line or line.startswith(" "):
            current_list = None
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            metadata[key] = []
            current_list = key
        else:
            metadata[key] = parse_scalar(value)
            current_list = None
    return fm_block, metadata, body


def write_note(path: Path, fm_block: str, body: str) -> None:
    out = fm_block + body.lstrip("\n") if fm_block else body
    if not out.endswith("\n"):
        out += "\n"
    path.write_text(unicodedata.normalize("NFC", out), encoding="utf-8")


def normalize_name(value: str) -> str:
    value = value.replace("_", " ").replace("-", " ")
    value = LEADING_ORDER_RE.sub("", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def strip_noise(text: str) -> str:
    text = CODE_RE.sub(" ", text)
    text = WIKI_RE.sub(" ", text)
    text = MD_LINK_RE.sub(" ", text)
    return text


def tokenize(*parts: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for part in parts:
        clean = strip_noise(part)
        for raw in TOKEN_RE.findall(clean):
            token = raw.strip("._-").lower()
            if len(token) < 2 or token in STOPWORDS:
                continue
            if token.replace(".", "").isdigit():
                continue
            counts[token] += 1
    return counts


def link(note: Note) -> str:
    return f"[[{note.rel.removesuffix('.md')}|{note.stem}]]"


def course_of(path: Path) -> str:
    try:
        parts = path.relative_to(VAULT).parts
    except ValueError:
        parts = path.parts
    if parts and parts[0] == "ComputerScience" and len(parts) >= 2:
        return parts[1]
    if parts:
        return parts[0]
    return ""


def build_notes() -> list[Note]:
    notes: list[Note] = []
    for path in iter_notes(VAULT):
        rel = path.relative_to(VAULT).as_posix()
        if rel == "README.md":
            continue
        _, fm, body = split_note(path)
        headings = " ".join(HEADING_RE.findall(body))
        title = str(fm.get("title") or "").strip() or path.stem
        tags = " ".join(str(t) for t in (fm.get("tags") or []))
        course = course_of(path)
        weighted = Counter()
        weighted.update(tokenize(title, path.stem, headings, tags))
        weighted.update(tokenize(title, path.stem))
        weighted.update(tokenize(body[:6000]))
        parent_key = normalize_name(path.parent.name)
        notes.append(
            Note(
                path=path,
                rel=rel,
                stem=path.stem,
                title=title,
                body=body,
                course=course,
                parent_key=parent_key,
                tokens=weighted,
            )
        )
    return notes


def cosine(a: Counter[str], b: Counter[str], idf: dict[str, float]) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] * idf[t] * idf[t] for t in common)
    na = math.sqrt(sum(v * v * idf[t] * idf[t] for t, v in a.items()))
    nb = math.sqrt(sum(v * v * idf[t] * idf[t] for t, v in b.items()))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def existing_course_anchor(notes: list[Note], rel: str) -> Note | None:
    for note in notes:
        if note.rel == rel and note.path.exists():
            return note
    return None


def choose_anchors(notes: list[Note]) -> dict[str, Note]:
    preferred = {
        "1-2_coding-basics": "ComputerScience/1-2_coding-basics/중간고사.md",
        "2-1_python": "ComputerScience/2-1_python/1. 변수와 자료형.md",
        "2-1_AI": "ComputerScience/2-1_AI/3. Backpropagation/이론/Backpropagation.md",
        "2-1_computer-architecture": "ComputerScience/2-1_computer-architecture/5. 기억 장치/과제_CacheFriendly코딩실습.md",
        "2-1_data-structures": "ComputerScience/2-1_data-structures/5. 정렬/정렬.md",
        "2-1_linux": "ComputerScience/2-1_linux/1. 리눅스의 기본.md",
        "2-1_probability-statistics": "ComputerScience/2-1_probability-statistics/3.Probability/Probability.md",
        "2-1_web-programming": "ComputerScience/2-1_web-programming/3. Spring Boot 기초/Spring Boot 기초 실습.md",
        "2-2_OSS": "ComputerScience/2-2_OSS/3. 문서 객체 모델/문서 객체 모델(DOM).md",
        "2-2_computer-network": "ComputerScience/2-2_computer-network/14. TCP와 소켓 프로그래밍/TCP와 소켓 프로그래밍.md",
        "2-2_database": "ComputerScience/2-2_database/7. 데이터베이스 언어 SQL/데이터 베이스 언어 SQL.md",
        "2-2_discrete-math": "ComputerScience/2-2_discrete-math/4. 그래프/그래프.md",
        "2-2_operating-system": "ComputerScience/2-2_operating-system/3. 프로세스와 프로세스 관리/프로세스와 프로세스 관리.md",
        "3-1_AI-system-design": "ComputerScience/3-1_AI-system-design/주문 및 결제 AI 시스템 개발.md",
        "3-1_ML-project": "ComputerScience/3-1_ML-project/Python 기초/실력과제.md",
        "3-1_distributed-computing": "ComputerScience/3-1_distributed-computing/쿠다.md",
        "3-1_intellectual-property": "ComputerScience/3-1_intellectual-property/2. 저작권제도와 등록요건/저작권 제도와 등록요건.md",
        "3-1_machine-learning": "ComputerScience/3-1_machine-learning/머신러닝 핵심 수학 개념.md",
        "3-1_mathematical-logic": "ComputerScience/3-1_mathematical-logic/논리학 개론.md",
        "3-1_programming-languages": "ComputerScience/3-1_programming-languages/필기/3. 구문론.md",
        "3-2_bigdata-analysis": "ComputerScience/3-2_bigdata-analysis/md/MLFlow 과제.md",
        "3-2_classics-reading": "ComputerScience/3-2_classics-reading/멋진신세계.md",
        "3-2_computer-graphics": "ComputerScience/3-2_computer-graphics/컴퓨터그래픽스-시험대비.md",
        "3-2_neural-network": "ComputerScience/3-2_neural-network/md/신경망_핵심이론_시험정리.md",
        "3-2_optimization-math": "ComputerScience/3-2_optimization-math/1. Matrix/1. Matrix.md",
        "4-1_AIOSS": "ComputerScience/4-1_AIOSS/md/Week0 - Orientation.md",
        "4-1_algorithm": "ComputerScience/4-1_algorithm/중간고사_정리.md",
        "4-1_computer-vision": "ComputerScience/4-1_computer-vision/중간고사_컴퓨터비전_정밀분석_정리.md",
        "4-1_creative-writing": "ComputerScience/4-1_creative-writing/중간고사_창의적글쓰기_정리.md",
        "elective_LLM": "ComputerScience/elective_LLM/검색 증강 생성 RAG/RAG.md",
        "elective_coding-test": "ComputerScience/elective_coding-test/자료구조/1. 배열과 리스트.md",
        "elective_convergence": "ComputerScience/elective_convergence/프로젝트 주제.md",
        "elective_docker-k8s": "ComputerScience/elective_docker-k8s/도커 기초.md",
        "elective_java": "ComputerScience/elective_java/1. Hello Java.md",
        "misc": "ComputerScience/misc/졸업학점.md",
        "LGAimer": "LGAimer/LG Aimers 9기 지원서 초안.md",
        "certifications": "certifications/체크리스트.md",
    }
    by_course: dict[str, list[Note]] = defaultdict(list)
    for note in notes:
        by_course[note.course].append(note)

    anchors: dict[str, Note] = {}
    for course, members in by_course.items():
        chosen = existing_course_anchor(notes, preferred.get(course, ""))
        if chosen is None:
            root_members = [n for n in members if n.path.parent == VAULT / "ComputerScience" / course]
            pool = root_members or members
            chosen = max(pool, key=lambda n: (len(n.tokens), -len(n.rel)))
        anchors[course] = chosen
    return anchors


def folder_anchor(note: Note, notes_by_parent: dict[Path, list[Note]]) -> Note | None:
    members = notes_by_parent.get(note.path.parent, [])
    if len(members) <= 1:
        return None
    folder_key = normalize_name(note.path.parent.name)
    candidates = [
        n for n in members if n != note and normalize_name(n.stem) == folder_key
    ]
    if candidates:
        return max(candidates, key=lambda n: len(n.tokens))
    same_named = [n for n in members if n != note and folder_key in normalize_name(n.stem)]
    if same_named:
        return max(same_named, key=lambda n: len(n.tokens))
    return None


def anchor_for_folder(folder: Path, notes_by_parent: dict[Path, list[Note]]) -> Note | None:
    members = notes_by_parent.get(folder, [])
    folder_key = normalize_name(folder.name)
    candidates = [n for n in members if normalize_name(n.stem) == folder_key]
    if candidates:
        return max(candidates, key=lambda n: len(n.tokens))
    same_named = [n for n in members if folder_key in normalize_name(n.stem)]
    if same_named:
        return max(same_named, key=lambda n: len(n.tokens))
    return None


def previous_topic_anchor(note: Note, notes_by_parent: dict[Path, list[Note]]) -> Note | None:
    parent = note.path.parent
    grand = parent.parent
    if grand == parent or grand == VAULT:
        return None
    sibling_folders = sorted([p for p in grand.iterdir() if p.is_dir()], key=lambda p: p.name)
    try:
        idx = sibling_folders.index(parent)
    except ValueError:
        return None
    for prev in reversed(sibling_folders[:idx]):
        anchor = anchor_for_folder(prev, notes_by_parent)
        if anchor is not None and anchor != note:
            return anchor
    return None


def select_up(
    note: Note,
    anchors: dict[str, Note],
    prereq_map: dict[str, list[str]],
    notes_by_parent: dict[Path, list[Note]],
) -> Note | None:
    local = folder_anchor(note, notes_by_parent)
    if local is not None:
        return local
    previous = previous_topic_anchor(note, notes_by_parent)
    if previous is not None:
        return previous
    course_anchor = anchors.get(note.course)
    if course_anchor is not None and course_anchor != note:
        return course_anchor
    for course in prereq_map.get(note.course, []):
        prereq = anchors.get(course)
        if prereq is not None and prereq != note:
            return prereq
    return None


def relation_scores(
    note: Note,
    notes: list[Note],
    idf: dict[str, float],
    prereq_map: dict[str, list[str]],
) -> list[tuple[float, Note]]:
    scope = set(prereq_map.get(note.course, []))
    scope.add(note.course)
    for course, prereqs in prereq_map.items():
        if note.course in prereqs:
            scope.add(course)
    scored: list[tuple[float, Note]] = []
    for other in notes:
        if other == note:
            continue
        if other.course not in scope and other.path.parent != note.path.parent:
            continue
        score = cosine(note.tokens, other.tokens, idf)
        if other.course == note.course:
            score += 0.08
        if other.path.parent == note.path.parent:
            score += 0.16
        if other.course in scope:
            score += 0.04
        title_overlap = set(tokenize(note.title)) & set(tokenize(other.title))
        if title_overlap:
            score += 0.05 * len(title_overlap)
        if score > 0.12:
            scored.append((score, other))
    scored.sort(key=lambda item: (-item[0], item[1].rel))
    return scored


def relation_block(
    note: Note,
    up: Note | None,
    related: list[Note],
    prereqs: list[Note],
) -> str:
    lines: list[str] = []
    if up is not None:
        lines.append(f"up:: {link(up)}")
    if prereqs:
        lines.append("prerequisites:: " + ", ".join(link(n) for n in prereqs))
    if related:
        lines.append("related:: " + ", ".join(link(n) for n in related))
    return "\n".join(lines)


def replace_relation_fields(body: str, block: str) -> str:
    cleaned = FIELD_RE.sub("", body)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).lstrip("\n")
    if not block:
        return cleaned
    return block + "\n\n" + cleaned


def main() -> None:
    notes = build_notes()
    anchors = choose_anchors(notes)
    by_parent: dict[Path, list[Note]] = defaultdict(list)
    for note in notes:
        by_parent[note.path.parent].append(note)

    doc_freq = Counter()
    for note in notes:
        doc_freq.update(set(note.tokens))
    idf = {
        token: math.log((1 + len(notes)) / (1 + freq)) + 1.0
        for token, freq in doc_freq.items()
    }

    prereq_map = {
        "2-1_python": ["1-2_coding-basics"],
        "2-1_AI": ["2-1_python", "2-1_probability-statistics"],
        "2-1_data-structures": ["2-1_python"],
        "2-1_linux": ["2-1_python"],
        "2-1_web-programming": ["2-1_python"],
        "2-2_operating-system": ["2-1_computer-architecture", "2-1_linux"],
        "2-2_database": ["2-1_python", "2-1_linux"],
        "2-2_computer-network": ["2-1_computer-architecture"],
        "2-2_discrete-math": ["1-2_coding-basics"],
        "2-2_OSS": ["2-1_web-programming"],
        "3-1_machine-learning": ["2-1_AI", "2-1_probability-statistics"],
        "3-1_ML-project": ["3-1_machine-learning", "2-1_python"],
        "3-1_AI-system-design": ["2-1_AI", "2-2_database"],
        "3-1_distributed-computing": ["2-2_operating-system", "2-2_computer-network"],
        "3-1_mathematical-logic": ["2-2_discrete-math"],
        "3-1_programming-languages": ["2-2_discrete-math", "2-1_data-structures"],
        "3-2_neural-network": ["3-1_machine-learning", "3-2_optimization-math"],
        "3-2_bigdata-analysis": ["3-1_machine-learning", "2-2_database"],
        "3-2_computer-graphics": ["3-2_optimization-math"],
        "3-2_optimization-math": ["2-1_probability-statistics", "2-1_AI"],
        "4-1_algorithm": ["2-1_data-structures", "2-2_discrete-math"],
        "4-1_computer-vision": ["3-2_neural-network", "3-2_computer-graphics"],
        "4-1_AIOSS": ["3-1_distributed-computing", "elective_docker-k8s"],
        "elective_LLM": ["3-1_machine-learning", "3-2_neural-network"],
        "elective_coding-test": ["2-1_data-structures", "4-1_algorithm"],
        "elective_docker-k8s": ["2-1_linux", "2-2_operating-system"],
        "LGAimer": ["3-1_machine-learning", "3-2_bigdata-analysis"],
        "certifications": ["2-2_operating-system", "2-2_database"],
    }

    updated = 0
    for note in notes:
        fm_block, _, body = split_note(note.path)
        up = select_up(note, anchors, prereq_map, by_parent)
        prereqs = [
            anchors[c]
            for c in prereq_map.get(note.course, [])
            if c in anchors and anchors[c] != note and anchors[c] != up
        ][:2]
        scored = relation_scores(note, notes, idf, prereq_map)
        excluded = {note.rel}
        if up is not None:
            excluded.add(up.rel)
        excluded.update(n.rel for n in prereqs)
        related = []
        same_course = [item for item in scored if item[1].course == note.course]
        cross_course = [item for item in scored if item[1].course != note.course]
        for _, other in same_course + cross_course:
            if other.rel in excluded:
                continue
            if "/시험/" in other.rel and "/시험/" not in note.rel and len(related) >= 1:
                continue
            related.append(other)
            if len(related) == 3:
                break
        block = relation_block(note, up, related, prereqs)
        new_body = replace_relation_fields(body, block)
        if new_body != body:
            write_note(note.path, fm_block, new_body)
            updated += 1

    print(f"notes={len(notes)} updated={updated}")


if __name__ == "__main__":
    main()
