#!/usr/bin/env python3
"""Build note-to-note relationships from note content and archive structure.

This deliberately avoids routing every note through a central map. It reads each
note, chooses a local content parent, then adds semantically related notes based
on headings, body tokens, tags, and local folder context. File stems are used as
a weak signal only after content-based renaming has made them reliable labels.
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
    "_templates",
}
EXCLUDE_NOTE_RELS = {
    "README.md",
    "커리큘럼 관계 정리.md",
}

FIELD_RE = re.compile(
    r"^(?:domain|up|siblings|related|prerequisites|next|central)::\s*.*$",
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
    domain_key: str
    parent_key: str
    tokens: Counter[str]


DOMAIN_INTERFACES = {
    "01_programming-foundations": "ComputerScience/01_programming-foundations/프로그래밍 기초 인터페이스.md",
    "02_math-theory": "ComputerScience/02_math-theory/수학 이론 인터페이스.md",
    "03_ai-ml-data": "ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스.md",
    "04_systems-infrastructure": "ComputerScience/04_systems-infrastructure/시스템 인프라 인터페이스.md",
    "05_software-engineering": "ComputerScience/05_software-engineering/소프트웨어 엔지니어링 인터페이스.md",
    "06_algorithms-graphics": "ComputerScience/06_algorithms-graphics/알고리즘 그래픽스 인터페이스.md",
    "07_professional-humanities": "ComputerScience/07_professional-humanities/전문 교양 인터페이스.md",
    "LGAimer": "ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스.md",
    "certifications": "ComputerScience/05_software-engineering/소프트웨어 엔지니어링 인터페이스.md",
}


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


def path_link(rel: str) -> str:
    return f"[[{rel.removesuffix('.md')}|{Path(rel).stem}]]"


def domain_key_of(path: Path) -> str:
    try:
        parts = path.relative_to(VAULT).parts
    except ValueError:
        parts = path.parts
    if parts and parts[0] == "ComputerScience" and len(parts) >= 2:
        return parts[1]
    if parts:
        return parts[0]
    return ""


def course_of(path: Path) -> str:
    try:
        parts = path.relative_to(VAULT).parts
    except ValueError:
        parts = path.parts
    if parts and parts[0] == "ComputerScience" and len(parts) >= 4:
        if re.match(r"^\d{2}_", parts[1]):
            return parts[2]
        return parts[1]
    if parts and parts[0] == "ComputerScience" and len(parts) >= 2:
        return parts[1]
    if parts:
        return parts[0]
    return ""


def build_notes() -> list[Note]:
    notes: list[Note] = []
    for path in iter_notes(VAULT):
        rel = path.relative_to(VAULT).as_posix()
        if rel in EXCLUDE_NOTE_RELS:
            continue
        _, fm, body = split_note(path)
        if str(fm.get("type") or "").strip() == "interface":
            continue
        heading_list = [h.strip() for h in HEADING_RE.findall(body) if h.strip()]
        headings = " ".join(heading_list)
        fm_title = str(fm.get("title") or "").strip()
        title = heading_list[0] if heading_list else (fm_title or path.stem)
        tags = " ".join(str(t) for t in (fm.get("tags") or []))
        course = course_of(path)
        domain_key = domain_key_of(path)
        weighted = Counter()
        weighted.update(tokenize(title, headings, tags))
        weighted.update(tokenize(body[:1200]))
        weighted.update(tokenize(body[1200:6000]))
        weighted.update(tokenize(path.parent.name, path.stem))
        parent_key = normalize_name(path.parent.name)
        notes.append(
            Note(
                path=path,
                rel=rel,
                stem=path.stem,
                title=title,
                body=body,
                course=course,
                domain_key=domain_key,
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
        "coding-basics": "ComputerScience/01_programming-foundations/coding-basics/중간고사.md",
        "python-programming": "ComputerScience/01_programming-foundations/python-programming/1. 변수와 자료형.md",
        "data-structures": "ComputerScience/01_programming-foundations/data-structures/5. 정렬/정렬.md",
        "coding-test": "ComputerScience/01_programming-foundations/coding-test/자료구조/1. 배열과 리스트.md",
        "java-programming": "ComputerScience/01_programming-foundations/java-programming/1. Hello Java.md",
        "probability-statistics": "ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability.md",
        "discrete-mathematics": "ComputerScience/02_math-theory/discrete-mathematics/4. 그래프/그래프.md",
        "optimization-math": "ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix.md",
        "mathematical-logic": "ComputerScience/02_math-theory/mathematical-logic/논리학 개론.md",
        "artificial-intelligence": "ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation.md",
        "machine-learning": "ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념.md",
        "ml-projects": "ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약.md",
        "neural-networks": "ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리.md",
        "big-data-analysis": "ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템.md",
        "computer-vision": "ComputerScience/03_ai-ml-data/computer-vision/중간고사_컴퓨터비전_정밀분석_정리.md",
        "large-language-models": "ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG.md",
        "ai-system-design": "ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발.md",
        "generative-ai-fine-tuning": "ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/프로젝트 주제.md",
        "linux": "ComputerScience/04_systems-infrastructure/linux/1. 리눅스의 기본.md",
        "computer-architecture": "ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/과제_CacheFriendly코딩실습.md",
        "operating-systems": "ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/프로세스와 프로세스 관리.md",
        "computer-networks": "ComputerScience/04_systems-infrastructure/computer-networks/14. TCP와 소켓 프로그래밍/TCP와 소켓 프로그래밍.md",
        "parallel-distributed-computing": "ComputerScience/04_systems-infrastructure/parallel-distributed-computing/쿠다.md",
        "container-orchestration": "ComputerScience/04_systems-infrastructure/container-orchestration/도커 기초.md",
        "web-programming": "ComputerScience/05_software-engineering/web-programming/3. Spring Boot 기초/Spring Boot 기초 실습.md",
        "database-systems": "ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/데이터 베이스 언어 SQL.md",
        "open-source-software": "ComputerScience/05_software-engineering/open-source-software/3. 문서 객체 모델/문서 객체 모델(DOM).md",
        "programming-languages": "ComputerScience/05_software-engineering/programming-languages/필기/3. 구문론.md",
        "aioss-open-source-delivery": "ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week0 - Orientation.md",
        "algorithm-design-analysis": "ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md",
        "computer-graphics": "ComputerScience/06_algorithms-graphics/computer-graphics/컴퓨터그래픽스-시험대비.md",
        "intellectual-property": "ComputerScience/07_professional-humanities/intellectual-property/2. 저작권제도와 등록요건/저작권 제도와 등록요건.md",
        "creative-writing": "ComputerScience/07_professional-humanities/creative-writing/중간고사_창의적글쓰기_정리.md",
        "classics-reading": "ComputerScience/07_professional-humanities/classics-reading/멋진신세계.md",
        "degree-portfolio": "ComputerScience/07_professional-humanities/degree-portfolio/졸업학점.md",
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
            root_members = [n for n in members if n.path.parent.name == course]
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
        same_domain = other.domain_key == note.domain_key and note.domain_key
        same_parent = other.path.parent == note.path.parent
        in_scope = other.course in scope
        score = cosine(note.tokens, other.tokens, idf)
        title_overlap = set(tokenize(note.title)) & set(tokenize(other.title))
        if not (same_domain or same_parent or in_scope or score > 0.075 or title_overlap):
            continue
        if other.course == note.course:
            score += 0.08
        if same_domain:
            score += 0.035
        if same_parent:
            score += 0.16
        if in_scope:
            score += 0.04
        if title_overlap:
            score += 0.05 * len(title_overlap)
        threshold = 0.09
        if same_parent:
            threshold = 0.015
        elif other.course == note.course:
            threshold = 0.03
        elif in_scope:
            threshold = 0.04
        elif same_domain:
            threshold = 0.055
        if score > threshold:
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
    domain_rel = DOMAIN_INTERFACES.get(note.domain_key)
    if domain_rel:
        lines.append(f"domain:: {path_link(domain_rel)}")
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
        "python-programming": ["coding-basics"],
        "artificial-intelligence": ["python-programming", "probability-statistics"],
        "data-structures": ["python-programming"],
        "linux": ["python-programming"],
        "web-programming": ["python-programming"],
        "operating-systems": ["computer-architecture", "linux"],
        "database-systems": ["python-programming", "linux"],
        "computer-networks": ["computer-architecture"],
        "discrete-mathematics": ["coding-basics"],
        "open-source-software": ["web-programming"],
        "machine-learning": ["artificial-intelligence", "probability-statistics"],
        "ml-projects": ["machine-learning", "python-programming"],
        "ai-system-design": ["artificial-intelligence", "database-systems"],
        "parallel-distributed-computing": ["operating-systems", "computer-networks"],
        "mathematical-logic": ["discrete-mathematics"],
        "programming-languages": ["discrete-mathematics", "data-structures"],
        "neural-networks": ["machine-learning", "optimization-math"],
        "big-data-analysis": ["machine-learning", "database-systems"],
        "computer-graphics": ["optimization-math"],
        "optimization-math": ["probability-statistics", "artificial-intelligence"],
        "algorithm-design-analysis": ["data-structures", "discrete-mathematics"],
        "computer-vision": ["neural-networks", "computer-graphics"],
        "aioss-open-source-delivery": ["parallel-distributed-computing", "container-orchestration", "open-source-software"],
        "large-language-models": ["machine-learning", "neural-networks"],
        "coding-test": ["data-structures", "algorithm-design-analysis"],
        "container-orchestration": ["linux", "operating-systems"],
        "generative-ai-fine-tuning": ["large-language-models", "machine-learning"],
        "classics-reading": ["creative-writing"],
        "degree-portfolio": ["creative-writing"],
        "intellectual-property": ["programming-languages"],
        "LGAimer": ["machine-learning", "big-data-analysis"],
        "certifications": ["operating-systems", "database-systems"],
    }

    updated = 0
    for note in notes:
        fm_block, _, body = split_note(note.path)
        up = select_up(note, anchors, prereq_map, by_parent)
        prereqs = [
            anchors[c]
            for c in prereq_map.get(note.course, [])
            if c in anchors and anchors[c] != note and anchors[c] != up
        ]
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
            related.append(other)
        block = relation_block(note, up, related, prereqs)
        new_body = replace_relation_fields(body, block)
        if new_body != body:
            write_note(note.path, fm_block, new_body)
            updated += 1

    print(f"notes={len(notes)} updated={updated}")


if __name__ == "__main__":
    main()
