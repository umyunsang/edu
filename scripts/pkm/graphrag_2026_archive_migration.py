#!/usr/bin/env python3
"""Migrate the vault to a 2026 GraphRAG-Bench aligned archive graph.

The previous research-driven pass attached broad trend/tech/ecosystem links to
ordinary notes. This pass uses 2026 GraphRAG evidence as the method layer only,
then wires source notes/PDFs/code artifacts through course profiles, evidence
indexes, content-derived concepts, query-mode nodes, and community reports.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interface_graph_wiring import DOMAINS, VAULT, link, nfc, write_note  # noqa: E402
from lib_frontmatter import read_note  # noqa: E402

TODAY = "2026-05-28"
KG_ROOT = "ComputerScience/00_graph-interfaces/archive-kg"
METHOD_ROOT = f"{KG_ROOT}/methods-2026"
COMMUNITY_ROOT = f"{KG_ROOT}/communities"
COURSE_ROOT = f"{KG_ROOT}/courses"
CONCEPT_ROOT = f"{KG_ROOT}/concepts"
EVIDENCE_ROOT = f"{KG_ROOT}/evidence"
QUERY_ROOT = f"{KG_ROOT}/query-modes"
MEDIA_ROOT = f"{KG_ROOT}/media"
SKELETON = f"{KG_ROOT}/2026 GraphRAG 아카이브 스켈레톤.md"
COVERAGE_REPORT = f"{KG_ROOT}/파일 커버리지 검증 리포트.md"
RENAME_REPORT = f"{KG_ROOT}/내용 기반 파일명 정합성 리포트.md"
EXTRACTION_REPORT = f"{KG_ROOT}/전문 내용 추출 검증 리포트.md"
HUB = "ComputerScience/00_graph-interfaces/지식그래프 허브.md"
TEXT_CACHE_ROOT = VAULT / "scripts/pkm/.cache/full-text"

OLD_RESEARCH_DIRS = (
    "ComputerScience/00_graph-interfaces/ontology",
    "ComputerScience/00_graph-interfaces/research",
    "ComputerScience/00_graph-interfaces/tech-stacks",
    "ComputerScience/00_graph-interfaces/ecosystems",
    "ComputerScience/00_graph-interfaces/competencies",
)
MANAGED_ARCHIVE_DIRS = (
    METHOD_ROOT,
    COMMUNITY_ROOT,
    COURSE_ROOT,
    CONCEPT_ROOT,
    EVIDENCE_ROOT,
    QUERY_ROOT,
    MEDIA_ROOT,
)

EXCLUDE_DIRS = {
    ".obsidian",
    ".git",
    ".claude",
    ".agents",
    ".aioss-eval",
    ".gemini",
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
GENERATED_DIR = "ComputerScience/00_graph-interfaces"
ROOT_SUPPORT = {"README.md", "AGENTS.md", "CLAUDE.md", "GEMINI.md", "커리큘럼 관계 정리.md"}
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".mp4", ".mov", ".zip"}
TEXTLIKE_SUFFIXES = {
    ".txt",
    ".log",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".xml",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".py",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".sql",
    ".sh",
    ".ipynb",
}

OLD_BROAD_FIELD_RE = re.compile(
    r"^(?:schema|source_model|relation_type|tech_stack|research|ecosystem|competency|evidence)::\s*.*\n?",
    re.MULTILINE,
)
KG_FIELD_RE = re.compile(
    r"^(?:kg_parent|kg_profile|kg_evidence|kg_concepts|kg_query_mode|kg_source_scope)::\s*.*\n?",
    re.MULTILINE,
)
FIELD_BLOCK_RE = re.compile(r"\A((?:(?:[A-Za-z가-힣0-9_-]+)::.*\n)+)\n?")
FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.S)
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
WIKILINK_FULL_RE = re.compile(r"\[\[([^\]|#]+)(#[^\]|]*)?(\|[^\]]*)?\]\]")
MD_LINK_RE = re.compile(r"(!?\[[^\]]*]\()([^)#]+)(#[^)]*)?(\))")
HEADING_RE = re.compile(r"^\s{0,3}#{1,4}\s+(.+?)\s*#*\s*$", re.MULTILINE)
KOREAN_OR_WORD_RE = re.compile(r"[가-힣A-Za-z][가-힣A-Za-z0-9_+./ -]{1,80}")
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")
GENERIC_STEM_RE = re.compile(
    r"^(?:answer[_ -]?\d+|\d{4,}|download|untitled|강의자료|수업자료|문제|문제\s*풀이|문제풀이|연습문제|풀이)$",
    re.I,
)

ARTIFACT_SUFFIXES = {
    ".ipynb",
    ".py",
    ".java",
    ".js",
    ".ts",
    ".c",
    ".cpp",
    ".h",
    ".sql",
    ".sh",
    ".json",
    ".docx",
    ".pptx",
    ".xlsx",
    ".log",
}

STOP_PHRASES = {
    "readme",
    "agents",
    "claude",
    "gemini",
    "중간",
    "기말",
    "중간고사",
    "기말고사",
    "시험",
    "시험정리",
    "정리",
    "문제",
    "문제풀이",
    "풀이",
    "과제",
    "실습",
    "강의",
    "강의자료",
    "확인문제",
    "퀴즈",
    "오답노트",
    "chapter",
    "week",
    "lecture",
    "assignment",
    "answer",
    "and",
    "final",
    "function",
    "import",
    "midterm",
    "model",
    "none",
    "null",
    "or",
    "processed",
    "return",
    "self",
    "converted",
    "source",
    "test",
    "true",
    "false",
    "untitled",
    "variable",
    "개요",
    "소개",
    "목차",
    "요약",
}
SHORT_CONCEPT_WHITELIST = {
    "ai",
    "ml",
    "os",
    "db",
    "sql",
    "api",
    "rag",
    "llm",
    "cnn",
    "rnn",
    "svm",
    "knn",
    "tcp",
    "udp",
    "gpu",
    "cpu",
    "dom",
    "html",
    "css",
    "oop",
    "bfs",
    "dfs",
    "mle",
    "map",
    "pca",
    "orm",
    "ui",
    "ux",
}


@dataclass
class Course:
    key: str
    label: str
    rel_dir: str
    domain_key: str
    domain_title: str
    domain_path: str
    module_path: str
    query_modes: tuple[str, ...]
    md_files: list[Path] = field(default_factory=list)
    pdf_files: list[Path] = field(default_factory=list)
    artifacts: list[Path] = field(default_factory=list)
    media_files: list[Path] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    concept_sources: dict[str, list[Path]] = field(default_factory=dict)
    extraction_stats: Counter[str] = field(default_factory=Counter)


METHOD_SOURCES = (
    (
        "GraphRAG-Bench 2026 리더보드",
        f"{METHOD_ROOT}/GraphRAG-Bench 2026 리더보드.md",
        "2026 기준 구조 선택의 벤치마크 레이어입니다. Novel split은 2026.02 AutoPrunedRetriever-llm이 63.72로 1위이며, Medical split은 2025.09 G-reasoner가 73.30으로 1위입니다.",
        "https://graphrag-bench.github.io/",
        "GraphRAG-Bench leaderboard",
    ),
    (
        "AutoPrunedRetriever 최소 추론 서브그래프",
        f"{METHOD_ROOT}/AutoPrunedRetriever 최소 추론 서브그래프.md",
        "2026.02 GraphRAG-Bench Novel 1위 행에서 확인한 방법론 기준입니다. 아카이브에는 모든 가능한 간선을 늘리는 대신, source evidence와 겹치는 핵심 개념을 남기는 최소 추론 서브그래프로 적용합니다.",
        "https://arxiv.org/abs/2602.04926",
        "arXiv 2602.04926",
    ),
    (
        "Youtu-GraphRAG 4단계 지식 트리",
        f"{METHOD_ROOT}/Youtu-GraphRAG 4단계 지식 트리.md",
        "ICLR 2026 채택 방법입니다. attributes, relations, keywords, communities의 4단계 지식 트리를 아카이브 구조의 source, evidence, concept, community 계층으로 옮깁니다.",
        "https://github.com/TencentCloudADP/youtu-graphrag",
        "TencentCloudADP/youtu-graphrag",
    ),
    (
        "FalkorDB GraphRAG-SDK 파이프라인",
        f"{METHOD_ROOT}/FalkorDB GraphRAG-SDK 파이프라인.md",
        "GraphRAG-Bench 재현 성능이 높은 오픈소스 파이프라인입니다. chunking, extraction, coreference, exact/semantic/LLM verified resolution, multipath retrieval를 Obsidian note graph로 축소 적용합니다.",
        "https://github.com/FalkorDB/GraphRAG-SDK/blob/main/docs/benchmark.md",
        "FalkorDB GraphRAG-SDK benchmark",
    ),
    (
        "LinearRAG 관계 추출 과잉 회피",
        f"{METHOD_ROOT}/LinearRAG 관계 추출 과잉 회피.md",
        "ICLR 2026 계열의 relation-free GraphRAG 흐름입니다. 노이즈가 많은 임의 관계명보다 source chunk, concept, course, community의 안정된 링크를 우선합니다.",
        "https://arxiv.org/abs/2508.14450",
        "LinearRAG arXiv",
    ),
    (
        "BRINK KG-RAG 근거 감사",
        f"{METHOD_ROOT}/BRINK KG-RAG 근거 감사.md",
        "EACL 2026 계열 평가 관점입니다. 직접 triple이나 누락 지식에 과적합된 KG-RAG 평가를 경계하고, 모든 연결에 로컬 근거 파일을 남기는 방식으로 적용합니다.",
        "https://arxiv.org/abs/2504.03188",
        "BRINK arXiv",
    ),
    (
        "Atomic Educational GraphRAG",
        f"{METHOD_ROOT}/Atomic Educational GraphRAG.md",
        "AAAI 2026 교육 도메인 GraphRAG 흐름입니다. 강의/과제/PDF를 atomic fact와 traceable evidence로 쪼개는 원칙을 과목별 근거 인덱스에 적용합니다.",
        "https://arxiv.org/abs/2504.00557",
        "Educational GraphRAG arXiv",
    ),
    (
        "WildGraphBench 현실 코퍼스 평가",
        f"{METHOD_ROOT}/WildGraphBench 현실 코퍼스 평가.md",
        "2026 현실 코퍼스 평가 관점입니다. multi-fact reasoning에는 그래프가 유리하지만, 세부 사실 검색에는 source-scoped evidence가 필요하다는 점을 반영합니다.",
        "https://arxiv.org/abs/2506.01143",
        "WildGraphBench arXiv",
    ),
)

QUERY_MODES = {
    "fact": (
        "Fact Retrieval",
        f"{QUERY_ROOT}/Fact Retrieval.md",
        "정의, 공식, 명령, 파일 위치처럼 단일 근거를 빠르게 찾는 질의 모드입니다.",
    ),
    "complex": (
        "Complex Reasoning",
        f"{QUERY_ROOT}/Complex Reasoning.md",
        "여러 강의 노트, PDF, 코드 산출물을 이어서 설명해야 하는 질의 모드입니다.",
    ),
    "context": (
        "Contextual Summarize",
        f"{QUERY_ROOT}/Contextual Summarize.md",
        "과목이나 프로젝트 묶음의 맥락을 요약하는 질의 모드입니다.",
    ),
    "creative": (
        "Creative Generation",
        f"{QUERY_ROOT}/Creative Generation.md",
        "지원서, 발표 스크립트, 프로젝트 제안처럼 근거 기반 산출물을 생성하는 질의 모드입니다.",
    ),
}

QUERY_BY_DOMAIN = {
    "programming": ("fact", "complex"),
    "math": ("fact", "complex"),
    "ai": ("fact", "complex", "context"),
    "systems": ("fact", "complex"),
    "software": ("fact", "complex", "creative"),
    "algorithms": ("fact", "complex"),
    "professional": ("context", "creative"),
    "external": ("context", "creative"),
    "certification": ("fact", "context"),
}


def rel(path: Path | str) -> str:
    p = Path(path)
    if p.is_absolute():
        p = p.relative_to(VAULT)
    return str(p).replace("\\", "/")


def wikilink(path: str | Path, alias: str | None = None) -> str:
    target = rel(path).removesuffix(".md")
    return f"[[{target}|{alias}]]" if alias else f"[[{target}]]"


def split_frontmatter(text: str) -> tuple[str, str]:
    match = FRONTMATTER_RE.match(text)
    if match:
        return text[: match.end()].rstrip() + "\n\n", text[match.end() :].lstrip("\n")
    return "", text


def strip_old_fields(body: str) -> str:
    body = OLD_BROAD_FIELD_RE.sub("", body)
    body = KG_FIELD_RE.sub("", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.lstrip("\n")


def insert_field_block(body: str, fields: list[tuple[str, str]]) -> str:
    if not fields:
        return body
    block = "".join(f"{key}:: {value}\n" for key, value in fields)
    match = FIELD_BLOCK_RE.match(body)
    if match:
        return body[: match.end()] + block + "\n" + body[match.end() :].lstrip("\n")
    return block + "\n" + body.lstrip("\n")


def upsert_text_file(path: Path | str, text: str) -> bool:
    full = VAULT / path
    full.parent.mkdir(parents=True, exist_ok=True)
    text = nfc(text.rstrip() + "\n")
    if full.exists() and nfc(full.read_text(encoding="utf-8")) == text:
        return False
    full.write_text(text, encoding="utf-8")
    return True


def note_header(title: str, tags: list[str], note_type: str = "interface") -> list[str]:
    lines = [
        "---",
        "aliases: []",
        "course: archive-kg",
        f"created: '{TODAY}'",
        f"date: '{TODAY}'",
        "semester: meta",
        "source: ''",
        "status: evergreen",
        "tags:",
    ]
    for tag in tags:
        lines.append(f"- {tag}")
    lines.extend(
        [
            f"title: {title}",
            f"type: {note_type}",
            f"updated: '{TODAY}'",
            "---",
            "",
        ]
    )
    return lines


def slugify(value: str, fallback: str = "concept") -> str:
    value = nfc(value)
    value = re.sub(r"[\\/:*?\"<>|#^[\]]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" ._-")
    if len(value) > 72:
        value = value[:72].rstrip(" ._-")
    return value or fallback


def concept_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def clean_phrase(value: str) -> str:
    value = nfc(value)
    value = re.sub(r"!?\[\[[^\]]+\]\]", " ", value)
    value = re.sub(r"`[^`]*`", " ", value)
    value = re.sub(r"\*\*|__|\*|_", " ", value)
    value = re.sub(r"^[0-9]+[.)_-]*\s*", "", value)
    value = re.sub(r"\.(md|pdf|ipynb|py|java|js|sql|docx|pptx)$", "", value, flags=re.I)
    value = value.replace("%20", " ")
    value = re.sub(r"[_-]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .,:;!?()[]{}\"'")


def keep_phrase(value: str) -> bool:
    if not value:
        return False
    low = value.lower()
    if low in STOP_PHRASES:
        return False
    compact = re.sub(r"\s+", "", low)
    if compact in STOP_PHRASES:
        return False
    if len(compact) <= 3 and compact not in SHORT_CONCEPT_WHITELIST:
        return False
    if len(value) < 2 or len(value) > 72:
        return False
    if re.fullmatch(r"[0-9.\- ]+", value):
        return False
    if not re.search(r"[가-힣A-Za-z]", value):
        return False
    return True


def candidate_phrases(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        key = concept_key(value)
        if key not in seen:
            seen.add(key)
            out.append(value)

    for heading in HEADING_RE.findall(text):
        heading = clean_phrase(heading)
        if keep_phrase(heading):
            add(heading)
        for part in re.split(r"\s*[,:;/|]\s*", heading):
            part = clean_phrase(part)
            if keep_phrase(part):
                add(part)
    for acronym in ACRONYM_RE.findall(text):
        if keep_phrase(acronym):
            add(acronym)
    body_added = 0
    for match in KOREAN_OR_WORD_RE.finditer(text):
        phrase = clean_phrase(match.group(0))
        if keep_phrase(phrase) and 2 <= len(phrase.split()) <= 5:
            add(phrase)
            body_added += 1
        if body_added >= 1600:
            break
    return out


def cache_key(path: Path) -> str:
    stat = path.stat()
    raw = f"{rel(path)}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def pdf_text(path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return ""
    TEXT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache = TEXT_CACHE_ROOT / f"{cache_key(path)}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="ignore")
    try:
        proc = subprocess.run(
            [pdftotext, str(path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return ""
    text = nfc(proc.stdout)
    cache.write_text(text, encoding="utf-8")
    return text


def textlike_full_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".ipynb":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            chunks: list[str] = []
            for cell in data.get("cells", []):
                source = cell.get("source", "")
                if isinstance(source, list):
                    chunks.extend(str(item) for item in source)
                else:
                    chunks.append(str(source))
            return nfc("\n".join(chunks))
        except Exception:
            return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".docx", ".pptx", ".xlsx"}:
        return office_zip_text(path)
    try:
        return nfc(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ""


def office_zip_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            chunks: list[str] = []
            for name in zf.namelist():
                if not name.endswith(".xml"):
                    continue
                if not (
                    name.startswith("word/")
                    or name.startswith("ppt/")
                    or name.startswith("xl/sharedStrings")
                    or name.startswith("xl/worksheets")
                ):
                    continue
                try:
                    root = ET.fromstring(zf.read(name))
                except Exception:
                    continue
                for node in root.iter():
                    if node.text and node.text.strip():
                        chunks.append(node.text.strip())
            return nfc("\n".join(chunks))
    except Exception:
        return ""


def title_lines_from_text(text: str, limit: int = 4) -> list[str]:
    lines: list[str] = []
    for raw in nfc(text).splitlines():
        line = clean_phrase(raw)
        low = line.lower()
        if not line or low in STOP_PHRASES:
            continue
        if line in {"by", "page", "copyright"}:
            continue
        if re.fullmatch(r"[-0-9 /.,]+", line):
            continue
        if keep_phrase(line) or len(line) >= 6:
            lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def content_title_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".md":
        try:
            fm, body = read_note(path)
        except Exception:
            fm, body = {}, path.read_text(encoding="utf-8", errors="ignore")
        title = clean_phrase(str(fm.get("title") or ""))
        if keep_phrase(title):
            return title
        headings = HEADING_RE.findall(body)
        if headings:
            title = clean_phrase(headings[0])
            if keep_phrase(title):
                return title
    elif suffix == ".pdf":
        lines = title_lines_from_text(pdf_text(path), 3)
        if lines:
            return clean_phrase(" ".join(lines[:2]))
    elif suffix in TEXTLIKE_SUFFIXES or suffix in {".docx", ".pptx", ".xlsx"}:
        lines = title_lines_from_text(textlike_full_text(path), 3)
        if lines:
            return clean_phrase(" ".join(lines[:2]))
    return ""


def iter_files() -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(VAULT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        dirpath_nfc = nfc(dirpath)
        for fn in filenames:
            if fn.startswith("."):
                continue
            files.append(Path(dirpath_nfc) / nfc(fn))
    return sorted(files)


def build_courses() -> dict[str, Course]:
    courses: dict[str, Course] = {}
    for domain in DOMAINS:
        for anchor in domain.courses:
            parts = Path(anchor.rel).parts
            if len(parts) < 3:
                continue
            rel_dir = str(Path(*parts[:3])).replace("\\", "/")
            key = parts[2]
            courses[key] = Course(
                key=key,
                label=anchor.label,
                rel_dir=rel_dir,
                domain_key=domain.key,
                domain_title=domain.title,
                domain_path=domain.path,
                module_path=f"ComputerScience/00_graph-interfaces/courses/{anchor.label} 인터페이스.md",
                query_modes=QUERY_BY_DOMAIN.get(domain.key, ("fact", "context")),
            )
    courses["LGAimer"] = Course(
        key="LGAimer",
        label="LG Aimers",
        rel_dir="LGAimer",
        domain_key="external",
        domain_title="외부 프로그램 커뮤니티",
        domain_path=f"{COMMUNITY_ROOT}/외부 프로그램 커뮤니티.md",
        module_path="ComputerScience/00_graph-interfaces/courses/LG Aimers 인터페이스.md",
        query_modes=QUERY_BY_DOMAIN["external"],
    )
    courses["certifications"] = Course(
        key="certifications",
        label="자격증",
        rel_dir="certifications",
        domain_key="certification",
        domain_title="자격증 검증 커뮤니티",
        domain_path=f"{COMMUNITY_ROOT}/자격증 검증 커뮤니티.md",
        module_path="ComputerScience/00_graph-interfaces/courses/자격증 인터페이스.md",
        query_modes=QUERY_BY_DOMAIN["certification"],
    )
    courses["shared-media"] = Course(
        key="shared-media",
        label="공유 미디어",
        rel_dir="image",
        domain_key="media",
        domain_title="공유 미디어 커뮤니티",
        domain_path=f"{COMMUNITY_ROOT}/공유 미디어 커뮤니티.md",
        module_path=HUB,
        query_modes=("fact", "context"),
    )
    courses["archive-operations"] = Course(
        key="archive-operations",
        label="아카이브 운영",
        rel_dir=".",
        domain_key="operations",
        domain_title="아카이브 운영 커뮤니티",
        domain_path=f"{COMMUNITY_ROOT}/아카이브 운영 커뮤니티.md",
        module_path=HUB,
        query_modes=("context", "creative"),
    )
    return courses


def find_course_for_path(path: Path, courses: dict[str, Course]) -> Course | None:
    r = rel(path)
    if r.startswith("image/"):
        return courses.get("shared-media")
    if "/" not in r and path.name in ROOT_SUPPORT | {"2026 GraphRAG 아카이브.canvas", "지식그래프 레벨 인터페이스.canvas", "커리큘럼 관계 그래프.canvas"}:
        return courses.get("archive-operations")
    if r.startswith("_templates/"):
        return courses.get("archive-operations")
    for course in sorted(courses.values(), key=lambda c: len(c.rel_dir), reverse=True):
        if course.key in {"shared-media", "archive-operations"}:
            continue
        if r == course.rel_dir or r.startswith(course.rel_dir.rstrip("/") + "/"):
            return course
    return None


def build_file_index(files: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        r = rel(path)
        index[nfc(path.name)].append(path)
        index[nfc(path.stem)].append(path)
        index[nfc(r)].append(path)
        index[nfc(r.removesuffix(".md"))].append(path)
    return index


MD_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")


def resolve_target(target: str, source: Path, index: dict[str, list[Path]]) -> Path | None:
    target = nfc(target.strip())
    if not target or target.startswith(("http://", "https://", "attachment:")):
        return None
    target = target.split("#", 1)[0].strip()
    target = target.split("|", 1)[0].strip()
    target = target.replace("%20", " ")
    candidates: list[Path] = []
    direct = VAULT / target
    if direct.exists():
        candidates.append(direct)
    relative = (source.parent / target).resolve()
    try:
        relative.relative_to(VAULT)
        if relative.exists():
            candidates.append(relative)
    except Exception:
        pass
    candidates.extend(index.get(target, []))
    candidates.extend(index.get(Path(target).name, []))
    candidates.extend(index.get(Path(target).stem, []))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def collect_sources(courses: dict[str, Course]) -> None:
    files = iter_files()
    index = build_file_index(files)
    for path in files:
        r = rel(path)
        if r.startswith(GENERATED_DIR + "/"):
            continue
        course = find_course_for_path(path, courses)
        if not course:
            continue
        suffix = path.suffix.lower()
        if suffix == ".md":
            course.md_files.append(path)
        elif suffix == ".pdf":
            course.pdf_files.append(path)
        elif suffix in MEDIA_SUFFIXES:
            course.media_files.append(path)
        else:
            course.artifacts.append(path)

    # Attach shared image/media files to the course notes that embed them. This
    # turns attachment nodes into course-local evidence instead of leaving them
    # only in the global shared-media bucket.
    for course in courses.values():
        if course.key in {"shared-media", "archive-operations"}:
            continue
        media_seen = {rel(p) for p in course.media_files}
        for md in course.md_files:
            try:
                text = md.read_text(encoding="utf-8")
            except Exception:
                continue
            targets = [m.group(1) for m in WIKILINK_RE.finditer(text)]
            targets.extend(m.group(1) for m in MD_IMAGE_RE.finditer(text))
            for target in targets:
                resolved = resolve_target(target, md, index)
                if not resolved:
                    continue
                if resolved.suffix.lower() in MEDIA_SUFFIXES and rel(resolved) not in media_seen:
                    course.media_files.append(resolved)
                    media_seen.add(rel(resolved))


def extract_concepts(courses: dict[str, Course]) -> None:
    for course in courses.values():
        counts: Counter[str] = Counter()
        source_map: dict[str, list[Path]] = defaultdict(list)
        for md in course.md_files:
            try:
                fm, body = read_note(md)
            except Exception:
                continue
            course.extraction_stats["markdown_full_files"] += 1
            course.extraction_stats["markdown_chars"] += len(body)
            heading_set = set(HEADING_RE.findall(body))
            seed = "\n".join(
                [
                    str(fm.get("title") or ""),
                    clean_phrase(md.stem),
                    body,
                ]
            )
            for phrase in candidate_phrases(seed):
                key = concept_key(phrase)
                counts[key] += 3 if phrase in heading_set else 1
                if len(source_map[key]) < 8:
                    source_map[key].append(md)
        for pdf in course.pdf_files:
            extracted = pdf_text(pdf)
            course.extraction_stats["pdf_full_files"] += 1
            course.extraction_stats["pdf_chars"] += len(extracted)
            if not extracted.strip():
                course.extraction_stats["pdf_text_empty_or_failed"] += 1
            seed = clean_phrase(pdf.stem) + "\n" + extracted
            for phrase in candidate_phrases(seed):
                key = concept_key(phrase)
                counts[key] += 2
                if len(source_map[key]) < 8:
                    source_map[key].append(pdf)
        for artifact in course.artifacts:
            phrase = clean_phrase(artifact.stem)
            if keep_phrase(phrase):
                key = concept_key(phrase)
                counts[key] += 2
                if len(source_map[key]) < 8:
                    source_map[key].append(artifact)
            if artifact.suffix.lower() in TEXTLIKE_SUFFIXES:
                text = textlike_full_text(artifact)
                course.extraction_stats["textlike_full_files"] += 1
                course.extraction_stats["textlike_chars"] += len(text)
                for phrase in candidate_phrases(text):
                    key = concept_key(phrase)
                    counts[key] += 1
                    if len(source_map[key]) < 8:
                        source_map[key].append(artifact)
            elif artifact.suffix.lower() in {".docx", ".pptx", ".xlsx"}:
                text = textlike_full_text(artifact)
                course.extraction_stats["office_xml_text_files"] += 1
                course.extraction_stats["office_xml_chars"] += len(text)
                for phrase in candidate_phrases(text):
                    key = concept_key(phrase)
                    counts[key] += 1
                    if len(source_map[key]) < 8:
                        source_map[key].append(artifact)
        for media in course.media_files:
            course.extraction_stats["media_filename_files"] += 1
            phrase = clean_phrase(media.stem)
            if keep_phrase(phrase):
                key = concept_key(phrase)
                counts[key] += 1
                if len(source_map[key]) < 8:
                    source_map[key].append(media)

        title_case: dict[str, str] = {}
        for key in counts:
            samples = [clean_phrase(p.stem) for p in source_map.get(key, [])]
            heading_like = [s for s in samples if concept_key(s) == key and keep_phrase(s)]
            title_case[key] = heading_like[0] if heading_like else slugify(key)

        # Prefer compact, recurring, course-specific phrases. Preserve common acronyms.
        ranked = sorted(
            counts.items(),
            key=lambda kv: (
                kv[1],
                bool(re.fullmatch(r"[A-Z0-9]{2,10}", title_case.get(kv[0], ""))),
                -len(kv[0]),
            ),
            reverse=True,
        )
        selected: list[str] = []
        seen_words: set[str] = set()
        limit = 18 if len(course.md_files) + len(course.pdf_files) > 40 else 12
        for key, _count in ranked:
            phrase = slugify(title_case.get(key, key))
            low = concept_key(phrase)
            if low in seen_words or not keep_phrase(phrase):
                continue
            if any(low in concept_key(x) or concept_key(x) in low for x in selected if len(low) > 5):
                continue
            selected.append(phrase)
            seen_words.add(low)
            if len(selected) >= limit:
                break
        if not selected:
            selected = [course.label]
            source_map[concept_key(course.label)] = course.md_files[:3] + course.pdf_files[:3]
        course.concepts = selected
        course.concept_sources = {concept: source_map.get(concept_key(concept), []) for concept in selected}


def remove_old_generated_research_dirs() -> int:
    removed = 0
    for rel_dir in OLD_RESEARCH_DIRS:
        full = VAULT / rel_dir
        if full.exists():
            shutil.rmtree(full)
            removed += 1
    return removed


def reset_managed_archive_dirs() -> int:
    full_root = VAULT / KG_ROOT
    if full_root.exists():
        robust_rmtree(full_root)
        return 1
    return 0


def robust_rmtree(path: Path) -> None:
    """Remove generated directories reliably on iCloud-backed folders."""
    try:
        shutil.rmtree(path)
        return
    except OSError:
        pass
    if not path.exists():
        return
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        current = Path(dirpath)
        for fn in filenames:
            try:
                (current / fn).unlink()
            except FileNotFoundError:
                pass
        for dn in dirnames:
            try:
                (current / dn).rmdir()
            except OSError:
                pass
    try:
        path.rmdir()
    except OSError:
        # One final shutil pass catches any late iCloud placeholder cleanup.
        shutil.rmtree(path, ignore_errors=True)


def method_note(title: str, path: str, summary: str, source_url: str, source_label: str) -> str:
    lines = note_header(title, ["type/interface", "pkm/kg-method-2026"])
    lines.extend(
        [
            f"kg_parent:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
            f"kg_skeleton:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
            "",
            f"# {title}",
            "",
            summary,
            "",
            "## 외부 근거",
            "",
            f"- [{source_label}]({source_url})",
            "",
            "## vault 적용 방식",
            "",
            "- 외부 방법론은 구조 선택 근거로만 사용합니다.",
            "- 실제 노드와 간선은 이 vault의 노트, PDF, 코드/산출물에서 확보한 로컬 근거에 연결합니다.",
        ]
    )
    return "\n".join(lines)


def write_method_layer(courses: dict[str, Course]) -> int:
    updated = 0
    method_links = [wikilink(path, title) for title, path, *_ in METHOD_SOURCES]
    query_links = [wikilink(path, title) for title, path, _ in QUERY_MODES.values()]
    community_links = [
        wikilink(f"{COMMUNITY_ROOT}/{domain.title.replace(' 인터페이스', '')} 커뮤니티.md", domain.title.replace(" 인터페이스", " 커뮤니티"))
        for domain in DOMAINS
    ]
    community_links += [
        wikilink(f"{COMMUNITY_ROOT}/외부 프로그램 커뮤니티.md"),
        wikilink(f"{COMMUNITY_ROOT}/자격증 검증 커뮤니티.md"),
        wikilink(f"{COMMUNITY_ROOT}/공유 미디어 커뮤니티.md"),
        wikilink(f"{COMMUNITY_ROOT}/아카이브 운영 커뮤니티.md"),
    ]

    lines = note_header("2026 GraphRAG 아카이브 스켈레톤", ["type/interface", "pkm/kg-skeleton"])
    lines.extend(
        [
            "method:: " + ", ".join(method_links),
            "query_mode:: " + ", ".join(query_links),
            "community:: " + ", ".join(community_links),
            "",
            "# 2026 GraphRAG 아카이브 스켈레톤",
            "",
            "이 스켈레톤은 2026년 기준 GraphRAG-Bench와 ICLR/EACL/AAAI 계열 KG-RAG 연구에서 반복적으로 확인되는 구조를 Obsidian vault에 맞게 옮긴 것입니다.",
            "",
            "## 적용 원칙",
            "",
            "- **Top-down interface first**: 허브와 스켈레톤에서 방법론, 질의 모드, 분야 커뮤니티, 과목 프로필, 근거 파일 순서로 내려갑니다.",
            "- **Full source extraction**: 원문 노트 전체, PDF 전체 텍스트, 텍스트/코드/노트북/Office 산출물 텍스트를 추출한 뒤 개념을 계산합니다.",
            "- **Atomic evidence**: 과목별 근거 인덱스가 원문 파일을 모아 출처와 적용 범위를 보존합니다.",
            "- **Minimal reasoning subgraph**: 일반 연구/스택 링크를 모든 노트에 붙이지 않고, 과목별 핵심 개념과 공유 개념을 연결합니다.",
            "- **4-level knowledge tree**: method/query, community, course/concept, evidence/source 계층으로 탐색합니다.",
            "- **Resolution before linking**: 파일명만 보지 않고 heading, frontmatter title, PDF 전체 텍스트, 코드/산출물 내용을 함께 보고 개념을 합칩니다.",
            "- **Query-mode view**: GraphRAG-Bench의 fact, complex, contextual, creative 질의 유형을 아카이브 탐색 인터페이스로 둡니다.",
            "",
            "## Top-down 계층",
            "",
            "1. 허브/스켈레톤: 전체 GraphRAG 방식과 탐색 계층",
            "2. 방법론/질의 모드: 2026 연구와 retrieval 목적별 관점",
            "3. 커뮤니티 리포트: 프로그래밍, 수학, AI, 시스템, 소프트웨어 등 분야별 인터페이스",
            "4. 과목 프로필/개념 노드: 과목별 reasoning subgraph와 concept codebook",
            "5. 근거 인덱스/원문 파일: 강의 노트, PDF, 코드/노트북, 과제 산출물, PNG 첨부",
            "",
            "## 과목 프로필",
            "",
        ]
    )
    for course in sorted(courses.values(), key=lambda c: c.label):
        lines.append(f"- {wikilink(course_profile_path(course), course.label)}")
    updated += int(upsert_text_file(SKELETON, "\n".join(lines)))

    hub_lines = note_header("지식그래프 허브", ["type/interface", "pkm/hub", "pkm/kg-skeleton"])
    hub_lines.extend(
        [
            f"kg_parent:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
            f"kg_skeleton:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
            "method:: " + ", ".join(method_links[:5]),
            "",
            "# 지식그래프 허브",
            "",
            "이 vault의 주 지식그래프 진입점입니다. README나 특정 정리 문서를 부모로 삼지 않고, 2026 GraphRAG 구조에 맞춘 스켈레톤, 커뮤니티, 과목 프로필, 근거 인덱스로 이동합니다.",
            "",
            "## 2026 스켈레톤",
            "",
            f"- {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
            "",
            "## 커뮤니티",
            "",
        ]
    )
    for item in community_links:
        hub_lines.append(f"- {item}")
    updated += int(upsert_text_file(HUB, "\n".join(hub_lines)))

    for title, path, summary, url, label in METHOD_SOURCES:
        updated += int(upsert_text_file(path, method_note(title, path, summary, url, label)))

    for key, (title, path, summary) in QUERY_MODES.items():
        lines = note_header(title, ["type/interface", "pkm/kg-query"])
        course_links = [
            wikilink(course_profile_path(course), course.label)
            for course in sorted(courses.values(), key=lambda c: c.label)
            if key in course.query_modes
        ]
        lines.extend(
            [
                f"kg_parent:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
                f"kg_skeleton:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
                "kg_courses:: " + ", ".join(course_links),
                "",
                f"# {title}",
                "",
                summary,
                "",
                "## 연결 과목",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in course_links)
        updated += int(upsert_text_file(path, "\n".join(lines)))
    return updated


def course_profile_path(course: Course) -> str:
    return f"{COURSE_ROOT}/{slugify(course.label)} 지식그래프.md"


def course_evidence_path(course: Course) -> str:
    return f"{EVIDENCE_ROOT}/{slugify(course.label)} 근거 인덱스.md"


def concept_path(course: Course, concept: str) -> str:
    return f"{CONCEPT_ROOT}/{slugify(course.key)}/{slugify(concept)}.md"


def community_path(course: Course) -> str:
    if course.domain_key in {"external", "certification"}:
        return course.domain_path
    title = course.domain_title.replace(" 인터페이스", " 커뮤니티")
    return f"{COMMUNITY_ROOT}/{slugify(title)}.md"


def first_links(paths: list[Path], limit: int = 12) -> list[str]:
    return [wikilink(p) for p in paths[:limit]]


def relation_tokens(value: str) -> set[str]:
    tokens = {clean_phrase(t).lower() for t in re.split(r"[^가-힣A-Za-z0-9]+", value)}
    return {
        t
        for t in tokens
        if len(t) >= 3
        and t not in STOP_PHRASES
        and t not in {"data", "model", "basic", "example", "note", "course", "lecture", "week"}
    }


def write_course_layers(courses: dict[str, Course]) -> int:
    updated = 0
    by_domain: dict[str, list[Course]] = defaultdict(list)
    for course in courses.values():
        by_domain[course.domain_key].append(course)

    all_concepts: list[tuple[Course, str, set[str], str]] = []
    for course in courses.values():
        for concept in course.concepts:
            all_concepts.append((course, concept, relation_tokens(concept), concept_key(concept)))

    for course in courses.values():
        concept_links = [wikilink(concept_path(course, c), c) for c in course.concepts]
        evidence_link = wikilink(course_evidence_path(course), f"{course.label} 근거 인덱스")
        query_links = [wikilink(QUERY_MODES[k][1], QUERY_MODES[k][0]) for k in course.query_modes]
        source_note_links = first_links(course.md_files, 20)
        source_pdf_links = first_links(course.pdf_files, 20)
        artifact_links = first_links(course.artifacts, 20)

        profile_lines = note_header(f"{course.label} 지식그래프", ["type/interface", "pkm/kg-course"])
        profile_lines.extend(
            [
                f"kg_parent:: {wikilink(community_path(course))}",
                f"kg_skeleton:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
                f"kg_community:: {wikilink(community_path(course))}",
                f"kg_evidence:: {evidence_link}",
                "kg_query_mode:: " + ", ".join(query_links),
                "kg_concepts:: " + ", ".join(concept_links),
                f"module:: {wikilink(course.module_path, course.label + ' 인터페이스')}",
                f"domain:: {wikilink(course.domain_path, course.domain_title)}",
                "",
                f"# {course.label} 지식그래프",
                "",
                f"{course.label} 과목/활동의 로컬 노트, PDF, 코드/산출물을 2026 GraphRAG 아카이브 스켈레톤에 맞춰 묶은 과목 프로필입니다.",
                "",
                "## Source inventory",
                "",
                f"- Markdown notes: {len(course.md_files)}",
        f"- PDFs: {len(course.pdf_files)}",
        f"- Code/artifacts: {len(course.artifacts)}",
        f"- Media/attachments: {len(course.media_files)}",
                "",
                "## 핵심 개념",
                "",
            ]
        )
        profile_lines.extend(f"- {item}" for item in concept_links)
        profile_lines.extend(["", "## 근거", "", f"- {evidence_link}"])
        if source_note_links:
            profile_lines.extend(["", "## 대표 노트", ""])
            profile_lines.extend(f"- {item}" for item in source_note_links)
        if source_pdf_links:
            profile_lines.extend(["", "## 대표 PDF", ""])
            profile_lines.extend(f"- {item}" for item in source_pdf_links)
        if artifact_links:
            profile_lines.extend(["", "## 대표 코드/산출물", ""])
            profile_lines.extend(f"- {item}" for item in artifact_links)
        media_links = first_links(course.media_files, 20)
        if media_links:
            profile_lines.extend(["", "## 대표 미디어/첨부", ""])
            profile_lines.extend(f"- {item}" for item in media_links)
        updated += int(upsert_text_file(course_profile_path(course), "\n".join(profile_lines)))

        evidence_lines = note_header(f"{course.label} 근거 인덱스", ["type/interface", "pkm/kg-evidence"])
        evidence_lines.extend(
            [
                f"kg_parent:: {wikilink(course_profile_path(course), course.label)}",
                f"kg_skeleton:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
                f"kg_course:: {wikilink(course_profile_path(course), course.label)}",
                "kg_concepts:: " + ", ".join(concept_links),
                "",
                f"# {course.label} 근거 인덱스",
                "",
                "이 노드는 과목 지식그래프가 참조하는 로컬 근거 파일을 모읍니다.",
                "",
                "## Markdown notes",
                "",
            ]
        )
        evidence_lines.extend(f"- {wikilink(path)}" for path in course.md_files)
        evidence_lines.extend(["", "## PDFs", ""])
        evidence_lines.extend(f"- {wikilink(path)}" for path in course.pdf_files)
        evidence_lines.extend(["", "## Code and artifacts", ""])
        evidence_lines.extend(f"- {wikilink(path)}" for path in course.artifacts)
        evidence_lines.extend(["", "## Media and attachments", ""])
        evidence_lines.extend(f"- {wikilink(path)}" for path in course.media_files)
        updated += int(upsert_text_file(course_evidence_path(course), "\n".join(evidence_lines)))

        for idx, concept in enumerate(course.concepts):
            sources = course.concept_sources.get(concept, [])[:8]
            local_related = [c for c in course.concepts if c != concept]
            my_key = concept_key(concept)
            my_tokens = relation_tokens(concept)
            cross_related: list[tuple[Course, str]] = []
            for other_course, other_concept, other_tokens, other_key in all_concepts:
                if other_course.key == course.key and other_concept == concept:
                    continue
                if other_course.key == course.key:
                    continue
                if my_key == other_key or (my_tokens and other_tokens and my_tokens & other_tokens):
                    cross_related.append((other_course, other_concept))
            related_links = [wikilink(concept_path(course, c), c) for c in local_related]
            related_links.extend(
                wikilink(concept_path(other_course, other_concept), f"{other_course.label}/{other_concept}")
                for other_course, other_concept in cross_related
            )
            concept_lines = note_header(concept, ["type/concept", "pkm/kg-concept"], note_type="concept")
            concept_lines.extend(
                [
                    f"kg_parent:: {wikilink(course_profile_path(course), course.label)}",
                    f"kg_skeleton:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
                    f"kg_course:: {wikilink(course_profile_path(course), course.label)}",
                    f"kg_evidence:: {evidence_link}",
                    f"kg_community:: {wikilink(community_path(course))}",
                    "related:: " + ", ".join(related_links),
                    "",
                    f"# {concept}",
                    "",
                    f"{course.label} 자료에서 추출된 개념 노드입니다.",
                    "",
                    "## 로컬 근거",
                    "",
                ]
            )
            if sources:
                concept_lines.extend(f"- {wikilink(src)}" for src in sources)
            else:
                concept_lines.append(f"- {evidence_link}")
            concept_lines.extend(["", "## 연결 개념", ""])
            concept_lines.extend(f"- {item}" for item in related_links)
            updated += int(upsert_text_file(concept_path(course, concept), "\n".join(concept_lines)))

    # Community reports.
    for domain in DOMAINS:
        courses_in_domain = sorted(by_domain.get(domain.key, []), key=lambda c: c.label)
        title = domain.title.replace(" 인터페이스", " 커뮤니티")
        write_community_report(title, f"{COMMUNITY_ROOT}/{slugify(title)}.md", courses_in_domain, domain.summary)
        updated += 1
    extra_groups = [
        ("외부 프로그램 커뮤니티", f"{COMMUNITY_ROOT}/외부 프로그램 커뮤니티.md", by_domain.get("external", []), "LG Aimers 등 외부 프로그램 자료를 전공 아카이브와 연결합니다."),
        ("자격증 검증 커뮤니티", f"{COMMUNITY_ROOT}/자격증 검증 커뮤니티.md", by_domain.get("certification", []), "자격증 필기/실기 근거를 전공 지식과 검증 모드로 연결합니다."),
        ("공유 미디어 커뮤니티", f"{COMMUNITY_ROOT}/공유 미디어 커뮤니티.md", by_domain.get("media", []), "image 폴더의 PNG, SVG, JPG, MP4, ZIP 등 공유 첨부자료를 과목 근거와 연결합니다."),
        ("아카이브 운영 커뮤니티", f"{COMMUNITY_ROOT}/아카이브 운영 커뮤니티.md", by_domain.get("operations", []), "루트 문서, 템플릿, canvas 등 vault 운영 파일을 지식그래프 운영 맥락으로 연결합니다."),
    ]
    for title, path, group, summary in extra_groups:
        write_community_report(title, path, sorted(group, key=lambda c: c.label), summary)
        updated += 1
    return updated


def write_coverage_report(courses: dict[str, Course]) -> bool:
    covered: set[str] = set()
    course_rows: list[str] = []
    for course in sorted(courses.values(), key=lambda c: c.label):
        course_files = course.md_files + course.pdf_files + course.artifacts + course.media_files
        for path in course_files:
            covered.add(rel(path))
        course_rows.append(
            "| "
            + " | ".join(
                [
                    course.label,
                    str(len(course.md_files)),
                    str(len(course.pdf_files)),
                    str(len(course.artifacts)),
                    str(len(course.media_files)),
                    str(len(course.concepts)),
                    wikilink(course_evidence_path(course), "근거"),
                ]
            )
            + " |"
        )

    suffix_counts: Counter[str] = Counter()
    for item in covered:
        suffix_counts[Path(item).suffix.lower() or "<none>"] += 1

    unmanaged: list[str] = []
    for path in iter_files():
        r = rel(path)
        if r.startswith(GENERATED_DIR + "/"):
            continue
        if r.startswith("ComputerScience/") and r.endswith("인터페이스.md"):
            continue
        if any(part in {".git", ".obsidian", ".agents", ".aioss-eval", ".gemini", "scripts", "__pycache__"} for part in path.parts):
            continue
        if r not in covered:
            unmanaged.append(r)

    lines = note_header("파일 커버리지 검증 리포트", ["type/interface", "pkm/kg-evidence"])
    lines.extend(
        [
            f"kg_skeleton:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
            "",
            "# 파일 커버리지 검증 리포트",
            "",
            "이 리포트는 generated archive-kg와 Obsidian/tooling 내부 파일을 제외한 vault 자료가 과목/활동 evidence layer에 들어갔는지 검증합니다.",
            "",
            "## 전체 커버리지",
            "",
            f"- Covered files: {len(covered)}",
            f"- Unmanaged source files: {len(unmanaged)}",
            "- Excluded operational scopes: hidden/system files, `.git`, `.obsidian`, `.agents`, `.aioss-eval`, `.gemini`, `scripts`, generated `ComputerScience/00_graph-interfaces`, generated field interface notes",
            "",
            "## 확장자별 covered files",
            "",
        ]
    )
    for suffix, count in suffix_counts.most_common():
        lines.append(f"- `{suffix}`: {count}")
    lines.extend(
        [
            "",
            "## 과목별 evidence coverage",
            "",
            "| Course | md | pdf | artifacts/text/code | media | concepts | evidence |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    lines.extend(course_rows)
    lines.extend(["", "## Unmanaged source files", ""])
    if unmanaged:
        lines.extend(f"- `{item}`" for item in unmanaged[:300])
        if len(unmanaged) > 300:
            lines.append(f"- ... {len(unmanaged) - 300} more")
    else:
        lines.append("- 없음")
    return upsert_text_file(COVERAGE_REPORT, "\n".join(lines))


def write_extraction_report(courses: dict[str, Course]) -> bool:
    totals: Counter[str] = Counter()
    rows: list[str] = []
    for course in sorted(courses.values(), key=lambda c: c.label):
        totals.update(course.extraction_stats)
        rows.append(
            "| "
            + " | ".join(
                [
                    course.label,
                    str(course.extraction_stats.get("markdown_full_files", 0)),
                    str(course.extraction_stats.get("pdf_full_files", 0)),
                    str(course.extraction_stats.get("pdf_text_empty_or_failed", 0)),
                    str(course.extraction_stats.get("textlike_full_files", 0)),
                    str(course.extraction_stats.get("office_xml_text_files", 0)),
                    str(course.extraction_stats.get("media_filename_files", 0)),
                ]
            )
            + " |"
        )
    lines = note_header("전문 내용 추출 검증 리포트", ["type/interface", "pkm/kg-evidence"])
    lines.extend(
        [
            f"kg_parent:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
            f"kg_skeleton:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
            "",
            "# 전문 내용 추출 검증 리포트",
            "",
            "확장자별 분류만 하지 않고, 실제 텍스트 추출 가능한 파일은 전문을 읽어 개념 후보와 rename 후보를 계산했습니다.",
            "",
            "## 전체 추출량",
            "",
            f"- Markdown full-body files: {totals.get('markdown_full_files', 0)}",
            f"- Markdown characters: {totals.get('markdown_chars', 0)}",
            f"- PDF full-text files: {totals.get('pdf_full_files', 0)}",
            f"- PDF extracted characters: {totals.get('pdf_chars', 0)}",
            f"- PDF empty/failed text extraction: {totals.get('pdf_text_empty_or_failed', 0)}",
            f"- Text/code/notebook full-text files: {totals.get('textlike_full_files', 0)}",
            f"- Text/code/notebook characters: {totals.get('textlike_chars', 0)}",
            f"- Office XML text files: {totals.get('office_xml_text_files', 0)}",
            f"- Office XML characters: {totals.get('office_xml_chars', 0)}",
            f"- Media filename/OCR-context files: {totals.get('media_filename_files', 0)}",
            "",
            "## 과목별 추출 검증",
            "",
            "| Course | md full | pdf full | pdf empty | text/code full | office xml | media ctx |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(rows)
    return upsert_text_file(EXTRACTION_REPORT, "\n".join(lines))


def protected_rename_rels() -> set[str]:
    protected = set(ROOT_SUPPORT)
    protected.update(anchor.rel for domain in DOMAINS for anchor in domain.courses)
    return {nfc(item) for item in protected}


def is_generic_stem(stem: str) -> bool:
    cleaned = clean_phrase(stem)
    if not cleaned:
        return False
    return bool(GENERIC_STEM_RE.fullmatch(cleaned)) or cleaned.lower() in STOP_PHRASES


def unique_destination(path: Path, desired_stem: str) -> Path:
    desired_stem = slugify(desired_stem, path.stem)
    candidate = path.with_name(f"{desired_stem}{path.suffix}")
    if candidate == path:
        return candidate
    idx = 2
    while candidate.exists():
        candidate = path.with_name(f"{desired_stem} {idx}{path.suffix}")
        idx += 1
    return candidate


def rename_title_for_generic(path: Path) -> str:
    title = content_title_for_path(path)
    if title:
        return title
    parent = clean_phrase(path.parent.name)
    stem = clean_phrase(path.stem)
    if parent and keep_phrase(parent):
        if "풀이" in stem:
            return f"{parent} 문제 풀이"
        if "문제" in stem:
            return f"{parent} 문제"
        return f"{parent} {stem}".strip()
    return ""


def wikilink_target_for(path: Path) -> str:
    r = rel(path)
    return r.removesuffix(".md") if path.suffix.lower() == ".md" else r


def target_matches_old_path(target: str, source: Path, old_path: Path) -> bool:
    clean = nfc(target).replace("%20", " ").strip()
    clean = clean.split("#", 1)[0].split("|", 1)[0].strip()
    old_r = rel(old_path)
    old_no_ext = old_r.removesuffix(".md")
    if clean in {old_r, old_no_ext}:
        return True
    try:
        relative = (source.parent / clean).resolve()
        if relative == old_path.resolve():
            return True
    except Exception:
        pass
    if clean in {old_path.name, old_path.stem} and source.parent == old_path.parent:
        return True
    return False


def rewrite_links_for_rename(old_path: Path, new_path: Path) -> int:
    changed = 0
    new_target = wikilink_target_for(new_path)
    new_rel = rel(new_path)
    old_rel = rel(old_path)
    for source in iter_files():
        if source.suffix.lower() not in {".md", ".canvas", ".json"}:
            continue
        try:
            text = nfc(source.read_text(encoding="utf-8"))
        except Exception:
            continue

        def wiki_repl(match: re.Match[str]) -> str:
            target, section, alias = match.group(1), match.group(2) or "", match.group(3) or ""
            if target_matches_old_path(target, source, old_path):
                return f"[[{new_target}{section}{alias}]]"
            return match.group(0)

        def md_repl(match: re.Match[str]) -> str:
            target = match.group(2)
            if target_matches_old_path(target, source, old_path):
                return f"{match.group(1)}{new_rel}{match.group(3) or ''}{match.group(4)}"
            return match.group(0)

        new = WIKILINK_FULL_RE.sub(wiki_repl, text)
        new = MD_LINK_RE.sub(md_repl, new)
        new = new.replace(old_rel, new_rel)
        if new != text:
            source.write_text(nfc(new), encoding="utf-8")
            changed += 1
    return changed


def apply_content_based_renames() -> dict[str, object]:
    protected = protected_rename_rels()
    applied: list[str] = []
    deferred: list[str] = []
    rewrites = 0
    for path in iter_files():
        if not path.exists():
            continue
        r = rel(path)
        if r.startswith(GENERATED_DIR + "/") or r in protected:
            continue
        if path.suffix.lower() not in {".md", ".pdf", ".txt", ".ipynb", ".docx", ".pptx", ".xlsx"}:
            continue
        if not is_generic_stem(path.stem):
            continue
        title = rename_title_for_generic(path)
        if not title or not keep_phrase(title):
            deferred.append(f"{r} -> 내용 기반 제목 부족")
            continue
        destination = unique_destination(path, title)
        if destination == path:
            continue
        rewrites += rewrite_links_for_rename(path, destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)
        applied.append(f"{r} -> {rel(destination)}")
    return {"applied": applied, "deferred": deferred, "rewrites": rewrites}


def write_rename_report(result: dict[str, object]) -> bool:
    applied = list(result.get("applied", []))
    deferred = list(result.get("deferred", []))
    lines = note_header("내용 기반 파일명 정합성 리포트", ["type/interface", "pkm/kg-evidence"])
    lines.extend(
        [
            f"kg_parent:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
            f"kg_skeleton:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
            "",
            "# 내용 기반 파일명 정합성 리포트",
            "",
            "generic 파일명은 frontmatter/H1/PDF 전문 첫 제목/텍스트 전문 첫 제목을 기준으로 rename했습니다. 링크는 Obsidian wikilink와 markdown link를 함께 재작성했습니다.",
            "",
            "## Summary",
            "",
            f"- Applied renames: {len(applied)}",
            f"- Link rewrite passes: {result.get('rewrites', 0)}",
            f"- Deferred candidates: {len(deferred)}",
            "",
            "## Applied",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in applied)
    lines.extend(["", "## Deferred", ""])
    if deferred:
        lines.extend(f"- `{item}`" for item in deferred)
    else:
        lines.append("- 없음")
    return upsert_text_file(RENAME_REPORT, "\n".join(lines))


def write_community_report(title: str, path: str, courses: list[Course], summary: str) -> None:
    course_links = [wikilink(course_profile_path(course), course.label) for course in courses]
    concept_links: list[str] = []
    evidence_links: list[str] = []
    for course in courses:
        concept_links.extend(wikilink(concept_path(course, c), c) for c in course.concepts[:5])
        evidence_links.append(wikilink(course_evidence_path(course), f"{course.label} 근거"))
    lines = note_header(title, ["type/interface", "pkm/kg-community"])
    lines.extend(
        [
            f"kg_parent:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
            f"kg_skeleton:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
            "kg_courses:: " + ", ".join(course_links),
            "kg_concepts:: " + ", ".join(concept_links[:40]),
            "kg_evidence:: " + ", ".join(evidence_links),
            "",
            f"# {title}",
            "",
            summary,
            "",
            "## 과목 프로필",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in course_links)
    lines.extend(["", "## 대표 개념", ""])
    lines.extend(f"- {item}" for item in concept_links[:40])
    lines.extend(["", "## 근거 인덱스", ""])
    lines.extend(f"- {item}" for item in evidence_links)
    upsert_text_file(path, "\n".join(lines))


def concepts_for_note(course: Course, path: Path, body: str, fm: dict) -> list[str]:
    haystack = "\n".join([str(fm.get("title") or ""), clean_phrase(path.stem), body]).lower()
    picked: list[str] = []
    for concept in course.concepts:
        parts = [p.lower() for p in re.split(r"\s+", concept) if len(p) >= 2]
        if concept.lower() in haystack or any(p in haystack for p in parts if p not in STOP_PHRASES):
            picked.append(concept)
        if len(picked) >= 5:
            break
    if not picked:
        picked = course.concepts[:3]
    return picked


def update_source_notes(courses: dict[str, Course]) -> tuple[int, int]:
    changed = 0
    cleaned = 0
    for path in iter_files():
        if path.suffix.lower() != ".md":
            continue
        r = rel(path)
        if r.startswith(GENERATED_DIR + "/"):
            continue
        text = nfc(path.read_text(encoding="utf-8"))
        fm_text, body = split_frontmatter(text)
        cleaned_body = strip_old_fields(body)
        if cleaned_body != body:
            cleaned += 1
        body = cleaned_body

        course = None if path.name in ROOT_SUPPORT else find_course_for_path(path, courses)
        fields: list[tuple[str, str]] = []
        if course:
            try:
                fm, note_body = read_note(path)
            except Exception:
                fm, note_body = {}, body
            concepts = concepts_for_note(course, path, note_body, fm)
            concept_links = [wikilink(concept_path(course, c), c) for c in concepts]
            query_links = [wikilink(QUERY_MODES[k][1], QUERY_MODES[k][0]) for k in course.query_modes]
            fields = [
                ("kg_parent", wikilink(course_profile_path(course), course.label)),
                ("kg_profile", wikilink(course_profile_path(course), course.label)),
                ("kg_evidence", wikilink(course_evidence_path(course), f"{course.label} 근거 인덱스")),
                ("kg_concepts", ", ".join(concept_links)),
                ("kg_query_mode", ", ".join(query_links)),
            ]
        body = insert_field_block(body, fields)
        out = fm_text + body
        if not out.endswith("\n"):
            out += "\n"
        if out != text:
            path.write_text(nfc(out), encoding="utf-8")
            changed += 1
    return changed, cleaned


def clean_excluded_note_fields() -> int:
    changed = 0
    excluded_parts = {".aioss-eval", ".gemini"}
    for dirpath, dirnames, filenames in os.walk(VAULT):
        dirnames[:] = [d for d in dirnames if d not in {".git", ".obsidian", ".agents", "scripts", "__pycache__"}]
        if not any(part in excluded_parts for part in Path(dirpath).parts):
            continue
        for fn in filenames:
            if not fn.endswith(".md"):
                continue
            path = Path(dirpath) / fn
            text = nfc(path.read_text(encoding="utf-8"))
            fm_text, body = split_frontmatter(text)
            body = strip_old_fields(body)
            out = fm_text + body
            if not out.endswith("\n"):
                out += "\n"
            if out != text:
                path.write_text(nfc(out), encoding="utf-8")
                changed += 1
    return changed


def clean_deleted_links() -> int:
    patterns = [
        r",?\s*\[\[ComputerScience/00_graph-interfaces/(?:ontology|research|tech-stacks|ecosystems|competencies)/[^\]]+\]\]",
        r",?\s*\[\[ComputerScience/00_graph-interfaces/(?:ontology|research|tech-stacks|ecosystems|competencies)/[^|\]]+\|[^\]]+\]\]",
    ]
    changed = 0
    for path in iter_files():
        if path.suffix.lower() != ".md":
            continue
        text = nfc(path.read_text(encoding="utf-8"))
        new = text
        for pat in patterns:
            new = re.sub(pat, "", new)
        new = re.sub(r"::\s*,\s*", ":: ", new)
        new = re.sub(r",\s*,", ",", new)
        new = re.sub(r"\n{3,}", "\n\n", new)
        if new != text:
            path.write_text(nfc(new), encoding="utf-8")
            changed += 1
    return changed


def update_graph_json() -> bool:
    path = VAULT / ".obsidian/graph.json"
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    keep = []
    remove_queries = {
        "tag:#pkm/schema",
        "tag:#pkm/relation",
        "tag:#pkm/model",
        "tag:#pkm/research",
        "tag:#pkm/tech-stack",
        "tag:#pkm/ecosystem",
        "tag:#pkm/competency",
    }
    for group in data.get("colorGroups", []):
        if group.get("query") not in remove_queries:
            keep.append(group)
    new_groups = [
        ("tag:#pkm/kg-skeleton", 0x5B5BD6),
        ("tag:#pkm/kg-method-2026", 0x8B5CF6),
        ("tag:#pkm/kg-community", 0x0F766E),
        ("tag:#pkm/kg-course", 0x2563EB),
        ("tag:#pkm/kg-concept", 0xF59E0B),
        ("tag:#pkm/kg-evidence", 0xDC2626),
        ("tag:#pkm/kg-query", 0x16A34A),
    ]
    existing = {group.get("query") for group in keep}
    for query, rgb in new_groups:
        if query not in existing:
            keep.insert(0, {"query": query, "color": {"a": 1, "rgb": rgb}})
    data["colorGroups"] = keep
    data["showTags"] = True
    data["showAttachments"] = True
    data["showArrow"] = True
    data["nodeSizeMultiplier"] = max(1.3, float(data.get("nodeSizeMultiplier", 1)))
    data["lineSizeMultiplier"] = max(1.45, float(data.get("lineSizeMultiplier", 1)))
    out = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    old = path.read_text(encoding="utf-8")
    if old != out:
        path.write_text(out, encoding="utf-8")
        return True
    return False


def create_canvas(courses: dict[str, Course]) -> bool:
    nodes: list[dict] = []
    edges: list[dict] = []
    nid = 0

    def add_file(file: str, x: int, y: int, w: int = 260, h: int = 90, color: str = "1") -> str:
        nonlocal nid
        node_id = f"n{nid}"
        nid += 1
        nodes.append({"id": node_id, "type": "file", "file": file, "x": x, "y": y, "width": w, "height": h, "color": color})
        return node_id

    def add_group(label: str, x: int, y: int, w: int, h: int, color: str) -> str:
        nonlocal nid
        node_id = f"g{nid}"
        nid += 1
        nodes.append({"id": node_id, "type": "group", "x": x, "y": y, "width": w, "height": h, "color": color, "label": label})
        return node_id

    def add_edge(a: str, b: str, label: str = "") -> None:
        nonlocal nid
        edge = {"id": f"e{nid}", "fromNode": a, "fromSide": "bottom", "toNode": b, "toSide": "top"}
        nid += 1
        if label:
            edge["label"] = label
        edges.append(edge)

    add_group("2026 method layer", -960, -760, 1920, 230, "6")
    skeleton_id = add_file(SKELETON, -160, -700, 320, 110, "6")
    method_ids = []
    for i, (_title, path, *_rest) in enumerate(METHOD_SOURCES[:6]):
        x = -900 + i * 300
        method_id = add_file(path, x, -560, 260, 90, "6")
        method_ids.append(method_id)
        add_edge(skeleton_id, method_id, "method")

    add_group("GraphRAG-Bench query modes", -700, -400, 1400, 200, "4")
    query_ids: dict[str, str] = {}
    for i, (key, (_title, path, _summary)) in enumerate(QUERY_MODES.items()):
        qid = add_file(path, -600 + i * 400, -330, 280, 90, "4")
        query_ids[key] = qid
        add_edge(skeleton_id, qid, "query")

    add_group("community reports", -1260, -110, 2520, 260, "2")
    community_ids: dict[str, str] = {}
    all_domains = [(d.key, d.title.replace(" 인터페이스", " 커뮤니티"), f"{COMMUNITY_ROOT}/{slugify(d.title.replace(' 인터페이스', ' 커뮤니티'))}.md") for d in DOMAINS]
    all_domains += [
        ("external", "외부 프로그램 커뮤니티", f"{COMMUNITY_ROOT}/외부 프로그램 커뮤니티.md"),
        ("certification", "자격증 검증 커뮤니티", f"{COMMUNITY_ROOT}/자격증 검증 커뮤니티.md"),
        ("media", "공유 미디어 커뮤니티", f"{COMMUNITY_ROOT}/공유 미디어 커뮤니티.md"),
        ("operations", "아카이브 운영 커뮤니티", f"{COMMUNITY_ROOT}/아카이브 운영 커뮤니티.md"),
    ]
    for i, (key, _title, path) in enumerate(all_domains):
        x = -1170 + (i % 5) * 480
        y = -40 + (i // 5) * 115
        cid = add_file(path, x, y, 360, 85, "2")
        community_ids[key] = cid
        add_edge(skeleton_id, cid, "community")

    add_group("course profiles and local evidence", -1540, 250, 3080, 980, "1")
    sorted_courses = sorted(courses.values(), key=lambda c: (c.domain_key, c.label))
    course_ids: dict[str, str] = {}
    evidence_ids: dict[str, str] = {}
    for i, course in enumerate(sorted_courses):
        x = -1480 + (i % 6) * 500
        y = 330 + (i // 6) * 160
        cid = add_file(course_profile_path(course), x, y, 280, 82, "1")
        eid = add_file(course_evidence_path(course), x + 295, y, 170, 82, "3")
        course_ids[course.key] = cid
        evidence_ids[course.key] = eid
        if course.domain_key in community_ids:
            add_edge(community_ids[course.domain_key], cid, "course")
        add_edge(cid, eid, "evidence")
        for mode in course.query_modes[:2]:
            add_edge(query_ids[mode], cid, "mode")

    add_group("sample concept codebook", -1540, 1300, 3080, 520, "3")
    for i, course in enumerate(sorted_courses[:30]):
        if not course.concepts:
            continue
        x = -1480 + (i % 6) * 500
        y = 1380 + (i // 6) * 92
        concept = course.concepts[0]
        kid = add_file(concept_path(course, concept), x, y, 360, 72, "3")
        add_edge(course_ids[course.key], kid, "concept")

    canvas = {"nodes": nodes, "edges": edges}
    return upsert_text_file("2026 GraphRAG 아카이브.canvas", json.dumps(canvas, ensure_ascii=False, indent=2))


def canvas_missing_files(canvas_path: str) -> int:
    data = json.loads((VAULT / canvas_path).read_text(encoding="utf-8"))
    missing = 0
    for node in data.get("nodes", []):
        if node.get("type") == "file" and not (VAULT / node.get("file", "")).exists():
            missing += 1
    return missing


def main() -> None:
    rename_result = apply_content_based_renames()
    courses = build_courses()
    collect_sources(courses)
    extract_concepts(courses)

    removed_dirs = remove_old_generated_research_dirs()
    reset_dirs = reset_managed_archive_dirs()
    method_updates = write_method_layer(courses)
    course_updates = write_course_layers(courses)
    coverage_updated = write_coverage_report(courses)
    extraction_updated = write_extraction_report(courses)
    rename_report_updated = write_rename_report(rename_result)
    note_updates, cleaned_notes = update_source_notes(courses)
    excluded_cleanups = clean_excluded_note_fields()
    deleted_link_cleanups = clean_deleted_links()
    graph_updated = update_graph_json()
    canvas_updated = create_canvas(courses)
    missing_canvas = canvas_missing_files("2026 GraphRAG 아카이브.canvas")

    md_total = sum(len(c.md_files) for c in courses.values())
    pdf_total = sum(len(c.pdf_files) for c in courses.values())
    artifact_total = sum(len(c.artifacts) for c in courses.values())
    concept_total = sum(len(c.concepts) for c in courses.values())
    media_total = sum(len(c.media_files) for c in courses.values())
    print(
        "graphrag_2026_archive_migration "
        f"courses={len(courses)} md={md_total} pdf={pdf_total} artifacts={artifact_total} "
        f"media={media_total} concepts={concept_total} removed_old_dirs={removed_dirs} method_updates={method_updates} "
        f"reset_archive_dirs={reset_dirs} "
        f"course_layer_updates={course_updates} source_note_updates={note_updates} "
        f"cleaned_old_broad_fields={cleaned_notes} excluded_note_cleanups={excluded_cleanups} "
        f"deleted_link_cleanups={deleted_link_cleanups} "
        f"content_renames={len(rename_result.get('applied', []))} rename_link_rewrites={rename_result.get('rewrites', 0)} "
        f"graph_json_updated={int(graph_updated)} canvas_updated={int(canvas_updated)} "
        f"coverage_report_updated={int(coverage_updated)} extraction_report_updated={int(extraction_updated)} "
        f"rename_report_updated={int(rename_report_updated)} "
        f"canvas_missing_files={missing_canvas}"
    )


if __name__ == "__main__":
    main()
