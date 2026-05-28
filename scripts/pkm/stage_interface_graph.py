#!/usr/bin/env python3
"""Add stage, module, and bridge interface nodes for Obsidian Graph View.

Folders do not become graph nodes in Obsidian. This script creates explicit
interface notes for the curriculum hierarchy and wires ordinary notes to those
notes with inline wikilink properties.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from interface_graph_wiring import DOMAINS, VAULT, link, nfc, split_frontmatter, write_note

TODAY = "2026-05-28"
INTERFACE_ROOT = "ComputerScience/00_graph-interfaces"
HUB = f"{INTERFACE_ROOT}/지식그래프 허브.md"

EXCLUDE_DIRS = {
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

INLINE_MANAGED_RE = re.compile(r"^(?:graph|stage|module|bridge)::\s*.*\n?", re.MULTILINE)
TOP_INLINE_RE = re.compile(r"\A((?:(?:[A-Za-z가-힣_-]+)::.*\n)+)\n?")
FRONTMATTER_TYPE_RE = re.compile(r"^type:\s*['\"]?interface['\"]?\s*$", re.MULTILINE)
COLOR_BY_DOMAIN_KEY = {
    "01_programming-foundations": "4",
    "02_math-theory": "5",
    "03_ai-ml-data": "1",
    "04_systems-infrastructure": "2",
    "05_software-engineering": "3",
    "06_algorithms-graphics": "6",
    "07_professional-humanities": "5",
}


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    path: str
    summary: str
    courses: tuple[str, ...]


@dataclass(frozen=True)
class Bridge:
    key: str
    title: str
    path: str
    summary: str
    courses: tuple[str, ...]


@dataclass(frozen=True)
class Module:
    course: str
    label: str
    path: str
    anchor_rel: str
    domain_key: str


STAGES: tuple[Stage, ...] = (
    Stage(
        "stage-1",
        "1단계 기초 구축 인터페이스",
        f"{INTERFACE_ROOT}/stages/1단계 기초 구축 인터페이스.md",
        "프로그래밍 문법, Linux 사용, 확률/이산수학처럼 이후 전공 지식을 받쳐 주는 기초 단계입니다.",
        (
            "coding-basics",
            "python-programming",
            "java-programming",
            "linux",
            "probability-statistics",
            "discrete-mathematics",
        ),
    ),
    Stage(
        "stage-2",
        "2단계 전공 핵심 인터페이스",
        f"{INTERFACE_ROOT}/stages/2단계 전공 핵심 인터페이스.md",
        "자료구조, 컴퓨터구조, 운영체제, 데이터베이스, 웹, OSS, 언어론으로 CS 핵심 체계를 형성합니다.",
        (
            "data-structures",
            "computer-architecture",
            "operating-systems",
            "database-systems",
            "web-programming",
            "open-source-software",
            "programming-languages",
            "mathematical-logic",
        ),
    ),
    Stage(
        "stage-3",
        "3단계 AI 데이터 심화 인터페이스",
        f"{INTERFACE_ROOT}/stages/3단계 AI 데이터 심화 인터페이스.md",
        "AI, 머신러닝, 최적화, 신경망, 빅데이터, LLM을 이론과 데이터 실습으로 확장합니다.",
        (
            "artificial-intelligence",
            "machine-learning",
            "optimization-math",
            "neural-networks",
            "big-data-analysis",
            "large-language-models",
        ),
    ),
    Stage(
        "stage-4",
        "4단계 시스템 실전 인터페이스",
        f"{INTERFACE_ROOT}/stages/4단계 시스템 실전 인터페이스.md",
        "네트워크, 분산/병렬처리, 컨테이너, 알고리즘, 그래픽스, 코딩테스트를 실전 문제 해결로 묶습니다.",
        (
            "computer-networks",
            "parallel-distributed-computing",
            "container-orchestration",
            "algorithm-design-analysis",
            "computer-graphics",
            "coding-test",
        ),
    ),
    Stage(
        "stage-5",
        "5단계 통합 프로젝트 인터페이스",
        f"{INTERFACE_ROOT}/stages/5단계 통합 프로젝트 인터페이스.md",
        "ML 프로젝트, AI 시스템 설계, 컴퓨터비전, AIOSS, 생성형 AI, LG Aimers를 산출물 중심으로 연결합니다.",
        (
            "ml-projects",
            "ai-system-design",
            "computer-vision",
            "aioss-open-source-delivery",
            "generative-ai-fine-tuning",
            "LGAimer",
        ),
    ),
    Stage(
        "stage-6",
        "6단계 전문 확장 인터페이스",
        f"{INTERFACE_ROOT}/stages/6단계 전문 확장 인터페이스.md",
        "지식재산, 글쓰기, 고전 읽기, 포트폴리오, 자격증을 전공 산출물의 설명 역량과 연결합니다.",
        (
            "intellectual-property",
            "creative-writing",
            "classics-reading",
            "degree-portfolio",
            "certifications",
        ),
    ),
)

BRIDGES: tuple[Bridge, ...] = (
    Bridge(
        "ai-implementation",
        "AI 구현 브리지",
        f"{INTERFACE_ROOT}/bridges/AI 구현 브리지.md",
        "Python 구현력에서 AI, ML, 신경망, CV, LLM, 생성형 AI 산출물까지 이어지는 구현 축입니다.",
        (
            "python-programming",
            "artificial-intelligence",
            "machine-learning",
            "neural-networks",
            "computer-vision",
            "large-language-models",
            "generative-ai-fine-tuning",
            "ml-projects",
            "ai-system-design",
            "LGAimer",
        ),
    ),
    Bridge(
        "data-service",
        "데이터 서비스 브리지",
        f"{INTERFACE_ROOT}/bridges/데이터 서비스 브리지.md",
        "확률통계, 데이터베이스, 빅데이터, ML 프로젝트, AI 시스템 설계를 서비스 데이터 흐름으로 연결합니다.",
        (
            "probability-statistics",
            "database-systems",
            "big-data-analysis",
            "ml-projects",
            "ai-system-design",
            "large-language-models",
            "web-programming",
        ),
    ),
    Bridge(
        "system-operations",
        "시스템 운영 브리지",
        f"{INTERFACE_ROOT}/bridges/시스템 운영 브리지.md",
        "Linux, 구조, OS, 네트워크, 분산처리, 컨테이너, AIOSS를 운영 가능한 시스템 관점으로 묶습니다.",
        (
            "linux",
            "computer-architecture",
            "operating-systems",
            "computer-networks",
            "parallel-distributed-computing",
            "container-orchestration",
            "aioss-open-source-delivery",
            "certifications",
        ),
    ),
    Bridge(
        "math-algorithm",
        "수학 알고리즘 브리지",
        f"{INTERFACE_ROOT}/bridges/수학 알고리즘 브리지.md",
        "이산수학, 확률통계, 최적화, 자료구조, 알고리즘, 그래픽스, CV, ML을 분석 도구로 연결합니다.",
        (
            "discrete-mathematics",
            "probability-statistics",
            "optimization-math",
            "data-structures",
            "algorithm-design-analysis",
            "coding-test",
            "computer-graphics",
            "computer-vision",
            "machine-learning",
        ),
    ),
    Bridge(
        "open-source-delivery",
        "오픈소스 delivery 브리지",
        f"{INTERFACE_ROOT}/bridges/오픈소스 delivery 브리지.md",
        "웹, OSS, 언어론, DB, 컨테이너, AIOSS, LG Aimers를 협업과 delivery workflow로 연결합니다.",
        (
            "open-source-software",
            "programming-languages",
            "web-programming",
            "database-systems",
            "container-orchestration",
            "aioss-open-source-delivery",
            "LGAimer",
        ),
    ),
    Bridge(
        "portfolio-output",
        "산출물 포트폴리오 브리지",
        f"{INTERFACE_ROOT}/bridges/산출물 포트폴리오 브리지.md",
        "전공 프로젝트, 글쓰기, 지식재산, 자격증, 대외활동을 설명 가능한 포트폴리오 산출물로 연결합니다.",
        (
            "creative-writing",
            "intellectual-property",
            "degree-portfolio",
            "classics-reading",
            "ai-system-design",
            "generative-ai-fine-tuning",
            "aioss-open-source-delivery",
            "LGAimer",
            "certifications",
            "ml-projects",
        ),
    ),
    Bridge(
        "archive-operations",
        "아카이브 운영 브리지",
        f"{INTERFACE_ROOT}/bridges/아카이브 운영 브리지.md",
        "README, 에이전트 지침, 커리큘럼 관계 문서, 그래프/Canvas를 아카이브 운영 계층으로 연결합니다.",
        ("archive-operations",),
    ),
)

ROOT_SUPPORT = {
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "커리큘럼 관계 정리.md",
}


def rel(path: Path) -> str:
    return path.relative_to(VAULT).as_posix()


def wiki(rel_path: str, alias: str | None = None) -> str:
    return link(rel_path, alias)


def iter_notes() -> list[Path]:
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(VAULT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        dirpath = unicodedata.normalize("NFC", dirpath)
        for filename in filenames:
            filename = unicodedata.normalize("NFC", filename)
            if filename.endswith(".md"):
                paths.append(Path(dirpath) / filename)
    return sorted(paths)


def is_interface_note(path: Path) -> bool:
    text = nfc(path.read_text(encoding="utf-8"))
    if path.relative_to(VAULT).as_posix().startswith(INTERFACE_ROOT + "/"):
        return True
    return bool(FRONTMATTER_TYPE_RE.search(text))


def classify_course(path: Path) -> str | None:
    rel_path = path.relative_to(VAULT).as_posix()
    parts = Path(rel_path).parts
    if rel_path in ROOT_SUPPORT:
        return "archive-operations"
    if not parts:
        return None
    if parts[0] == "LGAimer":
        return "LGAimer"
    if parts[0] == "certifications":
        return "certifications"
    if parts[0] == "ComputerScience":
        if len(parts) >= 2 and parts[1] == "00_graph-interfaces":
            return None
        if len(parts) >= 3 and parts[2].endswith(".md"):
            return None
        if len(parts) >= 3:
            return parts[2]
    return None


def build_modules() -> dict[str, Module]:
    modules: dict[str, Module] = {}
    for domain in DOMAINS:
        for course in domain.courses:
            course_key = Path(course.rel).parts[2] if course.rel.startswith("ComputerScience/") else course.label
            modules[course_key] = Module(
                course=course_key,
                label=course.label,
                path=f"{INTERFACE_ROOT}/courses/{course.label} 인터페이스.md",
                anchor_rel=course.rel,
                domain_key=domain.key,
            )
    modules["LGAimer"] = Module(
        "LGAimer",
        "LG Aimers",
        f"{INTERFACE_ROOT}/courses/LG Aimers 인터페이스.md",
        "LGAimer/LG Aimers 9기 지원서 초안.md",
        "ai",
    )
    modules["certifications"] = Module(
        "certifications",
        "자격증",
        f"{INTERFACE_ROOT}/courses/자격증 인터페이스.md",
        "certifications/체크리스트.md",
        "software",
    )
    return modules


def course_to_stage() -> dict[str, Stage]:
    result: dict[str, Stage] = {}
    for stage in STAGES:
        for course in stage.courses:
            result[course] = stage
    return result


def course_to_bridges() -> dict[str, list[Bridge]]:
    result: dict[str, list[Bridge]] = defaultdict(list)
    for bridge in BRIDGES:
        for course in bridge.courses:
            result[course].append(bridge)
    return result


def domain_by_key() -> dict[str, object]:
    return {domain.key: domain for domain in DOMAINS}


def field_value(paths: list[tuple[str, str]]) -> str:
    return ", ".join(wiki(path, title) for path, title in paths)


def set_inline_properties(body: str, props: list[tuple[str, str]]) -> str:
    body = INLINE_MANAGED_RE.sub("", body)
    body = re.sub(r"\n{3,}", "\n\n", body).lstrip("\n")
    if not props:
        return body
    prop_lines = [f"{name}:: {value}" for name, value in props if value]
    if not prop_lines:
        return body
    match = TOP_INLINE_RE.match(body)
    if not match:
        return "\n".join(prop_lines) + "\n\n" + body
    existing = match.group(1).rstrip("\n").splitlines()
    rest = body[match.end() :].lstrip("\n")
    out: list[str] = []
    inserted = False
    for line in existing:
        out.append(line)
        if line.startswith("domain::") and not inserted:
            out.extend(prop_lines)
            inserted = True
    if not inserted:
        out = prop_lines + out
    return "\n".join(out) + "\n\n" + rest


def update_existing_note(path: Path, props: list[tuple[str, str]]) -> bool:
    text = nfc(path.read_text(encoding="utf-8"))
    fm, body = split_frontmatter(text)
    new_body = set_inline_properties(body, props)
    out = fm + new_body.lstrip("\n")
    if not out.endswith("\n"):
        out += "\n"
    out = nfc(out)
    if out == text:
        return False
    path.write_text(out, encoding="utf-8")
    return True


def frontmatter(title: str, subtype: str, course: str = "meta") -> list[str]:
    return [
        "---",
        "aliases: []",
        f"course: {course}",
        f"created: '{TODAY}'",
        f"date: '{TODAY}'",
        "semester: meta",
        "source: ''",
        "status: evergreen",
        "tags:",
        "- type/interface",
        f"- pkm/{subtype}",
        f"title: {title}",
        "type: interface",
        f"updated: '{TODAY}'",
        "---",
        "",
    ]


def hub_note(stages: tuple[Stage, ...], bridges: tuple[Bridge, ...], modules: dict[str, Module]) -> str:
    related = (
        [(stage.path, stage.title) for stage in stages]
        + [(bridge.path, bridge.title) for bridge in bridges]
        + [(module.path, module.label + " 인터페이스") for module in modules.values()]
    )
    lines = frontmatter("지식그래프 허브", "hub")
    lines.append("related:: " + field_value(related))
    lines.extend(
        [
            "",
            "# 지식그래프 허브",
            "",
            "> [!info] Graph View에서 폴더가 아니라 실제 노트가 노드로 보이도록 만든 최상위 인터페이스입니다.",
            "",
            "## 단계 구조",
            "",
        ]
    )
    for stage in stages:
        lines.append(f"- {wiki(stage.path, stage.title)}")
    lines.extend(["", "## 브리지 구조", ""])
    for bridge in bridges:
        lines.append(f"- {wiki(bridge.path, bridge.title)}")
    lines.extend(
        [
            "",
            "## 그래프 디자인",
            "",
            "```mermaid",
            "flowchart TB",
            '    Hub["지식그래프 허브"]',
            '    subgraph StageLayer["단계 인터페이스"]',
            "        direction LR",
        ]
    )
    for index, stage in enumerate(stages, 1):
        lines.append(f'        S{index}["{stage.title}"]')
    lines.extend(
        [
            "    end",
            '    subgraph BridgeLayer["브리지 인터페이스"]',
            "        direction LR",
        ]
    )
    for index, bridge in enumerate(bridges, 1):
        lines.append(f'        B{index}["{bridge.title}"]')
    lines.extend(["    end"])
    for index in range(1, len(stages) + 1):
        lines.append(f"    Hub --> S{index}")
    for index in range(1, len(bridges) + 1):
        lines.append(f"    Hub --> B{index}")
    lines.extend(
        [
            "```",
            "",
            "## 과목 모듈 인터페이스",
            "",
        ]
    )
    for module in modules.values():
        lines.append(f"- {wiki(module.path, module.label + ' 인터페이스')}")
    return "\n".join(lines)


def stage_note(
    stage: Stage,
    modules: dict[str, Module],
    bridges_by_course: dict[str, list[Bridge]],
) -> str:
    course_modules = [modules[c] for c in stage.courses if c in modules]
    bridge_set = sorted(
        {bridge for course in stage.courses for bridge in bridges_by_course.get(course, [])},
        key=lambda b: b.title,
    )
    related = (
        [(HUB, "지식그래프 허브")]
        + [(module.path, module.label + " 인터페이스") for module in course_modules]
        + [(bridge.path, bridge.title) for bridge in bridge_set]
    )
    lines = frontmatter(stage.title, "stage", stage.key)
    lines.append("graph:: " + wiki(HUB, "지식그래프 허브"))
    lines.append("related:: " + field_value(related))
    lines.extend(["", f"# {stage.title}", "", stage.summary, "", "## 과목 모듈", ""])
    for module in course_modules:
        lines.append(f"- {wiki(module.path, module.label + ' 인터페이스')} -> {wiki(module.anchor_rel, module.label)}")
    lines.extend(["", "## 연결 브리지", ""])
    for bridge in bridge_set:
        lines.append(f"- {wiki(bridge.path, bridge.title)}")
    return "\n".join(lines)


def bridge_note(bridge: Bridge, modules: dict[str, Module]) -> str:
    course_modules = [modules[c] for c in bridge.courses if c in modules]
    support = []
    if bridge.key == "archive-operations":
        support = [(rel_path, Path(rel_path).stem) for rel_path in sorted(ROOT_SUPPORT)]
    related = (
        [(HUB, "지식그래프 허브")]
        + [(module.path, module.label + " 인터페이스") for module in course_modules]
        + support
    )
    lines = frontmatter(bridge.title, "bridge", bridge.key)
    lines.append("graph:: " + wiki(HUB, "지식그래프 허브"))
    lines.append("related:: " + field_value(related))
    lines.extend(["", f"# {bridge.title}", "", bridge.summary, "", "## 연결 과목", ""])
    for module in course_modules:
        lines.append(f"- {wiki(module.path, module.label + ' 인터페이스')} -> {wiki(module.anchor_rel, module.label)}")
    if support:
        lines.extend(["", "## 운영 문서", ""])
        for rel_path, title in support:
            lines.append(f"- {wiki(rel_path, title)}")
    return "\n".join(lines)


def module_note(
    module: Module,
    stage_by_course: dict[str, Stage],
    bridges_by_course: dict[str, list[Bridge]],
    domains: dict[str, object],
    note_count: int,
) -> str:
    stage = stage_by_course[module.course]
    bridges = bridges_by_course.get(module.course, [])
    domain = domains[module.domain_key]
    title = module.label + " 인터페이스"
    related = (
        [(HUB, "지식그래프 허브"), (domain.path, domain.title), (stage.path, stage.title)]
        + [(bridge.path, bridge.title) for bridge in bridges]
        + [(module.anchor_rel, module.label)]
    )
    lines = frontmatter(title, "module", module.course)
    lines.append("graph:: " + wiki(HUB, "지식그래프 허브"))
    lines.append("domain:: " + wiki(domain.path, domain.title))
    lines.append("stage:: " + wiki(stage.path, stage.title))
    if bridges:
        lines.append("bridge:: " + field_value([(bridge.path, bridge.title) for bridge in bridges]))
    lines.append("related:: " + field_value(related))
    lines.extend(
        [
            "",
            f"# {title}",
            "",
            f"{module.label} 관련 노트 {note_count}개를 분야, 단계, 브리지 인터페이스에 묶는 과목 모듈 노드입니다.",
            "",
            "## 대표 노트",
            "",
            f"- {wiki(module.anchor_rel, module.label)}",
            "",
            "## 연결 인터페이스",
            "",
            f"- 분야: {wiki(domain.path, domain.title)}",
            f"- 단계: {wiki(stage.path, stage.title)}",
        ]
    )
    for bridge in bridges:
        lines.append(f"- 브리지: {wiki(bridge.path, bridge.title)}")
    return "\n".join(lines)


def update_graph_json() -> bool:
    path = VAULT / ".obsidian/graph.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(
        {
            "showTags": True,
            "showAttachments": True,
            "showArrow": True,
            "collapse-color-groups": False,
            "collapse-display": False,
            "collapse-forces": False,
            "nodeSizeMultiplier": 1.25,
            "lineSizeMultiplier": 1.35,
            "textFadeMultiplier": 0,
            "linkDistance": 210,
            "repelStrength": 12,
            "scale": 0.55,
        }
    )
    stage_groups = [
        {"query": "tag:#pkm/hub", "color": {"a": 1, "rgb": 0x7C3AED}},
        {"query": "tag:#pkm/stage", "color": {"a": 1, "rgb": 0xF59E0B}},
        {"query": "tag:#pkm/bridge", "color": {"a": 1, "rgb": 0x14B8A6}},
        {"query": "tag:#pkm/module", "color": {"a": 1, "rgb": 0x64748B}},
        {"query": "tag:#pkm/domain", "color": {"a": 1, "rgb": 0x22C55E}},
        {"query": "path:ComputerScience/01_programming-foundations", "color": {"a": 1, "rgb": 0x16A34A}},
        {"query": "path:ComputerScience/02_math-theory", "color": {"a": 1, "rgb": 0xEAB308}},
        {"query": "path:ComputerScience/03_ai-ml-data", "color": {"a": 1, "rgb": 0xEF4444}},
        {"query": "path:ComputerScience/04_systems-infrastructure", "color": {"a": 1, "rgb": 0x2563EB}},
        {"query": "path:ComputerScience/05_software-engineering", "color": {"a": 1, "rgb": 0x10B981}},
        {"query": "path:ComputerScience/06_algorithms-graphics", "color": {"a": 1, "rgb": 0xA855F7}},
        {"query": "path:ComputerScience/07_professional-humanities", "color": {"a": 1, "rgb": 0xEC4899}},
    ]
    existing = data.get("colorGroups", [])
    query_to_group = {group.get("query"): group for group in existing if isinstance(group, dict)}
    for group in stage_groups:
        query_to_group[group["query"]] = group
    ordered = []
    seen: set[str] = set()
    for group in stage_groups + existing:
        query = group.get("query") if isinstance(group, dict) else None
        if query and query not in seen and query in query_to_group:
            ordered.append(query_to_group[query])
            seen.add(query)
    data["colorGroups"] = ordered
    out = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if path.read_text(encoding="utf-8") == out:
        return False
    path.write_text(out, encoding="utf-8")
    return True


def add_node(nodes: list[dict], node: dict) -> None:
    nodes.append(node)


def make_canvas(
    modules: dict[str, Module],
    stage_by_course: dict[str, Stage],
    bridges_by_course: dict[str, list[Bridge]],
    domains: dict[str, object],
) -> dict[str, list[dict]]:
    nodes: list[dict] = []
    edges: list[dict] = []

    def edge(from_id: str, to_id: str, label: str = "", color: str = "6") -> None:
        edge_id = f"e_{from_id}_{to_id}".replace("-", "_").replace(" ", "_")
        edges.append(
            {
                "id": edge_id,
                "fromNode": from_id,
                "fromSide": "bottom",
                "toNode": to_id,
                "toSide": "top",
                "color": color,
                **({"label": label} if label else {}),
            }
        )

    add_node(nodes, {"id": "g_hub", "type": "group", "x": -360, "y": -760, "width": 720, "height": 220, "color": "6", "label": "지식그래프 허브"})
    add_node(nodes, {"id": "hub", "type": "file", "file": HUB, "x": -150, "y": -700, "width": 300, "height": 100, "color": "6"})

    add_node(nodes, {"id": "g_stage", "type": "group", "x": -1120, "y": -460, "width": 2240, "height": 230, "color": "5", "label": "단계 인터페이스"})
    stage_ids: dict[str, str] = {}
    for index, stage in enumerate(STAGES):
        node_id = f"stage_{index + 1}"
        stage_ids[stage.key] = node_id
        add_node(nodes, {"id": node_id, "type": "file", "file": stage.path, "x": -1040 + index * 360, "y": -390, "width": 300, "height": 90, "color": "5"})
        edge("hub", node_id, "stage", "5")

    add_node(nodes, {"id": "g_domain", "type": "group", "x": -1260, "y": -150, "width": 2520, "height": 230, "color": "4", "label": "분야 인터페이스"})
    domain_ids: dict[str, str] = {}
    for index, domain in enumerate(DOMAINS):
        node_id = f"domain_{domain.key}"
        domain_ids[domain.key] = node_id
        add_node(nodes, {"id": node_id, "type": "file", "file": domain.path, "x": -1190 + index * 350, "y": -85, "width": 290, "height": 90, "color": COLOR_BY_DOMAIN_KEY.get(Path(domain.path).parts[1], "4")})
        edge("hub", node_id, "domain", "4")

    add_node(nodes, {"id": "g_bridge", "type": "group", "x": -1260, "y": 160, "width": 2520, "height": 250, "color": "6", "label": "브리지 인터페이스"})
    bridge_ids: dict[str, str] = {}
    for index, bridge in enumerate(BRIDGES):
        node_id = f"bridge_{bridge.key}"
        bridge_ids[bridge.key] = node_id
        add_node(nodes, {"id": node_id, "type": "file", "file": bridge.path, "x": -1190 + index * 350, "y": 230, "width": 290, "height": 90, "color": "6"})
        edge("hub", node_id, "bridge", "6")

    domain_course_map: dict[str, list[Module]] = defaultdict(list)
    for module in modules.values():
        domain_course_map[module.domain_key].append(module)

    x0 = -1260
    y0 = 520
    col_w = 360
    for index, domain in enumerate(DOMAINS):
        domain_modules = sorted(domain_course_map[domain.key], key=lambda m: m.label)
        group_h = max(260, 120 + len(domain_modules) * 92)
        group_id = f"group_courses_{domain.key}"
        x = x0 + index * col_w
        add_node(nodes, {"id": group_id, "type": "group", "x": x, "y": y0, "width": 330, "height": group_h, "color": COLOR_BY_DOMAIN_KEY.get(Path(domain.path).parts[1], "4"), "label": domain.title.replace(" 인터페이스", "")})
        for j, module in enumerate(domain_modules):
            node_id = f"module_{module.course}".replace("-", "_")
            add_node(nodes, {"id": node_id, "type": "file", "file": module.path, "x": x + 25, "y": y0 + 60 + j * 92, "width": 280, "height": 70, "color": COLOR_BY_DOMAIN_KEY.get(Path(domain.path).parts[1], "4")})
            edge(domain_ids[domain.key], node_id, "module", COLOR_BY_DOMAIN_KEY.get(Path(domain.path).parts[1], "4"))
            stage = stage_by_course[module.course]
            edge(stage_ids[stage.key], node_id, "stage", "5")
            for bridge in bridges_by_course.get(module.course, []):
                edge(bridge_ids[bridge.key], node_id, "bridge", "6")

    return {"nodes": nodes, "edges": edges}


def write_canvas(canvas: dict[str, list[dict]]) -> bool:
    path = VAULT / "지식그래프 레벨 인터페이스.canvas"
    out = json.dumps(canvas, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == out:
        return False
    path.write_text(out, encoding="utf-8")
    return True


def main() -> None:
    modules = build_modules()
    stage_by_course = course_to_stage()
    bridges_by_course = course_to_bridges()
    domains = domain_by_key()

    note_counts: Counter[str] = Counter()
    classified_notes: list[tuple[Path, str]] = []
    unclassified: list[str] = []
    for path in iter_notes():
        if is_interface_note(path):
            continue
        course = classify_course(path)
        if course is None:
            unclassified.append(rel(path))
            continue
        classified_notes.append((path, course))
        note_counts[course] += 1

    generated_updates = 0
    generated_updates += int(write_note(VAULT / HUB, hub_note(STAGES, BRIDGES, modules)))
    for stage in STAGES:
        generated_updates += int(write_note(VAULT / stage.path, stage_note(stage, modules, bridges_by_course)))
    for bridge in BRIDGES:
        generated_updates += int(write_note(VAULT / bridge.path, bridge_note(bridge, modules)))
    for course, module in modules.items():
        if course not in stage_by_course:
            raise KeyError(f"Missing stage mapping for {course}")
        generated_updates += int(
            write_note(
                VAULT / module.path,
                module_note(module, stage_by_course, bridges_by_course, domains, note_counts[course]),
            )
        )

    note_updates = 0
    for path, course in classified_notes:
        if course == "archive-operations":
            ops_bridge = next(bridge for bridge in BRIDGES if bridge.key == "archive-operations")
            props = [
                ("graph", wiki(HUB, "지식그래프 허브")),
                ("bridge", wiki(ops_bridge.path, ops_bridge.title)),
            ]
        else:
            module = modules.get(course)
            stage = stage_by_course.get(course)
            bridges = bridges_by_course.get(course, [])
            if not module or not stage:
                unclassified.append(rel(path))
                continue
            props = [
                ("stage", wiki(stage.path, stage.title)),
                ("module", wiki(module.path, module.label + " 인터페이스")),
                ("bridge", field_value([(bridge.path, bridge.title) for bridge in bridges])),
            ]
        note_updates += int(update_existing_note(path, props))

    domain_updates = 0
    for domain in DOMAINS:
        domain_courses = [Path(course.rel).parts[2] for course in domain.courses]
        stages = sorted({stage_by_course[c] for c in domain_courses if c in stage_by_course}, key=lambda s: s.key)
        bridges = sorted(
            {bridge for course in domain_courses for bridge in bridges_by_course.get(course, [])},
            key=lambda b: b.title,
        )
        module_links = [(modules[c].path, modules[c].label + " 인터페이스") for c in domain_courses if c in modules]
        props = [
            ("graph", wiki(HUB, "지식그래프 허브")),
            ("stage", field_value([(stage.path, stage.title) for stage in stages])),
            ("module", field_value(module_links)),
            ("bridge", field_value([(bridge.path, bridge.title) for bridge in bridges])),
        ]
        domain_updates += int(update_existing_note(VAULT / domain.path, props))

    graph_updated = update_graph_json()
    canvas_updated = write_canvas(make_canvas(modules, stage_by_course, bridges_by_course, domains))

    print(
        "stage_interface_graph "
        f"generated_notes={1 + len(STAGES) + len(BRIDGES) + len(modules)} "
        f"generated_updates={generated_updates} note_updates={note_updates} "
        f"domain_updates={domain_updates} graph_json_updated={int(graph_updated)} "
        f"canvas_updated={int(canvas_updated)} unclassified={len(unclassified)}"
    )
    if unclassified:
        for item in unclassified[:20]:
            print(f"unclassified: {item}")


if __name__ == "__main__":
    main()
