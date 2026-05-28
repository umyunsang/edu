#!/usr/bin/env python3
"""Configure Extended Graph for the edu top-down knowledge graph."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_frontmatter import iter_vault_notes, read_note, write_note  # noqa: E402

VAULT = Path(__file__).resolve().parents[2]
PLUGIN_ID = "extended-graph"
TODAY = "2026-05-28"

GRAPH_SETTINGS = {
    "collapse-filter": True,
    "search": "",
    "showTags": False,
    "showAttachments": False,
    "hideUnresolved": True,
    "showOrphans": True,
    "collapse-color-groups": False,
    "collapse-display": False,
    "showArrow": True,
    "textFadeMultiplier": 0,
    "nodeSizeMultiplier": 0.56,
    "lineSizeMultiplier": 0.72,
    "collapse-forces": False,
    "centerStrength": 0.42,
    "repelStrength": 17,
    "linkStrength": 0.72,
    "linkDistance": 270,
    "scale": 0.54,
    "close": False,
}

COLOR_GROUPS = [
    ("tag:#pkm/kg-skeleton", 0x5B5BD6),
    ("tag:#pkm/kg-method-2026", 0x8B5CF6),
    ("tag:#pkm/kg-community", 0x009E73),
    ("tag:#pkm/kg-course", 0x0072B2),
    ("tag:#pkm/kg-concept", 0xE69F00),
    ("tag:#pkm/kg-evidence", 0xD55E00),
    ("tag:#pkm/kg-query", 0x56B4E9),
    ("tag:#pkm/hub", 0x5B5BD6),
    ("tag:#pkm/stage", 0xE69F00),
    ("tag:#pkm/bridge", 0x56B4E9),
    ("tag:#pkm/module", 0x6B7280),
    ("tag:#pkm/domain", 0x009E73),
    ("path:ComputerScience/01_programming-foundations", 0x009E73),
    ("path:ComputerScience/02_math-theory", 0xE6B800),
    ("path:ComputerScience/03_ai-ml-data", 0xD55E00),
    ("path:ComputerScience/04_systems-infrastructure", 0x0072B2),
    ("path:ComputerScience/05_software-engineering", 0x56B4E9),
    ("path:ComputerScience/06_algorithms-graphics", 0xCC79A7),
    ("path:ComputerScience/07_professional-humanities", 0x8B5CF6),
]


def interactive(colormap: str = "rainbow") -> dict:
    return {
        "colormap": colormap,
        "colors": [],
        "unselected": [],
        "excludeRegex": {"regex": "", "flags": ""},
        "noneType": "none",
        "undefinedType": "undefined",
        "showOnGraph": True,
        "enableByDefault": True,
        "useForNodeColor": False,
    }


def shape_query(level: str, index: int) -> dict:
    return {
        "combinationLogic": "OR",
        "index": index,
        "rules": [
            {"source": "property", "property": "kg_level", "logic": "is", "value": level}
        ],
    }


EXTENDED_GRAPH_SETTINGS = {
    "enableFeatures": {
        "graph": {
            "auto-enabled": True,
            "tags": False,
            "properties": True,
            "property-key": True,
            "links": False,
            "linksSameColorAsNode": False,
            "folders": True,
            "imagesFromProperty": False,
            "imagesFromEmbeds": False,
            "imagesForAttachments": False,
            "focus": True,
            "shapes": True,
            "elements-stats": True,
            "names": True,
            "icons": False,
            "arrows": True,
            "layers": True,
        },
        "localgraph": {
            "auto-enabled": True,
            "tags": False,
            "properties": True,
            "property-key": True,
            "links": False,
            "linksSameColorAsNode": False,
            "folders": True,
            "imagesFromProperty": False,
            "imagesFromEmbeds": True,
            "imagesForAttachments": True,
            "focus": True,
            "shapes": True,
            "elements-stats": True,
            "names": True,
            "icons": False,
            "arrows": True,
            "layers": True,
        },
    },
    "interactiveSettings": {
        "kg_level": interactive("rainbow"),
        "kg_graph_size": interactive("YlOrRd"),
        "kg_role": interactive("tab10"),
        "kg_layer_label": interactive("rainbow"),
        "course": interactive("tab20"),
        "semester": interactive("tab20"),
        "type": interactive("tab10"),
    },
    "additionalProperties": {
        "kg_level": {"graph": True, "localgraph": True},
        "kg_graph_size": {"graph": True, "localgraph": True},
        "kg_role": {"graph": True, "localgraph": True},
        "kg_layer_label": {"graph": True, "localgraph": True},
        "course": {"graph": True, "localgraph": True},
        "semester": {"graph": True, "localgraph": True},
        "type": {"graph": True, "localgraph": True},
    },
    "canonicalizePropertiesWithDataview": True,
    "nodesSizeProperties": ["kg_graph_size"],
    "nodesSizeFunction": "default",
    "nodesSizeRange": {"min": 0.45, "max": 1.8},
    "nodesColorFunction": "default",
    "linksSizeFunction": "default",
    "linksColorFunction": "default",
    "graphStatsDirection": "normal",
    "recomputeStatsOnGraphChange": False,
    "maxNodes": 3200,
    "delay": 900,
    "enableCSS": True,
    "cssSnippetFilename": "graph-view-design",
    "imageProperties": [],
    "allowExternalImages": False,
    "allowExternalLocalImages": False,
    "shapeQueries": {
        "diamond": shape_query("0", 0),
        "hexagon": shape_query("1", 1),
        "square": shape_query("2", 2),
        "pentagon": shape_query("3", 3),
        "circle": shape_query("4", 4),
    },
    "layerProperties": ["kg_level"],
    "numberOfActiveLayers": 6,
    "layersOrder": "ASC",
    "displayLabelsInUI": True,
    "nodesWithoutLayerOpacity": 0.18,
    "useLayerCustomOpacity": True,
    "layersCustomOpacity": {
        "0": 1,
        "1": 0.94,
        "2": 0.86,
        "3": 0.74,
        "4": 0.58,
        "5": 0.42,
    },
    "layersLevels": {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
    },
    "defaultLevelForLayers": 5,
    "folderShowFullPath": False,
    "curvedLinks": False,
    "outlineLinks": False,
    "displayLinkTypeLabel": False,
    "useBitmapsForLinkLabels": True,
    "arrowScale": 1,
    "arrowFixedSize": True,
    "numberOfCharacters": 48,
    "showOnlyFileName": False,
    "noExtension": True,
    "usePropertiesForName": ["title"],
    "addBackgroundToName": True,
    "horizontalLegend": False,
}

ROLE_LEVELS = {
    "hub": (0, 180, "L0 hub"),
    "skeleton": (0, 176, "L0 skeleton"),
    "stage": (1, 150, "L1 stage"),
    "method": (1, 148, "L1 method"),
    "query": (1, 140, "L1 query"),
    "domain": (2, 126, "L2 domain"),
    "community": (2, 124, "L2 community"),
    "bridge": (2, 118, "L2 bridge"),
    "course-interface": (3, 102, "L3 course interface"),
    "course-profile": (3, 96, "L3 course profile"),
    "evidence-index": (3, 88, "L3 evidence"),
    "media-index": (3, 84, "L3 media"),
    "report": (3, 82, "L3 report"),
    "concept": (4, 66, "L4 concept"),
    "source-note": (4, 62, "L4 source"),
    "support": (4, 60, "L4 support"),
}

EXCLUDES = {
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


def rel(path: Path) -> str:
    return path.relative_to(VAULT).as_posix()


def infer_role(path: Path) -> str:
    r = rel(path)
    if r == "ComputerScience/00_graph-interfaces/지식그래프 허브.md":
        return "hub"
    if r == "ComputerScience/00_graph-interfaces/archive-kg/2026 GraphRAG 아카이브 스켈레톤.md":
        return "skeleton"
    if r.startswith("ComputerScience/00_graph-interfaces/stages/"):
        return "stage"
    if r.startswith("ComputerScience/00_graph-interfaces/archive-kg/methods-2026/"):
        return "method"
    if r.startswith("ComputerScience/00_graph-interfaces/archive-kg/query-modes/"):
        return "query"
    if (
        r.startswith("ComputerScience/01_")
        or r.startswith("ComputerScience/02_")
        or r.startswith("ComputerScience/03_")
        or r.startswith("ComputerScience/04_")
        or r.startswith("ComputerScience/05_")
        or r.startswith("ComputerScience/06_")
        or r.startswith("ComputerScience/07_")
    ) and path.name.endswith("인터페이스.md") and len(path.relative_to(VAULT).parts) == 3:
        return "domain"
    if r.startswith("ComputerScience/00_graph-interfaces/archive-kg/communities/"):
        return "community"
    if r.startswith("ComputerScience/00_graph-interfaces/bridges/"):
        return "bridge"
    if r.startswith("ComputerScience/00_graph-interfaces/courses/"):
        return "course-interface"
    if r.startswith("ComputerScience/00_graph-interfaces/archive-kg/courses/"):
        return "course-profile"
    if r.startswith("ComputerScience/00_graph-interfaces/archive-kg/evidence/"):
        return "evidence-index"
    if r.startswith("ComputerScience/00_graph-interfaces/archive-kg/media/"):
        return "media-index"
    if r.startswith("ComputerScience/00_graph-interfaces/archive-kg/concepts/"):
        return "concept"
    if r.startswith("ComputerScience/00_graph-interfaces/archive-kg/") and path.name.endswith("리포트.md"):
        return "report"
    if r in {"README.md", "AGENTS.md", "CLAUDE.md", "커리큘럼 관계 정리.md"}:
        return "support"
    return "source-note"


def configure_community_plugins() -> bool:
    path = VAULT / ".obsidian/community-plugins.json"
    plugins = json.loads(path.read_text(encoding="utf-8"))
    if PLUGIN_ID in plugins:
        return False
    plugins.append(PLUGIN_ID)
    path.write_text(json.dumps(plugins, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def configure_core_graph() -> bool:
    path = VAULT / ".obsidian/graph.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update(GRAPH_SETTINGS)
    existing = [group for group in data.get("colorGroups", []) if isinstance(group, dict)]
    groups_by_query = {group.get("query"): group for group in existing if group.get("query")}
    ordered: list[dict] = []
    seen: set[str] = set()
    for query, rgb in COLOR_GROUPS:
        group = groups_by_query.get(query, {"query": query})
        group["color"] = {"a": 1, "rgb": rgb}
        ordered.append(group)
        seen.add(query)
    for group in existing:
        query = group.get("query")
        if query and query not in seen:
            ordered.append(group)
            seen.add(query)
    data["colorGroups"] = ordered
    out = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if old == out:
        return False
    path.write_text(out, encoding="utf-8")
    return True


def configure_extended_graph() -> bool:
    plugin_dir = VAULT / ".obsidian/plugins/extended-graph"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    path = plugin_dir / "data.json"
    out = json.dumps(EXTENDED_GRAPH_SETTINGS, ensure_ascii=False, indent=2) + "\n"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if old == out:
        return False
    path.write_text(out, encoding="utf-8")
    return True


def configure_note_metadata() -> tuple[int, dict[str, int]]:
    changed = 0
    counts: dict[str, int] = {role: 0 for role in ROLE_LEVELS}
    for path in iter_vault_notes(VAULT, exclude=EXCLUDES):
        role = infer_role(path)
        level, size, layer_label = ROLE_LEVELS[role]
        counts[role] = counts.get(role, 0) + 1
        fm, body = read_note(path)
        next_fm = dict(fm)
        if next_fm.get("aliases") == "[]":
            next_fm["aliases"] = []
        next_fm["kg_level"] = level
        next_fm["kg_graph_size"] = size
        next_fm["kg_role"] = role
        next_fm["kg_layer_label"] = layer_label
        if next_fm == fm:
            continue
        write_note(path, next_fm, body)
        changed += 1
    return changed, counts


def update_design_note() -> bool:
    path = VAULT / "ComputerScience/00_graph-interfaces/Graph View 디자인 리서치.md"
    fm, body = read_note(path)
    marker = "## Extended Graph 적용"
    section = f"""{marker}

- 설치 후보는 [ElsaTam/obsidian-extended-graph 2.7.7](https://github.com/ElsaTam/obsidian-extended-graph/releases/tag/2.7.7)로 확정했습니다. 릴리스 에셋의 `manifest.json`은 upstream 기준 `2.7.6` 버전을 유지합니다.
- 노드 크기는 연결 수가 아니라 `kg_graph_size` frontmatter 속성으로 계산합니다.
- 계층 레이어는 `kg_level`을 사용합니다. Level 0은 허브/스켈레톤, Level 1은 단계/방법론/질의 모드, Level 2는 분야/커뮤니티/브리지, Level 3은 과목/근거/미디어 인덱스, Level 4는 개념과 원문 노트입니다.
- 전역 Graph View는 첨부파일 노드를 숨겨 PNG가 독립 노드로 그래프를 흐리지 않게 했습니다. PNG는 정합한 노트의 embed와 Local Graph 이미지 표시를 통해 확인합니다.
- 대형 vault 성능을 위해 `maxNodes`는 3200, 초기화 지연은 900ms, 통계 재계산은 수동/초기화 중심으로 둡니다.
"""
    if marker in body:
        before, _sep, rest = body.partition(marker)
        tail_start = rest.find("\n## ")
        tail = rest[tail_start:] if tail_start != -1 else ""
        next_body = before.rstrip() + "\n\n" + section + tail
    else:
        next_body = body.rstrip() + "\n\n" + section
    fm = dict(fm)
    fm["updated"] = TODAY
    if next_body == body and fm == read_note(path)[0]:
        return False
    write_note(path, fm, next_body)
    return True


def main() -> None:
    community = configure_community_plugins()
    graph = configure_core_graph()
    extended = configure_extended_graph()
    notes, counts = configure_note_metadata()
    design = update_design_note()
    print(
        "extended_graph_configured "
        f"community_plugins_updated={int(community)} "
        f"graph_json_updated={int(graph)} "
        f"data_json_updated={int(extended)} "
        f"note_metadata_changed={notes} "
        f"design_note_updated={int(design)}"
    )
    for role in sorted(counts):
        print(f"role_count {role}={counts[role]}")


if __name__ == "__main__":
    main()
