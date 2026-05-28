#!/usr/bin/env python3
"""Move semester-named course folders under content-domain interfaces.

The vault used to expose courses primarily by year/semester. This pass creates
field-level interface folders and moves each course folder under the domain that
best matches its actual note/PDF content. References are rewritten for wikilinks,
canvas file nodes, and plain vault-relative paths.
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
EXCLUDE_DIRS = {
    ".git",
    ".obsidian",
    ".agents",
    ".claude",
    ".playwright-cli",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    ".aioss-rag",
}

DIR_MOVES = {
    "ComputerScience/1-2_coding-basics": "ComputerScience/01_programming-foundations/coding-basics",
    "ComputerScience/2-1_python": "ComputerScience/01_programming-foundations/python-programming",
    "ComputerScience/2-1_data-structures": "ComputerScience/01_programming-foundations/data-structures",
    "ComputerScience/elective_coding-test": "ComputerScience/01_programming-foundations/coding-test",
    "ComputerScience/elective_java": "ComputerScience/01_programming-foundations/java-programming",
    "ComputerScience/2-1_probability-statistics": "ComputerScience/02_math-theory/probability-statistics",
    "ComputerScience/2-2_discrete-math": "ComputerScience/02_math-theory/discrete-mathematics",
    "ComputerScience/3-2_optimization-math": "ComputerScience/02_math-theory/optimization-math",
    "ComputerScience/3-1_mathematical-logic": "ComputerScience/02_math-theory/mathematical-logic",
    "ComputerScience/2-1_AI": "ComputerScience/03_ai-ml-data/artificial-intelligence",
    "ComputerScience/3-1_machine-learning": "ComputerScience/03_ai-ml-data/machine-learning",
    "ComputerScience/3-1_ML-project": "ComputerScience/03_ai-ml-data/ml-projects",
    "ComputerScience/3-2_neural-network": "ComputerScience/03_ai-ml-data/neural-networks",
    "ComputerScience/3-2_bigdata-analysis": "ComputerScience/03_ai-ml-data/big-data-analysis",
    "ComputerScience/4-1_computer-vision": "ComputerScience/03_ai-ml-data/computer-vision",
    "ComputerScience/elective_LLM": "ComputerScience/03_ai-ml-data/large-language-models",
    "ComputerScience/3-1_AI-system-design": "ComputerScience/03_ai-ml-data/ai-system-design",
    "ComputerScience/elective_convergence": "ComputerScience/03_ai-ml-data/generative-ai-fine-tuning",
    "ComputerScience/2-1_linux": "ComputerScience/04_systems-infrastructure/linux",
    "ComputerScience/2-1_computer-architecture": "ComputerScience/04_systems-infrastructure/computer-architecture",
    "ComputerScience/2-2_operating-system": "ComputerScience/04_systems-infrastructure/operating-systems",
    "ComputerScience/2-2_computer-network": "ComputerScience/04_systems-infrastructure/computer-networks",
    "ComputerScience/3-1_distributed-computing": "ComputerScience/04_systems-infrastructure/parallel-distributed-computing",
    "ComputerScience/elective_docker-k8s": "ComputerScience/04_systems-infrastructure/container-orchestration",
    "ComputerScience/2-1_web-programming": "ComputerScience/05_software-engineering/web-programming",
    "ComputerScience/2-2_database": "ComputerScience/05_software-engineering/database-systems",
    "ComputerScience/2-2_OSS": "ComputerScience/05_software-engineering/open-source-software",
    "ComputerScience/3-1_programming-languages": "ComputerScience/05_software-engineering/programming-languages",
    "ComputerScience/4-1_AIOSS": "ComputerScience/05_software-engineering/aioss-open-source-delivery",
    "ComputerScience/4-1_algorithm": "ComputerScience/06_algorithms-graphics/algorithm-design-analysis",
    "ComputerScience/3-2_computer-graphics": "ComputerScience/06_algorithms-graphics/computer-graphics",
    "ComputerScience/3-1_intellectual-property": "ComputerScience/07_professional-humanities/intellectual-property",
    "ComputerScience/4-1_creative-writing": "ComputerScience/07_professional-humanities/creative-writing",
    "ComputerScience/3-2_classics-reading": "ComputerScience/07_professional-humanities/classics-reading",
    "ComputerScience/misc": "ComputerScience/07_professional-humanities/degree-portfolio",
}

WIKI_RE = re.compile(r"(!?)\[\[([^\]#|]+)(#[^\]|]*)?(\|[^\]]*)?\]\]")


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def iter_rewrite_files() -> list[Path]:
    results: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(VAULT):
        dirnames[:] = [
            d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")
        ]
        dirpath = nfc(dirpath)
        for filename in filenames:
            filename = nfc(filename)
            if filename.endswith((".md", ".canvas", ".json")):
                results.append(Path(dirpath) / filename)
    return sorted(results)


def rewrite_wikilinks(text: str, moves: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        bang, target, section, alias = match.groups()
        clean = nfc(target.strip())
        suffix = ".md" if clean.endswith(".md") else ""
        no_ext = clean.removesuffix(".md")
        for old, new in moves.items():
            if no_ext == old or no_ext.startswith(old + "/"):
                updated = new + no_ext[len(old) :]
                return f"{bang}[[{updated}{suffix}{section or ''}{alias or ''}]]"
        return match.group(0)

    return WIKI_RE.sub(repl, text)


def rewrite_plain_paths(text: str, moves: dict[str, str]) -> str:
    out = text
    for old, new in sorted(moves.items(), key=lambda item: -len(item[0])):
        out = out.replace(old + "/", new + "/")
        out = out.replace(old + ".md", new + ".md")
        out = out.replace('"' + old + '"', '"' + new + '"')
    return out


def main() -> None:
    applied: dict[str, str] = {}
    for old, new in DIR_MOVES.items():
        src = VAULT / old
        dst = VAULT / new
        if not src.exists():
            continue
        if dst.exists():
            raise FileExistsError(f"Refusing to overwrite existing folder: {new}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        applied[old] = new

    rewritten = 0
    for path in iter_rewrite_files():
        try:
            text = nfc(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
        new_text = rewrite_plain_paths(rewrite_wikilinks(text, applied), applied)
        if new_text != text:
            path.write_text(nfc(new_text), encoding="utf-8")
            rewritten += 1

    print(f"field_interface_refactor moved={len(applied)} rewritten_files={rewritten}")


if __name__ == "__main__":
    main()
