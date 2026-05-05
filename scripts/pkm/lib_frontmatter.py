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
    return yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=True,
        default_flow_style=False,
        width=4096,
    )


def read_note(path: Path) -> tuple[dict, str]:
    """Read a markdown note. Robust to files that start with `---` as a horizontal rule
    (not as YAML frontmatter delimiter). On YAML parse failure, treat as no frontmatter.
    """
    raw = Path(path).read_text(encoding="utf-8")
    raw = unicodedata.normalize("NFC", raw)
    try:
        post = frontmatter.loads(raw)
        return dict(post.metadata), post.content
    except (yaml.YAMLError, Exception):
        return {}, raw


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


def iter_vault_notes(
    root: Path,
    exclude: Iterable[str] = (".obsidian", ".git", ".claude", ".pytest_cache", "scripts", "docs"),
) -> Iterable[Path]:
    exclude_parts = {str(e) for e in exclude}
    for p in sorted(Path(root).rglob("*.md")):
        rel = p.relative_to(root)
        if any(part in exclude_parts for part in rel.parts):
            continue
        yield p
