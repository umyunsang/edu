"""Frontmatter read/write utilities for the pkm-refactor.

Korean-text safe: UTF-8, NFC normalization, no BOM.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Iterable

try:
    import frontmatter
except ModuleNotFoundError:  # pragma: no cover - depends on local Python env
    frontmatter = None

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on local Python env
    yaml = None

DEFAULT_EXCLUDE = (
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
)


def _yaml_dump(data: dict) -> str:
    if yaml is None:
        lines: list[str] = []
        for key in sorted(data):
            value = data[key]
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"- {item!r}")
            else:
                lines.append(f"{key}: {value!r}")
        return "\n".join(lines) + "\n"
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
    if frontmatter is None:
        if not raw.startswith("---\n"):
            return {}, raw
        end = raw.find("\n---", 4)
        if end == -1:
            return {}, raw
        body = raw[raw.find("\n", end + 4) + 1 :]
        yaml_text = raw[4:end]
        if yaml is not None:
            try:
                parsed = yaml.safe_load(yaml_text) or {}
                if isinstance(parsed, dict):
                    return parsed, body
            except Exception:
                pass
        metadata: dict = {}
        current_list: str | None = None
        for line in yaml_text.splitlines():
            if not line.strip():
                continue
            if line.startswith("- ") and current_list:
                metadata.setdefault(current_list, [])
                if isinstance(metadata[current_list], list):
                    metadata[current_list].append(line[2:].strip().strip("'\""))
                continue
            if ":" not in line or line.startswith(" "):
                current_list = None
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                metadata[key] = value.strip("'\"")
                current_list = None
            else:
                metadata[key] = []
                current_list = key
        return metadata, body
    try:
        post = frontmatter.loads(raw)
        return dict(post.metadata), post.content
    except Exception:
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
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
) -> Iterable[Path]:
    """Robust os.walk-based iteration with NFC normalization (macOS HFS+/iCloud
    returns filenames in NFD by default which breaks string equality with NFC literals)."""
    import os
    exclude_parts = {str(e) for e in exclude}
    root = Path(root)
    results: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_parts]
        # NFC-normalize the dirpath so resulting Path objects use NFC consistently
        dirpath_nfc = unicodedata.normalize("NFC", dirpath)
        for fn in filenames:
            if not fn.endswith(".md"):
                continue
            fn_nfc = unicodedata.normalize("NFC", fn)
            results.append(Path(dirpath_nfc) / fn_nfc)
    for p in sorted(results):
        yield p
