#!/usr/bin/env python3
"""Inspect PNG attachments, rewrite duplicates, embed usable orphans, delete the rest."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphrag_2026_archive_migration import (  # noqa: E402
    COURSE_ROOT,
    KG_ROOT,
    SKELETON,
    VAULT,
    build_courses,
    find_course_for_path,
    nfc,
    note_header,
    rel,
    slugify,
    upsert_text_file,
    wikilink,
)

TODAY = "2026-05-28"
IMAGE_DIR = VAULT / "image"
REPORT = f"{KG_ROOT}/PNG 내용 검증 및 정리 리포트.md"
MEDIA_ROOT = f"{KG_ROOT}/media"
OCR_CACHE = VAULT / "scripts/pkm/.cache/png-ocr"

WIKI_IMAGE_RE = re.compile(r"!\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
MD_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
PNG_SECTION_RE = re.compile(r"\n## PNG 시각자료\n\n.*?(?=\n## |\Z)", re.S)
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
GENERIC_PNG_RE = re.compile(r"^(?:Pasted image \d+|스크린샷.*|Screenshot.*|Screen Shot.*|image[-_ ]?\d+|IMG[-_ ].*)$", re.I)
TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]{2,}")
STOP = {
    "image",
    "pasted",
    "png",
    "jpg",
    "jpeg",
    "screenshot",
    "스크린샷",
    "photo",
    "kakaotalk",
    "annotated",
    "page",
    "final",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
    "v8",
}

PREFIX_COURSE = {
    "3-1_machine-learning": "machine-learning",
    "3-1_distributed-computing": "parallel-distributed-computing",
    "3-1_intellectual-property": "intellectual-property",
    "3-1_programming-languages": "programming-languages",
    "3-2_bigdata-analysis": "big-data-analysis",
    "3-2_optimization-math": "optimization-math",
    "4-1_algorithm": "algorithm-design-analysis",
    "4-1_computer-vision": "computer-vision",
    "4-1_creative-writing": "creative-writing",
    "elective_LLM": "large-language-models",
}


@dataclass
class Ref:
    source: Path
    target_name: str


def all_text_files() -> list[Path]:
    out: list[Path] = []
    for suffix in ("*.md", "*.canvas"):
        for path in VAULT.rglob(suffix):
            if any(part in {".git", ".obsidian", ".agents", "__pycache__"} for part in path.parts):
                continue
            out.append(path)
    return sorted(out)


def image_files() -> list[Path]:
    return sorted(IMAGE_DIR.glob("*.png"))


def decode_target(value: str) -> str:
    value = urllib.parse.unquote(value.strip())
    value = value.split("#", 1)[0].split("|", 1)[0].strip()
    return nfc(Path(value).name)


def scan_refs() -> dict[str, list[Ref]]:
    refs: dict[str, list[Ref]] = defaultdict(list)
    png_names = {p.name for p in image_files()}
    for source in all_text_files():
        try:
            text = source.read_text(encoding="utf-8")
        except Exception:
            continue
        for match in WIKI_IMAGE_RE.finditer(text):
            name = decode_target(match.group(1))
            if name in png_names:
                refs[name].append(Ref(source, name))
        for match in MD_IMAGE_RE.finditer(text):
            name = decode_target(match.group(1))
            if name in png_names:
                refs[name].append(Ref(source, name))
    return refs


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_tracked_png_count() -> int:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "image/*.png"],
            cwd=VAULT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return 0
    return len([line for line in proc.stdout.splitlines() if line.strip()])


def duplicate_state(paths: list[Path]) -> tuple[int, int]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[sha256(path)].append(path)
    duplicate_groups = [items for items in groups.values() if len(items) > 1]
    return len(duplicate_groups), sum(len(items) for items in duplicate_groups)


def choose_canonical(group: list[Path], refs: dict[str, list[Ref]]) -> Path:
    def score(path: Path) -> tuple[int, int, int, str]:
        referenced = 1 if refs.get(path.name) else 0
        semantic = 1 if "__" in path.name or not path.name.startswith("Pasted image ") else 0
        ref_count = len(refs.get(path.name, []))
        return (referenced, semantic, ref_count, path.name)

    return sorted(group, key=score, reverse=True)[0]


def replace_image_refs(duplicate: Path, canonical: Path) -> int:
    changed = 0
    dup_name = duplicate.name
    canon_name = canonical.name
    dup_encoded = urllib.parse.quote(dup_name)
    canon_encoded = urllib.parse.quote(canon_name)
    for source in all_text_files():
        try:
            text = source.read_text(encoding="utf-8")
        except Exception:
            continue
        new = text.replace(dup_name, canon_name).replace(dup_encoded, canon_encoded)
        if new != text:
            source.write_text(nfc(new), encoding="utf-8")
            changed += 1
    return changed


def ocr_text(path: Path) -> str:
    OCR_CACHE.mkdir(parents=True, exist_ok=True)
    cache = OCR_CACHE / f"{sha256(path)}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="ignore")
    try:
        proc = subprocess.run(
            ["tesseract", str(path), "stdout", "-l", "kor+eng", "--psm", "6"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        text = proc.stdout[:12000]
    except Exception:
        text = ""
    cache.write_text(nfc(text), encoding="utf-8")
    return text


def image_size(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["magick", "identify", "-format", "%w x %h", str(path)],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def tokens(text: str) -> set[str]:
    values = {m.group(0).lower() for m in TOKEN_RE.finditer(nfc(text))}
    return {v for v in values if v not in STOP and not v.isdigit() and len(v) >= 2}


def infer_course_key(path: Path) -> str | None:
    name = path.name
    for prefix, key in PREFIX_COURSE.items():
        if name.startswith(prefix):
            return key
    return None


def note_corpus(courses: dict) -> dict[str, list[tuple[Path, set[str], str]]]:
    by_course: dict[str, list[tuple[Path, set[str], str]]] = defaultdict(list)
    for path in VAULT.rglob("*.md"):
        if any(part in {".git", ".obsidian", ".agents", ".aioss-eval", ".gemini", "__pycache__"} for part in path.parts):
            continue
        r = rel(path)
        if r.startswith(KG_ROOT + "/"):
            continue
        course = find_course_for_path(path, courses)
        if not course:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        text_tokens = tokens(path.stem + "\n" + text)
        by_course[course.key].append((path, text_tokens, text))
    return by_course


def best_note_for_png(path: Path, ocr: str, corpus: dict[str, list[tuple[Path, set[str], str]]], courses: dict) -> tuple[Path | None, str | None, int]:
    png_tokens = tokens(path.stem + "\n" + ocr)
    inferred = infer_course_key(path)
    candidates: list[tuple[str, Path, set[str]]] = []
    if inferred and inferred in corpus:
        candidates.extend((inferred, p, toks) for p, toks, _ in corpus[inferred])
    if not candidates:
        for key, rows in corpus.items():
            candidates.extend((key, p, toks) for p, toks, _ in rows)
    best_path: Path | None = None
    best_key: str | None = inferred
    best_score = 0
    for key, note, note_tokens in candidates:
        overlap = len(png_tokens & note_tokens)
        stem_bonus = 2 if any(t in tokens(note.stem) for t in png_tokens) else 0
        score = overlap + stem_bonus
        if score > best_score:
            best_score = score
            best_path = note
            best_key = key
    if best_score < 5:
        return None, inferred, best_score
    return best_path, best_key, best_score


def append_embed_to_note(note: Path, image: Path, heading: str = "PNG 시각자료") -> bool:
    text = note.read_text(encoding="utf-8")
    embed = f"![[{image.name}]]"
    if embed in text:
        normalized = normalize_png_sections(text)
        if normalized != text:
            note.write_text(nfc(normalized), encoding="utf-8")
        return False
    out = upsert_png_embed(text, embed, heading)
    note.write_text(nfc(out), encoding="utf-8")
    return True


def upsert_png_embed(text: str, embed: str, heading: str = "PNG 시각자료") -> str:
    embeds: list[str] = []
    for block in PNG_SECTION_RE.findall("\n" + text):
        embeds.extend(re.findall(r"!\[\[[^\]]+\.png\]\]", block, flags=re.I))
    if embed not in embeds:
        embeds.append(embed)
    base = PNG_SECTION_RE.sub("\n", "\n" + text).strip()
    return base + f"\n\n## {heading}\n\n" + "\n\n".join(dict.fromkeys(embeds)) + "\n"


def normalize_png_sections(text: str) -> str:
    blocks = PNG_SECTION_RE.findall("\n" + text)
    if not blocks:
        return text
    embeds: list[str] = []
    for block in blocks:
        embeds.extend(re.findall(r"!\[\[[^\]]+\.png\]\]", block, flags=re.I))
    base = PNG_SECTION_RE.sub("\n", "\n" + text).strip()
    if not embeds:
        return base + "\n"
    return base + "\n\n## PNG 시각자료\n\n" + "\n\n".join(dict.fromkeys(embeds)) + "\n"


def normalize_all_png_sections() -> int:
    changed = 0
    for path in VAULT.rglob("*.md"):
        if any(part in {".git", ".obsidian", ".agents", "__pycache__"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        new = normalize_png_sections(text)
        if new != text:
            path.write_text(nfc(new), encoding="utf-8")
            changed += 1
    return changed


def media_index_path(course_label: str) -> str:
    return f"{MEDIA_ROOT}/{slugify(course_label)} PNG 시각자료.md"


def clean_title_fragment(value: str) -> str:
    value = nfc(value)
    value = re.sub(r"!?\[\[[^\]]+\]\]", " ", value)
    value = re.sub(r"[`*=~]+", " ", value)
    value = re.sub(r"[\\/:*?\"<>|#^[\]]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" .,:;!?()[]{}\"'")
    return value[:48].strip()


def ocr_title_fragment(ocr: str) -> str:
    for raw in ocr.splitlines():
        line = clean_title_fragment(raw)
        if len(line) >= 4 and not line.isdigit():
            return line
    return ""


def unique_png_destination(path: Path, stem: str) -> Path:
    stem = slugify(stem, path.stem)
    candidate = path.with_name(f"{stem}{path.suffix}")
    idx = 2
    while candidate.exists() and candidate != path:
        candidate = path.with_name(f"{stem} {idx}{path.suffix}")
        idx += 1
    return candidate


def is_generic_png(path: Path) -> bool:
    stem = nfc(path.stem).strip()
    return bool(GENERIC_PNG_RE.match(stem))


def best_ref_for_png(png: Path, refs: dict[str, list[Ref]], courses: dict) -> tuple[Path | None, str | None]:
    candidates = refs.get(png.name, [])
    ranked: list[tuple[int, Path, str | None]] = []
    for ref in candidates:
        source = ref.source
        course = find_course_for_path(source, courses)
        score = 0
        r = rel(source)
        if source.suffix.lower() == ".md":
            score += 5
        if not r.startswith(KG_ROOT + "/"):
            score += 20
        if course:
            score += 10
        if "archive-kg/media" in r or r.endswith(".canvas"):
            score -= 10
        ranked.append((score, source, course.key if course else None))
    if not ranked:
        return None, infer_course_key(png)
    _score, source, course_key = sorted(ranked, key=lambda item: item[0], reverse=True)[0]
    return source, course_key


def source_title(source: Path) -> str:
    try:
        text = source.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return clean_title_fragment(source.stem)
    title_match = re.search(r"^title:\s*(.+)$", text, flags=re.MULTILINE)
    if title_match:
        title = clean_title_fragment(title_match.group(1).strip("'\" "))
        if title:
            return title
    heading = HEADING_RE.search(text)
    if heading:
        title = clean_title_fragment(heading.group(1))
        if title:
            return title
    return clean_title_fragment(source.stem)


def nearest_heading_for_image(source: Path, image_name: str) -> str:
    try:
        text = source.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    encoded = urllib.parse.quote(image_name)
    positions = [idx for needle in (image_name, encoded) if (idx := text.find(needle)) >= 0]
    if not positions:
        return ""
    pos = min(positions)
    headings = [(m.start(), clean_title_fragment(m.group(1))) for m in HEADING_RE.finditer(text[:pos])]
    headings = [(idx, heading) for idx, heading in headings if heading and heading != "PNG 시각자료"]
    return headings[-1][1] if headings else ""


def planned_png_destination(path: Path, stem: str, reserved: set[str]) -> Path:
    stem = slugify(stem, path.stem)
    candidate = path.with_name(f"{stem}{path.suffix}")
    idx = 2
    while (candidate.exists() and candidate != path) or candidate.name in reserved:
        candidate = path.with_name(f"{stem} {idx}{path.suffix}")
        idx += 1
    reserved.add(candidate.name)
    return candidate


def rewrite_image_refs(mapping: dict[str, str]) -> int:
    if not mapping:
        return 0
    changed = 0
    encoded_mapping = {urllib.parse.quote(old): urllib.parse.quote(new) for old, new in mapping.items()}
    for source in all_text_files():
        try:
            text = source.read_text(encoding="utf-8")
        except Exception:
            continue
        new_text = text
        for old, new in mapping.items():
            new_text = new_text.replace(old, new)
        for old, new in encoded_mapping.items():
            new_text = new_text.replace(old, new)
        if new_text != text:
            source.write_text(nfc(new_text), encoding="utf-8")
            changed += 1
    return changed


def normalize_markdown_image_refs() -> int:
    png_names = {p.name for p in image_files()}
    changed = 0

    def is_local_target(target: str) -> bool:
        target = target.strip()
        lower = target.lower()
        if lower.startswith(("http://", "https://", "data:", "obsidian://")):
            return False
        decoded = urllib.parse.unquote(target)
        return "/image/" in decoded or decoded.startswith("image/") or decoded.startswith("../")

    for source in all_text_files():
        try:
            text = source.read_text(encoding="utf-8")
        except Exception:
            continue

        def replace(match: re.Match) -> str:
            target = match.group(1)
            name = decode_target(target)
            if is_local_target(target) and name in png_names:
                return f"![[{name}]]"
            return match.group(0)

        new_text = MD_IMAGE_RE.sub(replace, text)
        if new_text != text:
            source.write_text(nfc(new_text), encoding="utf-8")
            changed += 1
    return changed


def rename_referenced_generic_pngs(refs: dict[str, list[Ref]], courses: dict) -> tuple[list[str], int, int]:
    planned: list[tuple[Path, Path, str]] = []
    reserved = {p.name for p in image_files() if not is_generic_png(p)}
    ocr_scanned = 0
    for png in image_files():
        if not is_generic_png(png):
            continue
        source, course_key = best_ref_for_png(png, refs, courses)
        if source is None:
            continue
        ocr = ocr_text(png)
        ocr_scanned += 1
        course_prefix = course_key or infer_course_key(png) or "shared-media"
        heading = nearest_heading_for_image(source, png.name)
        fragment = heading or ocr_title_fragment(ocr) or source_title(source)
        source_fragment = source_title(source)
        stem_parts = [course_prefix, source_fragment]
        if fragment and fragment != source_fragment:
            stem_parts.append(fragment)
        destination = planned_png_destination(png, "__".join(stem_parts), reserved)
        if destination != png:
            planned.append((png, destination, f"{rel(png)} -> {rel(destination)}"))
    mapping = {old.name: new.name for old, new, _item in planned}
    rewrite_passes = rewrite_image_refs(mapping)
    renamed: list[str] = []
    for old, new, item in planned:
        if old.exists():
            old.rename(new)
            renamed.append(item)
    return renamed, ocr_scanned, rewrite_passes


def normalized_png_stem(stem: str) -> str:
    parts = [clean_title_fragment(part) for part in nfc(stem).split("__")]
    parts = [part for part in parts if part]
    return slugify("__".join(parts), stem)


def normalize_semantic_png_names() -> tuple[list[str], int]:
    planned: list[tuple[Path, Path, str]] = []
    reserved: set[str] = set()
    for png in image_files():
        clean_stem = normalized_png_stem(png.stem)
        destination = planned_png_destination(png, clean_stem, reserved)
        if destination != png:
            planned.append((png, destination, f"{rel(png)} -> {rel(destination)}"))
    mapping = {old.name: new.name for old, new, _item in planned}
    rewrite_passes = rewrite_image_refs(mapping)
    renamed: list[str] = []
    for old, new, item in planned:
        if old.exists():
            old.rename(new)
            renamed.append(item)
    return renamed, rewrite_passes


def rename_unreferenced_png(path: Path, course_key: str | None, note: Path | None, ocr: str, courses: dict) -> Path:
    if "__" in path.stem and not path.name.startswith("Pasted image "):
        return path
    fragment = ocr_title_fragment(ocr)
    if note is not None:
        course_prefix = course_key or "course"
        stem = f"{course_prefix}__{note.stem}"
        if fragment:
            stem += f"__{fragment}"
    elif course_key and course_key in courses:
        stem = f"{course_key}__{fragment or courses[course_key].label} PNG 시각자료"
    else:
        return path
    destination = unique_png_destination(path, stem)
    if destination != path:
        path.rename(destination)
        return destination
    return path


def write_media_index(course, images: list[tuple[Path, str, str]]) -> bool:
    lines = note_header(f"{course.label} PNG 시각자료", ["type/interface", "pkm/kg-evidence"])
    lines.extend(
        [
            f"kg_parent:: {wikilink(f'{COURSE_ROOT}/{slugify(course.label)} 지식그래프.md', course.label)}",
            f"kg_skeleton:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
            f"kg_course:: {wikilink(f'{COURSE_ROOT}/{slugify(course.label)} 지식그래프.md', course.label)}",
            "",
            f"# {course.label} PNG 시각자료",
            "",
            "OCR/파일명/과목 prefix로 내용 확인 후 과목 단위로 보존한 PNG embed 인덱스입니다.",
            "",
        ]
    )
    for image, size, ocr in images:
        lines.extend(
            [
                f"## {image.stem}",
                "",
                f"- size: `{size or 'unknown'}`",
                f"- ocr: {ocr[:160].replace(chr(10), ' ') if ocr.strip() else '텍스트 없음'}",
                "",
                f"![[{image.name}]]",
                "",
            ]
        )
    return upsert_text_file(media_index_path(course.label), "\n".join(lines))


def main() -> None:
    courses = build_courses()
    refs = scan_refs()
    pngs = image_files()

    by_hash: dict[str, list[Path]] = defaultdict(list)
    for png in pngs:
        by_hash[sha256(png)].append(png)

    duplicate_deleted: list[str] = []
    duplicate_rewrites = 0
    for group in by_hash.values():
        if len(group) < 2:
            continue
        canonical = choose_canonical(group, refs)
        for duplicate in group:
            if duplicate == canonical:
                continue
            duplicate_rewrites += replace_image_refs(duplicate, canonical)
            try:
                duplicate.unlink()
                duplicate_deleted.append(f"{rel(duplicate)} -> {rel(canonical)}")
            except FileNotFoundError:
                pass

    refs = scan_refs()
    pngs = image_files()
    referenced_renamed, referenced_ocr_scanned, referenced_rewrite_passes = rename_referenced_generic_pngs(refs, courses)
    if referenced_renamed:
        refs = scan_refs()
        pngs = image_files()
    normalized_semantic_names, normalized_rewrite_passes = normalize_semantic_png_names()
    if normalized_semantic_names:
        refs = scan_refs()
        pngs = image_files()
    markdown_embed_rewrites = normalize_markdown_image_refs()
    if markdown_embed_rewrites:
        refs = scan_refs()
        pngs = image_files()
    referenced = {IMAGE_DIR / name for name in refs}
    orphans = [p for p in pngs if p not in referenced]

    corpus = note_corpus(courses)
    embedded_to_notes: list[str] = []
    embedded_to_indexes: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    orphan_deleted: list[str] = []
    ocr_scanned = 0
    semantic_renamed: list[str] = []

    for png in orphans:
        ocr = ocr_text(png)
        ocr_scanned += 1
        size = image_size(png)
        note, course_key, score = best_note_for_png(png, ocr, corpus, courses)
        if note is not None:
            before = png
            png = rename_unreferenced_png(png, course_key, note, ocr, courses)
            if png != before:
                semantic_renamed.append(f"{rel(before)} -> {rel(png)}")
            if append_embed_to_note(note, png):
                embedded_to_notes.append(f"{rel(png)} -> {rel(note)} score={score}")
            continue
        if course_key and course_key in courses:
            before = png
            png = rename_unreferenced_png(png, course_key, None, ocr, courses)
            if png != before:
                semantic_renamed.append(f"{rel(before)} -> {rel(png)}")
            embedded_to_indexes[course_key].append((png, size, ocr))
            continue
        png.unlink()
        orphan_deleted.append(rel(png))

    media_indexes_written = 0
    for key, rows in embedded_to_indexes.items():
        if rows and write_media_index(courses[key], rows):
            media_indexes_written += 1

    normalized_png_sections = normalize_all_png_sections()
    refs = scan_refs()
    remaining_pngs = image_files()
    remaining_unembedded = sorted(str(rel(p)) for p in remaining_pngs if p.name not in refs)
    current_duplicate_groups, current_duplicate_files = duplicate_state(remaining_pngs)
    current_generic_names = sum(1 for p in remaining_pngs if is_generic_png(p))

    lines = note_header("PNG 내용 검증 및 정리 리포트", ["type/interface", "pkm/kg-evidence"])
    lines.extend(
        [
            f"kg_skeleton:: {wikilink(SKELETON, '2026 GraphRAG 아카이브 스켈레톤')}",
            "",
            "# PNG 내용 검증 및 정리 리포트",
            "",
            "PNG 파일을 OCR, 크기, exact SHA-256 duplicate, 현재 embed 참조, 과목 prefix/노트 token overlap으로 검증했습니다.",
            "",
            "## Summary",
            "",
            f"- Initial PNG files: {len(pngs) + len(duplicate_deleted)}",
            f"- Exact duplicate files deleted: {len(duplicate_deleted)}",
            f"- Duplicate reference rewrite passes: {duplicate_rewrites}",
            f"- Referenced generic PNGs OCR-scanned: {referenced_ocr_scanned}",
            f"- Referenced generic PNGs semantically renamed: {len(referenced_renamed)}",
            f"- Referenced PNG rewrite passes: {referenced_rewrite_passes}",
            f"- Semantic PNG filenames normalized: {len(normalized_semantic_names)}",
            f"- Semantic PNG normalization rewrite passes: {normalized_rewrite_passes}",
            f"- Local markdown image refs converted to embeds: {markdown_embed_rewrites}",
            f"- Orphan PNGs OCR-scanned: {ocr_scanned}",
            f"- Orphan PNGs semantically renamed: {len(semantic_renamed)}",
            f"- Embedded into source notes: {len(embedded_to_notes)}",
            f"- Course media indexes written: {media_indexes_written}",
            f"- Source notes with PNG sections normalized: {normalized_png_sections}",
            f"- Unmatched orphan PNGs deleted: {len(orphan_deleted)}",
            f"- Remaining PNG files: {len(remaining_pngs)}",
            f"- Remaining unembedded PNG files: {len(remaining_unembedded)}",
            f"- Current exact duplicate groups: {current_duplicate_groups}",
            f"- Current exact duplicate files: {current_duplicate_files}",
            f"- Current generic/Pasted PNG filenames: {current_generic_names}",
            "",
            "## Deleted exact duplicates",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in duplicate_deleted[:300])
    if len(duplicate_deleted) > 300:
        lines.append(f"- ... {len(duplicate_deleted) - 300} more")
    lines.extend(["", "## Semantically renamed referenced PNGs", ""])
    lines.extend(f"- `{item}`" for item in referenced_renamed[:300])
    if len(referenced_renamed) > 300:
        lines.append(f"- ... {len(referenced_renamed) - 300} more")
    lines.extend(["", "## Normalized semantic PNG filenames", ""])
    lines.extend(f"- `{item}`" for item in normalized_semantic_names[:300])
    if len(normalized_semantic_names) > 300:
        lines.append(f"- ... {len(normalized_semantic_names) - 300} more")
    lines.extend(["", "## Semantically renamed orphan PNGs", ""])
    lines.extend(f"- `{item}`" for item in semantic_renamed[:300])
    if len(semantic_renamed) > 300:
        lines.append(f"- ... {len(semantic_renamed) - 300} more")
    lines.extend(["", "## Embedded into source notes", ""])
    lines.extend(f"- `{item}`" for item in embedded_to_notes[:300])
    if len(embedded_to_notes) > 300:
        lines.append(f"- ... {len(embedded_to_notes) - 300} more")
    lines.extend(["", "## Deleted unmatched orphan PNGs", ""])
    lines.extend(f"- `{item}`" for item in orphan_deleted[:300])
    if len(orphan_deleted) > 300:
        lines.append(f"- ... {len(orphan_deleted) - 300} more")
    lines.extend(["", "## Remaining unembedded PNGs", ""])
    lines.extend(f"- `{item}`" for item in remaining_unembedded[:300])
    if len(remaining_unembedded) > 300:
        lines.append(f"- ... {len(remaining_unembedded) - 300} more")
    upsert_text_file(REPORT, "\n".join(lines))

    print(
        "png_content_embed_cleanup "
        f"duplicates_deleted={len(duplicate_deleted)} duplicate_rewrites={duplicate_rewrites} "
        f"referenced_ocr_scanned={referenced_ocr_scanned} "
        f"referenced_renamed={len(referenced_renamed)} referenced_rewrites={referenced_rewrite_passes} "
        f"normalized_semantic_names={len(normalized_semantic_names)} normalized_rewrites={normalized_rewrite_passes} "
        f"markdown_embed_rewrites={markdown_embed_rewrites} "
        f"orphans_scanned={ocr_scanned} embedded_notes={len(embedded_to_notes)} "
        f"semantic_renamed={len(semantic_renamed)} "
        f"media_indexes={media_indexes_written} normalized_sections={normalized_png_sections} "
        f"orphan_deleted={len(orphan_deleted)} "
        f"remaining_png={len(remaining_pngs)} remaining_unembedded={len(remaining_unembedded)}"
    )


if __name__ == "__main__":
    main()
