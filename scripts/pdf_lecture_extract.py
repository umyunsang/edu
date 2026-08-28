#!/usr/bin/env python3
"""
pdf_lecture_extract.py — 강의 PDF를 "LLM이 읽을 수 있는 추출 번들"로 변환한다.

이 스크립트는 마크다운 정리문서를 직접 쓰지 않는다.
정리문서 작성은 에이전트(Claude Code / Codex / OpenKnowledge MCP)의 몫이고,
이 스크립트는 그 에이전트가 읽을 수 있는 **근거 자료**를 만든다.

출력 번들 (기본: .ok/local/pdf-extract/<도메인>__<과목>__<slug>/)
  meta.json      PDF 메타데이터 + sha256 + 페이지별 텍스트 밀도 + 렌더링된 페이지 목록
  text.md        페이지별 레이아웃 보존 텍스트 (```page N ... ``` 블록)
  pages/pNNN.png 텍스트가 빈약한(=내용이 이미지에 있는) 페이지만 선택적으로 렌더링

의존성: poppler-utils (pdftotext, pdftoppm, pdfinfo). 파이썬 서드파티 패키지 없음.

사용법:
  python3 scripts/pdf_lecture_extract.py <pdf ...> [--out DIR] [--dpi 110]
                                         [--min-chars 180] [--max-render 40]
                                         [--render all|sparse|none] [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

DEFAULT_OUT = Path(".ok/local/pdf-extract")
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def die(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def require_tools() -> None:
    missing = [t for t in ("pdftotext", "pdftoppm", "pdfinfo") if not shutil.which(t)]
    if missing:
        die(f"poppler 도구 없음: {', '.join(missing)} (brew install poppler)")


def slugify(name: str) -> str:
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r"[\\/:*?\"<>|]+", "-", name)
    name = re.sub(r"\s+", "-", name.strip())
    return name[:120] or "untitled"


def bundle_key(pdf: Path) -> str:
    """번들 디렉토리 이름 = ``<도메인>__<과목>__<파일slug>``.

    PDF 파일명만으로 키를 만들면 과목이 다른 동명 파일이 서로를 덮어쓴다.
    실측(2026-08-28): 476개 중 ``2장_확인문제`` 가 open-source-software 와
    web-programming 사이에서 1건 충돌한다. 과목 경로를 접두로 붙여 격리한다.

    ``<과목>/sources/`` 밖의 PDF 는 접두를 붙일 근거가 없으므로 파일 slug 만 쓴다.
    """
    stem = slugify(pdf.stem)
    parent = pdf.parent
    if parent.name != "sources":
        return stem
    subject = parent.parent.name
    domain = parent.parent.parent.name
    prefix = "__".join(p for p in (domain, subject) if p)
    return f"{slugify(prefix)}__{stem}" if prefix else stem


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lfs_oid(path: Path) -> str | None:
    """LFS 포인터 파일이면 그 안에 적힌 sha256 oid를 그대로 돌려준다."""
    try:
        head = path.open("rb").read(200)
    except OSError:
        return None
    if not head.startswith(LFS_POINTER_PREFIX):
        return None
    m = re.search(rb"oid sha256:([0-9a-f]{64})", head)
    return m.group(1).decode() if m else None


def pdf_info(path: Path) -> dict:
    out = subprocess.run(
        ["pdfinfo", str(path)], capture_output=True, text=True, errors="replace"
    ).stdout
    info: dict[str, str] = {}
    for line in out.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            info[k.strip()] = v.strip()
    return info


def page_texts(path: Path, pages: int) -> list[str]:
    texts: list[str] = []
    for n in range(1, pages + 1):
        res = subprocess.run(
            ["pdftotext", "-layout", "-f", str(n), "-l", str(n), str(path), "-"],
            capture_output=True,
            text=True,
            errors="replace",
        )
        texts.append(res.stdout)
    return texts


def content_chars(text: str) -> int:
    """글머리 기호/공백/페이지 번호만 있는 슬라이드를 '빈 슬라이드'로 취급한다."""
    stripped = re.sub(r"[\s\u00a0]", "", text)
    stripped = re.sub(r"[•▪◦●○➢➔⚫·\-–—*∙]", "", stripped)
    stripped = re.sub(r"^\d{1,3}$", "", stripped)
    return len(stripped)


def render_pages(path: Path, page_nums: list[int], dest: Path, dpi: int) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    for n in page_nums:
        prefix = dest / f"p{n:03d}"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-f", str(n), "-l", str(n),
             "-singlefile", str(path), str(prefix)],
            capture_output=True,
        )
        png = prefix.with_suffix(".png")
        if png.exists():
            made.append(png.name)
    return made


def extract(pdf: Path, out_root: Path, dpi: int, min_chars: int,
            max_render: int, render_mode: str) -> dict:
    if not pdf.exists():
        die(f"파일 없음: {pdf}")

    oid = lfs_oid(pdf)
    if oid is not None:
        die(f"{pdf}: Git LFS 포인터 상태입니다. 먼저 `git lfs pull --include=\"{pdf}\"` 하세요.")

    info = pdf_info(pdf)
    try:
        pages = int(info.get("Pages", "0"))
    except ValueError:
        pages = 0
    if pages <= 0:
        die(f"{pdf}: 페이지 수를 읽지 못했습니다 (손상 PDF?)")

    bundle = out_root / bundle_key(pdf)
    bundle.mkdir(parents=True, exist_ok=True)

    texts = page_texts(pdf, pages)
    densities = [content_chars(t) for t in texts]

    if render_mode == "all":
        want = list(range(1, pages + 1))
    elif render_mode == "none":
        want = []
    else:  # sparse: 내용이 이미지에 들어있는 페이지만
        want = [i + 1 for i, d in enumerate(densities) if d < min_chars]
    if max_render >= 0:
        want = want[:max_render]

    rendered = render_pages(pdf, want, bundle / "pages", dpi) if want else []

    lines = [
        f"# {pdf.name}",
        "",
        f"- source_path: `{pdf}`",
        f"- pages: {pages}",
        f"- pdf_title: {info.get('Title', '')}",
        f"- created: {info.get('CreationDate', '')}",
        "",
        "> 각 `page N` 블록은 pdftotext -layout 원문입니다. 밀도가 낮은 페이지는",
        "> 내용이 이미지 안에 있으므로 `pages/pNNN.png` 를 함께 보십시오.",
        "",
    ]
    for i, t in enumerate(texts, start=1):
        img = f"pages/p{i:03d}.png"
        has_img = (bundle / img).exists()
        lines.append(f"```page {i} chars={densities[i-1]}"
                     + (f" image={img}" if has_img else "") + "\n"
                     + t.rstrip() + "\n```")
        lines.append("")
    (bundle / "text.md").write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "schema": "edu.pdf-extract.v1",
        "source_path": str(pdf),
        "source_name": pdf.name,
        "sha256": file_sha256(pdf),
        "bytes": pdf.stat().st_size,
        "pages": pages,
        "pdf_title": info.get("Title", ""),
        "pdf_author": info.get("Author", ""),
        "pdf_created": info.get("CreationDate", ""),
        "page_chars": densities,
        "text_poor_pages": [i + 1 for i, d in enumerate(densities) if d < min_chars],
        "rendered_pages": rendered,
        "render_mode": render_mode,
        "dpi": dpi,
        "min_chars": min_chars,
        "bundle": str(bundle),
    }
    (bundle / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--min-chars", type=int, default=180,
                    help="이 글자 수 미만이면 '내용이 이미지에 있는 페이지'로 간주")
    ap.add_argument("--max-render", type=int, default=40,
                    help="PDF 하나당 렌더링할 최대 페이지 수 (-1이면 무제한)")
    ap.add_argument("--render", choices=["all", "sparse", "none"], default="sparse")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    require_tools()
    results = [
        extract(p, args.out, args.dpi, args.min_chars, args.max_render, args.render)
        for p in args.pdfs
    ]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for m in results:
            print(f"{m['source_name']}: {m['pages']}p, "
                  f"이미지필요 {len(m['text_poor_pages'])}p, "
                  f"렌더링 {len(m['rendered_pages'])}p -> {m['bundle']}")


if __name__ == "__main__":
    main()
