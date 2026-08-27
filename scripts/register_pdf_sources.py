#!/usr/bin/env python3
"""
register_pdf_sources.py — 강의 PDF를 claude-obsidian 출처 원장에 등록한다.

`wiki/meta/ledgers/source-ledger.json` 은 "이 아카이브의 지식이 어떤 원본에서 나왔는가"를
기계가 읽을 수 있게 기록하는 곳이다. 결과는 `claude-obsidian.source-ledger.v1` 스키마를
정확히 만족하므로 `claude-obsidian lint` 가 그대로 통과한다.

## 이 저장소에서의 두 가지 특수 사정

1. **PDF는 전부 Git LFS로 관리된다.** 작업 트리의 파일이 포인터 텍스트일 수 있으므로
   `origin.kind` 는 `file` 이 아니라 `manual` 을 쓴다. `file` 로 두면 lint가 작업 트리
   바이트를 해싱해 LFS 포인터와 비교하다가 전부 불일치로 잡는다.
   진짜 내용 해시는 포인터의 `oid sha256:` 에서 읽으므로 **전체 다운로드가 필요 없다.**

2. **수업 폴더의 PDF가 전부 강의자료인 것은 아니다.** Obsidian Better Export PDF로
   내보낸 노트 파생물이 섞여 있다. 이걸 `authority: primary` 로 등록하면
   "자기가 쓴 글을 근거로 자기가 쓴 글"이 되어 출처 원장이 무의미해진다.
   동명의 `.md` 가 옆에 있거나 PDF 메타데이터에 obsidian이 있으면 `secondary` 로 낮춘다.

또한 각 정리문서의 프론트매터(`source:`)를 읽어 "이 원본이 어떤 노트를 뒷받침하는가"를
`vault_pages` 에 역으로 채운다. (스키마의 `pages` 는 `wiki/` 하위 경로만 허용하므로
과목 폴더의 노트는 커스텀 필드에 넣는다.)

사용법:
  python3 scripts/register_pdf_sources.py                 # 저장소 전체
  python3 scripts/register_pdf_sources.py --course "ComputerScience/.../parallel-distributed-computing"
  python3 scripts/register_pdf_sources.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "wiki" / "meta" / "ledgers" / "source-ledger.json"
SCHEMA = "claude-obsidian.source-ledger.v1"
LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"
ORIGIN_KIND = "manual"          # LFS 포인터일 수 있으므로 file 이 아니다 (모듈 docstring 참고)
REFRESH_YEARS = 3               # 강의자료는 개정이 드물다


def stable_source_id(kind: str, locator: str, sha: str | None) -> str:
    """claude-obsidian `stable_source_id` 와 동일한 규칙으로 ID를 만든다."""
    normalized = PurePosixPath(locator).as_posix() if kind == "file" else locator
    digest = hashlib.sha256(
        f"{kind.casefold()}\0{normalized}\0{(sha or '').casefold()}".encode(
            "utf-8", errors="surrogatepass"
        )
    ).hexdigest()
    return f"src-{digest[:20]}"


def content_hash(path: Path) -> tuple[str, int, str]:
    """(sha256, bytes, 획득 방법). LFS 포인터면 그 안의 oid를 그대로 쓴다."""
    head = path.open("rb").read(200)
    if head.startswith(LFS_PREFIX):
        text = path.read_text(encoding="utf-8", errors="replace")
        oid = re.search(r"oid sha256:([0-9a-f]{64})", text)
        size = re.search(r"size (\d+)", text)
        if oid:
            return oid.group(1), int(size.group(1)) if size else 0, "lfs-pointer"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), path.stat().st_size, "file-bytes"


def classify(pdf: Path) -> tuple[str, str]:
    """(source_type, authority) 판정. 모듈 docstring의 2번 사정 참고."""
    if pdf.with_suffix(".md").exists():
        return "obsidian-export", "secondary"
    try:
        head = pdf.open("rb").read(200)
    except OSError:
        return "lecture-slides", "primary"
    if not head.startswith(LFS_PREFIX):
        out = subprocess.run(
            ["pdfinfo", str(pdf)], capture_output=True, text=True, errors="replace"
        ).stdout.lower()
        if "obsidian" in out:
            return "obsidian-export", "secondary"
    return "lecture-slides", "primary"


def frontmatter(path: Path) -> dict[str, str]:
    """의존성 없는 최소 YAML 프론트매터 파서 (평면 스칼라만 읽는다)."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    out: dict[str, str] = {}
    for line in text[3:end].splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
        if m and m.group(2):
            out[m.group(1)] = m.group(2).strip().strip("'\"")
    return out


def build_backlinks() -> dict[str, list[str]]:
    """정리문서 -> 근거 PDF 역인덱스."""
    links: dict[str, list[str]] = {}
    for md in ROOT.rglob("*.md"):
        if any(p in {".git", ".ok", "node_modules", ".agents", ".claude"} for p in md.parts):
            continue
        src = frontmatter(md).get("source", "")
        if not src.lower().endswith(".pdf"):
            continue
        try:
            key = (md.parent / src).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            continue
        links.setdefault(key, []).append(md.relative_to(ROOT).as_posix())
    return links


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", action="append", default=[],
                    help="특정 폴더만 처리 (저장소 루트 기준 상대경로)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scopes = [ROOT / c for c in args.course] or [ROOT]
    pdfs: list[Path] = []
    for scope in scopes:
        pdfs.extend(scope.rglob("*.pdf"))
    pdfs = sorted(
        p for p in pdfs
        if not re.search(r" 2\.pdf$", p.name)            # macOS 중복 사본
        and ".ok/" not in p.as_posix()
        and "/.git/" not in p.as_posix()
    )

    backlinks = build_backlinks()

    # lint의 감사 기준일은 UTC 날짜다. retrieved_at 이 그보다 미래면 오류가 난다.
    today = datetime.now(timezone.utc).date()
    retrieved = today.isoformat()
    refresh_due = (today + timedelta(days=365 * REFRESH_YEARS)).isoformat()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    previous: dict[str, dict] = {}
    if LEDGER.exists():
        try:
            previous = json.loads(LEDGER.read_text(encoding="utf-8")).get("sources", {})
        except json.JSONDecodeError:
            previous = {}

    sources: dict[str, dict] = {}
    for pdf in pdfs:
        rel = pdf.relative_to(ROOT).as_posix()
        sha, size, how = content_hash(pdf)
        stype, authority = classify(pdf)
        sid = stable_source_id(ORIGIN_KIND, rel, sha)
        parts = PurePosixPath(rel).parts
        course = parts[2] if len(parts) > 3 and parts[0] == "ComputerScience" else parts[0]

        sources[sid] = {
            "origin": {"kind": ORIGIN_KIND, "locator": rel},
            "title": PurePosixPath(rel).name,
            "content_kind": "document",
            "authority": authority,
            "review_status": "active",
            "content_sha256": sha,
            "retrieved_at": previous.get(sid, {}).get("retrieved_at", retrieved),
            "refresh_due": refresh_due,
            "pages": [],
            "independence_key": f"course:{course}",
            "supersedes": None,
            # --- 이 저장소 전용 확장 필드 ---
            "source_type": stype,
            "bytes": size,
            "hash_origin": how,
            "vault_pages": sorted(backlinks.get(rel, [])),
        }

    ledger = {"schema": SCHEMA, "generated_at": now,
              "sources": dict(sorted(sources.items()))}

    added = len(set(sources) - set(previous))
    removed = len(set(previous) - set(sources))
    linked = sum(1 for s in sources.values() if s["vault_pages"])
    derived = sum(1 for s in sources.values() if s["source_type"] == "obsidian-export")
    lfs = sum(1 for s in sources.values() if s["hash_origin"] == "lfs-pointer")

    print(f"PDF {len(pdfs)}개 · 원장 {len(sources)}건 (신규 {added} / 사라짐 {removed})")
    print(f"  강의자료 {len(sources) - derived}건 · 노트 내보내기 파생물 {derived}건")
    print(f"  정리문서와 연결된 원본 {linked}건 · LFS 포인터에서 해시 취득 {lfs}건")

    if args.dry_run:
        print("(dry-run: 파일을 쓰지 않았습니다)")
        return
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"기록: {LEDGER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
