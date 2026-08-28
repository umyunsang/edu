#!/usr/bin/env python3
"""normalize_course_layout.py — 과목 폴더를 docs/course-layout.md 표준으로 옮긴다.

  <course>/00. 인덱스.md · NN. 주제.md   정리문서 (루트 평면)
  <course>/pdf/                          원본 강의자료
  <course>/code/                         실습 코드
  <course>/assets/                       기타 첨부

기존 .md 는 재작성 대상이므로 --delete-md 를 줄 때만 지운다.
git mv 를 쓰므로 이력이 보존되고, sparse-checkout 저장소에서도 동작한다.

사용법:
  python3 scripts/normalize_course_layout.py <course-dir> [--apply] [--delete-md]
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

PDF   = {'.pdf'}
CODE  = {'.ipynb', '.py', '.sql', '.java', '.c', '.cpp', '.h', '.js', '.ts', '.sh', '.r', '.m'}
ASSET = {'.docx', '.xlsx', '.pptx', '.ppt', '.csv', '.tsv', '.png', '.jpg', '.jpeg',
         '.gif', '.webp', '.zip', '.mp4', '.mov', '.txt', '.json', '.yaml', '.yml'}
SKIP_DIR = {'.git', '.ok', '.omo', '.obsidian', '.planning', '.remember', '.codex',
            '.gjc', '.claude', '.pi', '.opencode', '.agents', '.cursor', '.github',
            '.codegraph', '.ruff_cache', '.superpowers', '.playwright-mcp', '.render',
            '.vault-meta', '.pytest_cache', '__pycache__'}
BUCKET_DIRS = {'pdf', 'code', 'assets'}


def git(*args: str, cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True)


def plan(course: Path, repo: str, delete_md: bool):
    moves: list[tuple[Path, Path]] = []
    deletes: list[Path] = []
    taken: set[str] = set()

    for root, dirs, files in os.walk(course):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR and not d.startswith('.')]
        rel_root = Path(root).relative_to(course)
        top = rel_root.parts[0] if rel_root.parts else ''
        for name in sorted(files):
            if name == '.DS_Store':
                continue
            src = Path(root) / name
            ext = src.suffix.lower()

            if ext == '.md':
                if delete_md:
                    deletes.append(src)
                continue

            if ext in PDF:
                bucket = 'pdf'
            elif ext in CODE:
                bucket = 'code'
            elif ext in ASSET:
                bucket = 'assets'
            else:
                continue

            # 이미 올바른 자리면 건너뛴다
            if top == bucket and len(rel_root.parts) == 1:
                taken.add(f'{bucket}/{name}')
                continue

            # 이름 충돌 시 원래 상위 폴더명을 접두로
            target_name = name
            key = f'{bucket}/{target_name}'
            if key in taken or (course / bucket / target_name).exists():
                prefix = rel_root.parts[0] if rel_root.parts else 'root'
                prefix = prefix.replace('/', '_').strip()
                target_name = f'{prefix}__{name}'
                key = f'{bucket}/{target_name}'
            taken.add(key)
            moves.append((src, course / bucket / target_name))

    return moves, deletes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('course')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--delete-md', action='store_true')
    a = ap.parse_args()

    course = Path(a.course).resolve()
    if not course.is_dir():
        print(f'error: not a directory: {course}', file=sys.stderr)
        return 2
    repo = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=str(course),
                          capture_output=True, text=True).stdout.strip()
    moves, deletes = plan(course, repo, a.delete_md)

    print(f'course : {course.relative_to(repo)}')
    print(f'moves  : {len(moves)}')
    for s, d in moves[:200]:
        print(f'  {s.relative_to(course)}  ->  {d.relative_to(course)}')
    if deletes:
        print(f'deletes: {len(deletes)} (.md)')
        for p in deletes[:200]:
            print(f'  {p.relative_to(course)}')
    if not a.apply:
        print('\n(dry-run — --apply 로 실제 수행)')
        return 0

    for bucket in BUCKET_DIRS:
        (course / bucket).mkdir(exist_ok=True)
    ok = fail = 0
    for s, d in moves:
        d.parent.mkdir(parents=True, exist_ok=True)
        r = git('mv', '--sparse', str(s), str(d), cwd=repo)
        if r.returncode != 0:
            r = git('mv', str(s), str(d), cwd=repo)
        if r.returncode == 0:
            ok += 1
        else:
            fail += 1
            print(f'  move failed: {s.relative_to(course)} :: {r.stderr.strip()[:90]}')
    for p in deletes:
        r = git('rm', '-q', '--sparse', str(p), cwd=repo)
        if r.returncode != 0:
            git('rm', '-q', str(p), cwd=repo)
    # 빈 폴더 제거
    removed = 0
    for root, dirs, files in os.walk(course, topdown=False):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR and not d.startswith('.')]
        p = Path(root)
        if p == course:
            continue
        try:
            if not any(p.iterdir()):
                p.rmdir()
                removed += 1
        except OSError:
            pass
    print(f'\nmoved {ok}, failed {fail}, deleted {len(deletes)}, empty dirs removed {removed}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
