#!/usr/bin/env python3
"""Use PDF/folder evidence to repair weak or empty note identities."""
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

RENAMES = {
    "ComputerScience/01_programming-foundations/python-programming/문제풀이/문제풀이 1~10.md": (
        "ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 1-10.md",
        "프로그래머스 Python 기초 문제 1-10",
        "동봉 PDF `문제풀이_1_10.pdf` 기준: 문자열 p/y 개수, 하샤드 수 등 Python 기초 알고리즘 문제 1-10 풀이 묶음.",
    ),
    "ComputerScience/01_programming-foundations/python-programming/문제풀이/문제풀이 10~20.md": (
        "ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 11-20.md",
        "프로그래머스 Python 기초 문제 11-20",
        "동봉 PDF `문제풀이_11_20.pdf` 기준: 직사각형 별찍기, 가운데 글자 가져오기 등 Python 기초 알고리즘 문제 11-20 풀이 묶음.",
    ),
    "ComputerScience/04_systems-infrastructure/operating-systems/과제/Page/page.c.md": (
        "ComputerScience/04_systems-infrastructure/operating-systems/과제/Page/페이지 교체 알고리즘 구현 과제.md",
        "페이지 교체 알고리즘 구현 과제",
        "동봉 PDF `페이지교체 알고리즘.pdf`와 폴더명 기준: 운영체제 가상 메모리의 페이지 교체 알고리즘 구현 과제.",
    ),
    "ComputerScience/04_systems-infrastructure/operating-systems/과제/Banker/banker.c.md": (
        "ComputerScience/04_systems-infrastructure/operating-systems/과제/Banker/Banker Algorithm 구현 과제.md",
        "Banker Algorithm 구현 과제",
        "동봉 PDF `BankerAlgorithm.pdf` 기준: 교착상태 회피를 위한 Banker Algorithm 구현 과제.",
    ),
    "ComputerScience/04_systems-infrastructure/operating-systems/과제/SJF/sjf.c.md": (
        "ComputerScience/04_systems-infrastructure/operating-systems/과제/SJF/SJF CPU 스케줄링 구현 과제.md",
        "SJF CPU 스케줄링 구현 과제",
        "동봉 PDF `SJF 과제.pdf` 기준: Shortest Job First CPU 스케줄링 구현 과제.",
    ),
    "ComputerScience/04_systems-infrastructure/operating-systems/과제/SRTF/srtf.c.md": (
        "ComputerScience/04_systems-infrastructure/operating-systems/과제/SRTF/SRTF CPU 스케줄링 구현 과제.md",
        "SRTF CPU 스케줄링 구현 과제",
        "동봉 PDF `SRTF 과제.pdf` 기준: Shortest Remaining Time First CPU 스케줄링 구현 과제.",
    ),
    "ComputerScience/04_systems-infrastructure/operating-systems/과제/MemoryAlloc/allocation.c.md": (
        "ComputerScience/04_systems-infrastructure/operating-systems/과제/MemoryAlloc/메모리 할당 알고리즘 구현 과제.md",
        "메모리 할당 알고리즘 구현 과제",
        "동봉 PDF `MemoryAlloc.pdf` 기준: 운영체제 메모리 할당 알고리즘 구현 과제.",
    ),
    "ComputerScience/04_systems-infrastructure/operating-systems/과제/FCFS/fcfs.c.md": (
        "ComputerScience/04_systems-infrastructure/operating-systems/과제/FCFS/FCFS CPU 스케줄링 구현 과제.md",
        "FCFS CPU 스케줄링 구현 과제",
        "동봉 PDF `FCFS 과제.pdf` 기준: First-Come First-Served CPU 스케줄링 구현 과제.",
    ),
    "ComputerScience/04_systems-infrastructure/computer-networks/16. 보안/보안.md": (
        "ComputerScience/04_systems-infrastructure/computer-networks/16. 보안/네트워크 보안.md",
        "네트워크 보안",
        "동봉 PDF `16장 보안.pdf` 기준: 컴퓨터 네트워크 16장 보안 단원 요약.",
    ),
    "ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/인공지능_중간고사_엄윤상_1705817.md": (
        "ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험.md",
        "CIFAR10 MLP 이미지 분류 중간 실습시험",
        "동봉 PDF `24_인공지능_중간고사.pdf` 기준: CIFAR10 데이터셋을 활용한 MLP 기반 이미지 분류 네트워크 설계 실습시험.",
    ),
    "ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/올림픽 요구사항.md": (
        "ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/음성 인식 고객 추가 요구사항.md",
        "음성 인식 고객 추가 요구사항",
        "동봉 PDF `고객님의추가요구사항.pdf` 기준: 음성을 텍스트로 변환하는 웹 프로그램에 고객 추가 요구사항을 반영하는 과제.",
    ),
    "ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/과제1.md": (
        "ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 이미지 스타일 변환 과제.md",
        "생성형 AI 이미지 스타일 변환 과제",
        "동봉 PDF `_GenAI_FineTuning.pdf`와 본문 이미지 기준: 사진 스타일 변환, 4컷 카툰, 의상 변경, 이모티콘 제작 실습.",
    ),
    "ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/과제2.md": (
        "ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/Civitai LoRA 실내공간 스타일 생성 과제.md",
        "Civitai LoRA 실내공간 스타일 생성 과제",
        "동봉 PDF `_GenAI_FineTuning.pdf`와 본문 링크 기준: Civitai 모델을 활용한 실내공간 스타일 생성 실습.",
    ),
    "ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/프로젝트 주제.md": (
        "ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 파인튜닝 프로젝트 주제.md",
        "생성형 AI 파인튜닝 프로젝트 주제",
        "동봉 PDF `_GenAI_FineTuning.pdf` 기준: 역사 재현, 동화 각색, 미래 시나리오 등 멀티미디어 생성형 AI 파인튜닝 프로젝트 후보.",
    ),
    "ComputerScience/03_ai-ml-data/neural-networks/md/Ch2. 퍼셉트론 상세 정리.md": (
        "ComputerScience/03_ai-ml-data/neural-networks/md/2장 퍼셉트론 상세 정리.md",
        "2장 퍼셉트론 상세 정리",
        "동봉 강의 PDF `2장_퍼셉트론.pdf` 기준: 퍼셉트론 구조와 논리 연산 학습 단원.",
    ),
    "ComputerScience/03_ai-ml-data/neural-networks/md/Ch4. 경사 하강법 최적화.md": (
        "ComputerScience/03_ai-ml-data/neural-networks/md/4장 신경망 학습과 경사 하강법.md",
        "4장 신경망 학습과 경사 하강법",
        "동봉 강의 PDF `AIE309_4장_신경망학습.pdf` 기준: 손실함수, 미분, 경사 하강법을 중심으로 한 신경망 학습 단원.",
    ),
    "ComputerScience/05_software-engineering/programming-languages/과제/4장 연습문제.md": (
        "ComputerScience/05_software-engineering/programming-languages/과제/4장 재귀 하강 파서 연습문제.md",
        "4장 재귀 하강 파서 연습문제",
        "본문 REPORT 기준: 프로그래밍언어론 4장 재귀 하강 파서 파싱 과정 과제.",
    ),
    "ComputerScience/05_software-engineering/database-systems/0. 시험/final.md": (
        "ComputerScience/05_software-engineering/database-systems/0. 시험/기말시험 범위 및 연습문제.md",
        "기말시험 범위 및 연습문제",
        "동봉 PDF `answer_4577.pdf`와 본문 기준: 데이터베이스 개론 기말 범위 및 교재 연습문제 해답 참고 노트.",
    ),
}

TITLE_ONLY = {
    "ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍.md": "객체 지향 프로그래밍",
    "ComputerScience/01_programming-foundations/coding-test/정렬/1. 버블 정렬.md": "버블 정렬",
    "ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/4. 조합 논리 회로.md": "조합 논리 회로",
    "ComputerScience/04_systems-infrastructure/computer-architecture/중간 시험 범위.md": "컴퓨터구조 중간 시험 범위",
    "ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/과제_CacheFriendly코딩실습.md": "Cache Friendly 코딩 실습",
}

SUMMARY_ONLY = {
    "ComputerScience/01_programming-foundations/coding-test/정렬/1. 버블 정렬.md": "정렬 기초 중 버블 정렬을 코딩 테스트 관점에서 다루는 노트.",
    "ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/4. 조합 논리 회로.md": "동봉 PDF `4.조합 논리 회로.pdf` 기준: 조합 논리 회로의 기본 구조와 컴퓨터구조 중간 범위.",
    "ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/시스템 구성도.md": "주문 및 결제 AI 시스템 개발 노트 기준: 스마트 오더 플랫폼의 백엔드, 데이터베이스, LLM/RAG, 추천 시스템, 결제 모듈 구성도.",
    "ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/주문 생성.md": "주문 및 결제 AI 시스템 개발 노트 기준: 사용자가 메뉴를 선택한 뒤 주문 레코드와 결제 흐름으로 이어지는 데이터 흐름.",
    "ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회.md": "주문 및 결제 AI 시스템 개발 노트 기준: 메뉴 데이터가 벡터 저장소와 AI 챗봇 응답에 사용되는 조회 흐름.",
    "ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/장바구니에 메뉴 추가.md": "주문 및 결제 AI 시스템 개발 노트 기준: 추천/조회된 메뉴가 장바구니 상태로 반영되는 데이터 흐름.",
    "ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/AI 메뉴 추천.md": "주문 및 결제 AI 시스템 개발 노트 기준: 사용자 자연어 입력, 메뉴 벡터 저장소, 추천 알고리즘이 연결되는 AI 메뉴 추천 흐름.",
}

COURSE_BY_LEAF = {
    "coding-basics": "coding-basics",
    "python-programming": "python-programming",
    "data-structures": "data-structures",
    "coding-test": "coding-test",
    "java-programming": "java-programming",
    "probability-statistics": "probability-statistics",
    "discrete-mathematics": "discrete-mathematics",
    "optimization-math": "optimization-math",
    "mathematical-logic": "mathematical-logic",
    "artificial-intelligence": "artificial-intelligence",
    "machine-learning": "machine-learning",
    "ml-projects": "ml-projects",
    "neural-networks": "neural-networks",
    "big-data-analysis": "big-data-analysis",
    "computer-vision": "computer-vision",
    "large-language-models": "large-language-models",
    "ai-system-design": "ai-system-design",
    "generative-ai-fine-tuning": "generative-ai-fine-tuning",
    "linux": "linux",
    "computer-architecture": "computer-architecture",
    "operating-systems": "operating-systems",
    "computer-networks": "computer-networks",
    "parallel-distributed-computing": "parallel-distributed-computing",
    "container-orchestration": "container-orchestration",
    "web-programming": "web-programming",
    "database-systems": "database-systems",
    "open-source-software": "open-source-software",
    "programming-languages": "programming-languages",
    "aioss-open-source-delivery": "aioss-open-source-delivery",
    "algorithm-design-analysis": "algorithm-design-analysis",
    "computer-graphics": "computer-graphics",
    "intellectual-property": "intellectual-property",
    "creative-writing": "creative-writing",
    "classics-reading": "classics-reading",
    "degree-portfolio": "degree-portfolio",
}

WIKI_RE = re.compile(r"(!?)\[\[([^\]#|]+)(#[^\]|]*)?(\|[^\]]*)?\]\]")
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.S)
RELATION_RE = re.compile(r"^(?:up|siblings|related|prerequisites|next|central)::.*$", re.M)


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def rel(path: Path) -> str:
    return nfc(path.relative_to(VAULT).as_posix())


def quote_yaml(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def iter_text_files() -> list[Path]:
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


def rewrite_wikilinks(text: str, mapping: dict[str, tuple[str, str]]) -> str:
    def repl(match: re.Match[str]) -> str:
        bang, target, section, alias = match.groups()
        target = nfc(target.strip())
        mapped = mapping.get(target) or mapping.get(target.removesuffix(".md"))
        if mapped is None:
            return match.group(0)
        new_target, new_alias = mapped
        if alias:
            old_alias = alias[1:]
            old_stem = target.rsplit("/", 1)[-1].removesuffix(".md")
            if old_alias in {old_stem, target, target.removesuffix(".md")}:
                alias = "|" + new_alias
        return f"{bang}[[{new_target}{section or ''}{alias or ''}]]"

    return WIKI_RE.sub(repl, text)


def set_frontmatter(path: Path, title: str | None = None, course: str | None = None) -> bool:
    text = nfc(path.read_text(encoding="utf-8"))
    match = FRONTMATTER_RE.match(text)
    if not match:
        return False
    body_start = match.end()
    lines = match.group(1).splitlines()
    out: list[str] = []
    title_done = title is None
    course_done = course is None
    for line in lines:
        if title is not None and line.startswith("title:"):
            out.append(f"title: {quote_yaml(title)}")
            title_done = True
        elif course is not None and line.startswith("course:"):
            out.append(f"course: {course}")
            course_done = True
        else:
            out.append(line)
    if not title_done and title is not None:
        out.append(f"title: {quote_yaml(title)}")
    if not course_done and course is not None:
        out.append(f"course: {course}")
    new_text = "---\n" + "\n".join(out) + "\n---\n\n" + text[body_start:].lstrip("\n")
    if new_text != text:
        path.write_text(nfc(new_text), encoding="utf-8")
        return True
    return False


def body_without_relations(text: str) -> str:
    match = FRONTMATTER_RE.match(text)
    body = text[match.end() :] if match else text
    return RELATION_RE.sub("", body).strip()


def ensure_summary(path: Path, title: str, summary: str) -> bool:
    text = nfc(path.read_text(encoding="utf-8"))
    if "## 근거" in text or "## PDF 근거" in text:
        return False
    if len(body_without_relations(text)) > 80 and "Stub." not in text:
        return False
    match = FRONTMATTER_RE.match(text)
    prefix = text[: match.end()] if match else ""
    body = text[match.end() :].lstrip("\n") if match else text
    relation_lines = "\n".join(RELATION_RE.findall(body))
    rest = RELATION_RE.sub("", body).strip()
    chunks = []
    if relation_lines:
        chunks.append(relation_lines)
    chunks.append(f"# {title}\n\n## PDF 근거\n- {summary}")
    if rest and "Stub." not in rest:
        chunks.append(rest)
    new_text = prefix + "\n\n" + "\n\n".join(chunks).strip() + "\n"
    if new_text != text:
        path.write_text(nfc(new_text), encoding="utf-8")
        return True
    return False


def course_for(path: Path) -> str | None:
    try:
        parts = path.relative_to(VAULT).parts
    except ValueError:
        return None
    if len(parts) >= 3 and parts[0] == "ComputerScience" and parts[1][:2].isdigit():
        return COURSE_BY_LEAF.get(parts[2])
    return None


def main() -> None:
    pairs: list[tuple[str, str, str, str]] = []
    for old_rel, (new_rel, title, summary) in RENAMES.items():
        src = VAULT / old_rel
        dst = VAULT / new_rel
        if src.exists():
            if dst.exists():
                raise FileExistsError(f"Refusing to overwrite existing path: {new_rel}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            pairs.append((old_rel, new_rel, title, summary))

    mapping: dict[str, tuple[str, str]] = {}
    for old_rel, new_rel, title, _ in pairs:
        mapping[old_rel] = (new_rel.removesuffix(".md"), title)
        mapping[old_rel.removesuffix(".md")] = (new_rel.removesuffix(".md"), title)
        old_stem = Path(old_rel).stem
        mapping[old_stem] = (new_rel.removesuffix(".md"), title)

    link_rewrites = 0
    for path in iter_text_files():
        text = nfc(path.read_text(encoding="utf-8"))
        new_text = rewrite_wikilinks(text, mapping)
        for old_rel, new_rel, _, _ in pairs:
            new_text = new_text.replace(old_rel, new_rel)
            new_text = new_text.replace(old_rel.removesuffix(".md"), new_rel.removesuffix(".md"))
        if new_text != text:
            path.write_text(nfc(new_text), encoding="utf-8")
            link_rewrites += 1

    title_updates = 0
    summary_updates = 0
    for _, new_rel, title, summary in pairs:
        path = VAULT / new_rel
        title_updates += int(set_frontmatter(path, title=title, course=course_for(path)))
        summary_updates += int(ensure_summary(path, title, summary))

    for rel_path, title in TITLE_ONLY.items():
        path = VAULT / rel_path
        if path.exists():
            title_updates += int(set_frontmatter(path, title=title, course=course_for(path)))
    for rel_path, summary in SUMMARY_ONLY.items():
        path = VAULT / rel_path
        if path.exists():
            summary_updates += int(ensure_summary(path, Path(rel_path).stem, summary))

    course_updates = 0
    for path in iter_text_files():
        if path.suffix != ".md":
            continue
        course = course_for(path)
        if course:
            course_updates += int(set_frontmatter(path, course=course))

    print(
        "pdf_backed_identity_refactor "
        f"renamed={len(pairs)} link_files={link_rewrites} "
        f"title_updates={title_updates} summaries={summary_updates} course_updates={course_updates}"
    )


if __name__ == "__main__":
    main()
