#!/usr/bin/env python3
"""Bottom-up content identity cleanup for the edu Obsidian vault.

This pass fixes note/folder identities whose current names clearly conflict
with their content: placeholders, extraction prefixes, and obvious typos. It
then rewrites wikilinks/canvas references so relationship rebuilding can run on
the corrected archive shape.
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

FOLDER_RENAMES = {
    "ComputerScience/1-2_coding-basics/4. 아두이누": "ComputerScience/1-2_coding-basics/4. 아두이노",
}

FILE_RENAMES = {
    "LGAimer/무제.md": ("LGAimer/LG Aimers 9기 평가 및 제출 가이드.md", "LG Aimers 9기 평가 및 제출 가이드"),
    "ComputerScience/misc/misc__발표 스크립트.md": (
        "ComputerScience/misc/GovOn 온프레미스 AI 발표 스크립트.md",
        "GovOn 온프레미스 AI 발표 스크립트",
    ),
    "ComputerScience/3-1_programming-languages/programming-languages__연습문제.md": (
        "ComputerScience/3-1_programming-languages/7장-12장 연습문제 종합.md",
        "7장-12장 연습문제 종합",
    ),
    "ComputerScience/elective_LLM/검색 증강 생성 RAG/LLM__검색 증강 생성 RAG__LangChain.md": (
        "ComputerScience/elective_LLM/검색 증강 생성 RAG/LangChain.md",
        "LangChain",
    ),
    "ComputerScience/3-1_intellectual-property/과제/intellectual-property__과제__과제.md": (
        "ComputerScience/3-1_intellectual-property/과제/사용자 맞춤형 의자 시리즈 특허 명세서.md",
        "사용자 맞춤형 의자 시리즈 특허 명세서",
    ),
    "ComputerScience/3-1_AI-system-design/ot/AI-system-design__ot__발표 스크립트.md": (
        "ComputerScience/3-1_AI-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트.md",
        "AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트",
    ),
    "ComputerScience/3-1_AI-system-design/3주차/AI-system-design__3주차__발표 스크립트.md": (
        "ComputerScience/3-1_AI-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트.md",
        "AI 챗봇 특허 저작권 보호 전략 발표 스크립트",
    ),
    "ComputerScience/2-2_OSS/2. 자바스크립트 객체 다루기/OSS__2. 자바스크립트 객체 다루기__연습문제.md": (
        "ComputerScience/2-2_OSS/2. 자바스크립트 객체 다루기/연습문제.md",
        "자바스크립트 객체 다루기 연습문제",
    ),
    "ComputerScience/2-2_OSS/3. 문서 객체 모델/OSS__3. 문서 객체 모델__연습문제.md": (
        "ComputerScience/2-2_OSS/3. 문서 객체 모델/연습문제.md",
        "문서 객체 모델 연습문제",
    ),
    "ComputerScience/2-2_OSS/0. Html. javascript 기초/OSS__0. Html. javascript 기초__연습문제.md": (
        "ComputerScience/2-2_OSS/0. Html. javascript 기초/연습문제.md",
        "HTML JavaScript 기초 연습문제",
    ),
    "ComputerScience/3-1_ML-project/LangChain/ML-project__LangChain__LangChain.md": (
        "ComputerScience/3-1_ML-project/LangChain/LangChain.md",
        "LangChain",
    ),
    "ComputerScience/3-2_bigdata-analysis/md/bigdata-analysis__md__과제.md": (
        "ComputerScience/3-2_bigdata-analysis/md/K-POP 아티스트 인기도 분석 시스템.md",
        "K-POP 아티스트 인기도 분석 시스템: YouTube 데이터 기반 분기별 트렌드 예측 및 시각화",
    ),
    "ComputerScience/3-2_bigdata-analysis/md/architecture_text.md": (
        "ComputerScience/3-2_bigdata-analysis/md/아키텍처 다이어그램 텍스트 버전.md",
        "아키텍처 다이어그램 텍스트 버전",
    ),
    "ComputerScience/3-2_bigdata-analysis/md/architecture_diagram.md": (
        "ComputerScience/3-2_bigdata-analysis/md/아키텍처 다이어그램.md",
        "아키텍처 다이어그램",
    ),
    "ComputerScience/2-1_probability-statistics/10.Normal_RV/probability-statistics__10.Normal_RV__문제 풀이.md": (
        "ComputerScience/2-1_probability-statistics/10.Normal_RV/문제 풀이.md",
        "Normal Random Variable 문제 풀이",
    ),
    "ComputerScience/2-1_probability-statistics/3.Probability/probability-statistics__3.Probability__문제 풀이.md": (
        "ComputerScience/2-1_probability-statistics/3.Probability/문제 풀이.md",
        "Probability 문제 풀이",
    ),
    "ComputerScience/2-1_probability-statistics/6.Random_Variables/probability-statistics__6.Random_Variables__문제 풀이.md": (
        "ComputerScience/2-1_probability-statistics/6.Random_Variables/문제 풀이.md",
        "Random Variables 문제 풀이",
    ),
    "ComputerScience/2-1_probability-statistics/2.Combinations/probability-statistics__2.Combinations__문제 풀이.md": (
        "ComputerScience/2-1_probability-statistics/2.Combinations/문제 풀이.md",
        "Combinations 문제 풀이",
    ),
    "ComputerScience/2-1_probability-statistics/7-0.Variance/probability-statistics__7-0.Variance__문제 풀이.md": (
        "ComputerScience/2-1_probability-statistics/7-0.Variance/문제 풀이.md",
        "Variance 문제 풀이",
    ),
    "ComputerScience/2-1_probability-statistics/4.Bayes_theorem/probability-statistics__4.Bayes_theorem__문제 풀이.md": (
        "ComputerScience/2-1_probability-statistics/4.Bayes_theorem/문제 풀이.md",
        "Bayes theorem 문제 풀이",
    ),
    "ComputerScience/2-1_probability-statistics/9.Continuous_RVs/probability-statistics__9.Continuous_RVs__문제 풀이.md": (
        "ComputerScience/2-1_probability-statistics/9.Continuous_RVs/문제 풀이.md",
        "Continuous Random Variables 문제 풀이",
    ),
    "ComputerScience/2-1_probability-statistics/11.joint_RVs/probability-statistics__11.joint_RVs__문제 풀이.md": (
        "ComputerScience/2-1_probability-statistics/11.joint_RVs/문제 풀이.md",
        "Joint Random Variables 문제 풀이",
    ),
    "ComputerScience/2-1_probability-statistics/5.Independence/probability-statistics__5.Independence__문제 풀이.md": (
        "ComputerScience/2-1_probability-statistics/5.Independence/문제 풀이.md",
        "Independence 문제 풀이",
    ),
    "ComputerScience/2-1_web-programming/6. HTML 기초2/web-programming__6. HTML 기초2__문제 풀이.md": (
        "ComputerScience/2-1_web-programming/6. HTML 기초2/문제 풀이.md",
        "HTML 기초2 문제 풀이",
    ),
    "ComputerScience/2-1_web-programming/7. 웹 시스템 제작/web-programming__7. 웹 시스템 제작__문제 풀이.md": (
        "ComputerScience/2-1_web-programming/7. 웹 시스템 제작/문제 풀이.md",
        "웹 시스템 제작 문제 풀이",
    ),
    "ComputerScience/2-1_web-programming/3. Spring Boot 기초/web-programming__3. Spring Boot 기초__확인문제.md": (
        "ComputerScience/2-1_web-programming/3. Spring Boot 기초/확인문제.md",
        "Spring Boot 기초 확인문제",
    ),
    "ComputerScience/2-1_web-programming/4. 쿠키와 세션/web-programming__4. 쿠키와 세션__확인문제.md": (
        "ComputerScience/2-1_web-programming/4. 쿠키와 세션/확인문제.md",
        "쿠키와 세션 확인문제",
    ),
    "ComputerScience/2-1_web-programming/5. 데이터베이스/web-programming__5. 데이터베이스__확인문제.md": (
        "ComputerScience/2-1_web-programming/5. 데이터베이스/확인문제.md",
        "데이터베이스 확인문제",
    ),
    "ComputerScience/2-1_web-programming/2. Spring Boot 개발 환경 세팅/web-programming__2. Spring Boot 개발 환경 세팅__확인문제.md": (
        "ComputerScience/2-1_web-programming/2. Spring Boot 개발 환경 세팅/확인문제.md",
        "Spring Boot 개발 환경 세팅 확인문제",
    ),
    "ComputerScience/2-1_web-programming/1. HTML 기초/web-programming__1. HTML 기초__연습문제.md": (
        "ComputerScience/2-1_web-programming/1. HTML 기초/연습문제.md",
        "HTML 기초 연습문제",
    ),
    "ComputerScience/2-2_database/0. 시험/database__0. 시험__연습문제.md": (
        "ComputerScience/2-2_database/0. 시험/데이터베이스 연습문제.md",
        "데이터베이스 연습문제",
    ),
    "ComputerScience/3-1_distributed-computing/REPORT.md": (
        "ComputerScience/3-1_distributed-computing/CUDA 프로그램 연습 및 CUDA API 이해.md",
        "CUDA 프로그램 연습 및 CUDA API 이해",
    ),
    "ComputerScience/3-1_distributed-computing/1. WhyParlleism.md": (
        "ComputerScience/3-1_distributed-computing/1. Why Parallelism.md",
        "1. Why Parallelism",
    ),
    "ComputerScience/3-2_computer-graphics/final exam.md": (
        "ComputerScience/3-2_computer-graphics/컴퓨터그래픽스 기말 정리.md",
        "컴퓨터그래픽스 기말 정리",
    ),
    "ComputerScience/3-2_computer-graphics/middle exam.md": (
        "ComputerScience/3-2_computer-graphics/OpenGL 기본 구조 시험 대비.md",
        "OpenGL 기본 구조 시험 대비",
    ),
    "ComputerScience/3-2_computer-graphics/polygon.md": (
        "ComputerScience/3-2_computer-graphics/OpenGL 폴리곤 그리기 정리.md",
        "OpenGL 폴리곤 그리기 정리",
    ),
    "ComputerScience/2-2_computer-network/0. Quiz/ip rounting.md": (
        "ComputerScience/2-2_computer-network/0. Quiz/Routing Information Protocol (RIP).md",
        "Routing Information Protocol (RIP)",
    ),
    "ComputerScience/elective_docker-k8s/쿠버네티스  기초.md": (
        "ComputerScience/elective_docker-k8s/파드(Pod).md",
        "파드(Pod)",
    ),
    "ComputerScience/3-1_AI-system-design/test.md": (
        "ComputerScience/3-1_AI-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안.md",
        "스마트 오더 플랫폼 B2B 어드민 기능 제안",
    ),
    "ComputerScience/2-1_AI/3. Backpropagation/실습/Overfitting 해결/Data Augumentation.md": (
        "ComputerScience/2-1_AI/3. Backpropagation/실습/Overfitting 해결/Data Augmentation.md",
        "Data Augmentation",
    ),
    "ComputerScience/2-1_AI/2. MLP(Multi Layer Perceptron)/이론/MLP theorm.md": (
        "ComputerScience/2-1_AI/2. MLP(Multi Layer Perceptron)/이론/MLP 이론.md",
        "MLP 이론",
    ),
    "ComputerScience/2-1_probability-statistics/8.Poisson/More Discreete Distributions (시험 X).md": (
        "ComputerScience/2-1_probability-statistics/8.Poisson/More Discrete Distributions (시험 X).md",
        "More Discrete Distributions (시험 X)",
    ),
    "ComputerScience/3-1_ML-project/Pandas/Data_analysis.md": (
        "ComputerScience/3-1_ML-project/Pandas/데이터 분석 및 처리 과정 요약.md",
        "데이터 분석 및 처리 과정 요약",
    ),
    "ComputerScience/3-1_ML-project/Sklearn/Regression/Multiple/Multi_Regression.md": (
        "ComputerScience/3-1_ML-project/Sklearn/Regression/Multiple/다중 선형 회귀.md",
        "다중 선형 회귀",
    ),
    "ComputerScience/3-1_ML-project/Sklearn/Regression/KNNR/KNNR.md": (
        "ComputerScience/3-1_ML-project/Sklearn/Regression/KNNR/KNN 회귀.md",
        "KNN 회귀",
    ),
    "ComputerScience/3-1_ML-project/Sklearn/Classifier/KNNC/KNNC.md": (
        "ComputerScience/3-1_ML-project/Sklearn/Classifier/KNNC/KNN 분류.md",
        "KNN 분류",
    ),
    "ComputerScience/2-1_probability-statistics/22.map/22_MAP.md": (
        "ComputerScience/2-1_probability-statistics/22.map/Maximum A Posteriori.md",
        "Maximum A Posteriori",
    ),
}

TITLE_ONLY = {
    "ComputerScience/elective_coding-test/자료구조/1. 배열과 리스트.md": "배열과 리스트",
    "ComputerScience/elective_coding-test/자료구조/2. 구간 합.md": "구간 합",
    "ComputerScience/elective_coding-test/자료구조/3. 투 포인터.md": "투 포인터",
    "ComputerScience/elective_coding-test/자료구조/4. 슬라이딩 윈도우.md": "슬라이딩 윈도우",
    "ComputerScience/elective_coding-test/자료구조/5. 스택과 큐.md": "스택과 큐",
}

BODY_REPLACEMENTS = {
    "ComputerScience/3-1_distributed-computing/1. Why Parallelism.md": {
        "WhyParlleism": "Why Parallelism",
    },
    "ComputerScience/2-1_AI/3. Backpropagation/실습/Overfitting 해결/Data Augmentation.md": {
        "Data Augumentation": "Data Augmentation",
    },
    "ComputerScience/2-1_probability-statistics/8.Poisson/More Discrete Distributions (시험 X).md": {
        "Discreete": "Discrete",
    },
    "ComputerScience/3-1_distributed-computing/CUDA 프로그램 연습 및 CUDA API 이해.md": {
        "CUDA API이해": "CUDA API 이해",
    },
}

WIKI_RE = re.compile(r"(!?)\[\[([^\]#|]+)(#[^\]|]*)?(\|[^\]]*)?\]\]")
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.S)


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def rel(path: Path) -> str:
    return nfc(path.relative_to(VAULT).as_posix())


def iter_text_files() -> list[Path]:
    results: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(VAULT):
        dirnames[:] = [
            d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")
        ]
        dirpath = nfc(dirpath)
        for filename in filenames:
            filename = nfc(filename)
            if filename.endswith((".md", ".canvas")):
                results.append(Path(dirpath) / filename)
    return sorted(results)


def quote_yaml(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def set_frontmatter_title(path: Path, title: str) -> bool:
    text = nfc(path.read_text(encoding="utf-8"))
    match = FRONTMATTER_RE.match(text)
    if not match:
        return False
    body_start = match.end()
    lines = match.group(1).splitlines()
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith("title:"):
            out.append(f"title: {quote_yaml(title)}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"title: {quote_yaml(title)}")
    new_text = "---\n" + "\n".join(out) + "\n---\n\n" + text[body_start:].lstrip("\n")
    if new_text != text:
        path.write_text(nfc(new_text), encoding="utf-8")
        return True
    return False


def rewrite_text_links(text: str, target_map: dict[str, tuple[str, str]]) -> str:
    def repl(match: re.Match[str]) -> str:
        bang, target, section, alias = match.groups()
        target = nfc(target.strip())
        mapped = target_map.get(target)
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


def build_target_map(pairs: list[tuple[str, str, str]]) -> dict[str, tuple[str, str]]:
    basename_counts: dict[str, int] = {}
    for old_rel, _, _ in pairs:
        old_stem = Path(old_rel).stem
        basename_counts[old_stem] = basename_counts.get(old_stem, 0) + 1

    mapping: dict[str, tuple[str, str]] = {}
    for old_rel, new_rel, new_title in pairs:
        old_no_ext = old_rel.removesuffix(".md")
        new_no_ext = new_rel.removesuffix(".md")
        mapping[old_rel] = (new_no_ext, new_title)
        mapping[old_no_ext] = (new_no_ext, new_title)
        old_stem = Path(old_rel).stem
        if basename_counts.get(old_stem, 0) == 1:
            mapping[old_stem] = (new_no_ext, new_title)
            mapping[old_stem + ".md"] = (new_no_ext, new_title)
    return mapping


def checked_rename(old_rel: str, new_rel: str) -> bool:
    src = VAULT / old_rel
    dst = VAULT / new_rel
    if not src.exists():
        return False
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {new_rel}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    return True


def apply_body_replacements(path: Path) -> bool:
    replacements = BODY_REPLACEMENTS.get(rel(path))
    if not replacements:
        return False
    text = nfc(path.read_text(encoding="utf-8"))
    new_text = text
    for old, new in replacements.items():
        new_text = new_text.replace(old, new)
    if new_text != text:
        path.write_text(nfc(new_text), encoding="utf-8")
        return True
    return False


def main() -> None:
    folder_renamed = 0
    file_pairs: list[tuple[str, str, str]] = []

    for old_rel, new_rel in FOLDER_RENAMES.items():
        if checked_rename(old_rel, new_rel):
            folder_renamed += 1
            for path in (VAULT / new_rel).rglob("*.md"):
                old_file_rel = (Path(old_rel) / path.relative_to(VAULT / new_rel)).as_posix()
                new_file_rel = rel(path)
                file_pairs.append((old_file_rel, new_file_rel, path.stem))

    for old_rel, (new_rel, title) in FILE_RENAMES.items():
        if checked_rename(old_rel, new_rel):
            file_pairs.append((old_rel, new_rel, title))

    target_map = build_target_map(file_pairs)
    link_rewrites = 0
    for path in iter_text_files():
        text = nfc(path.read_text(encoding="utf-8"))
        new_text = rewrite_text_links(text, target_map)
        if new_text != text:
            path.write_text(nfc(new_text), encoding="utf-8")
            link_rewrites += 1

    title_updates = 0
    body_updates = 0
    for _, new_rel, title in file_pairs:
        path = VAULT / new_rel
        if path.suffix == ".md" and path.exists():
            title_updates += int(set_frontmatter_title(path, title))
            body_updates += int(apply_body_replacements(path))

    for rel_path, title in TITLE_ONLY.items():
        path = VAULT / rel_path
        if path.exists():
            title_updates += int(set_frontmatter_title(path, title))

    print(
        "bottom_up_refactor "
        f"folders={folder_renamed} files={len([p for p in file_pairs if p[0].endswith('.md')])} "
        f"link_files={link_rewrites} title_updates={title_updates} body_updates={body_updates}"
    )


if __name__ == "__main__":
    main()
