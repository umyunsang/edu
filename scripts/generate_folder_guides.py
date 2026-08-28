#!/usr/bin/env python3
"""
generate_folder_guides.py — 폴더별 `.ok/frontmatter.yml` 에이전트 가이드를 생성한다.

OpenKnowledge는 에이전트가 `ls` / `cat` / `search` 를 호출할 때마다 해당 폴더의
`.ok/frontmatter.yml` 을 함께 보여준다. 즉 이 파일이 **LLM에게 폴더의 의미를 알려주는
1차 인터페이스**다. 사람은 Obsidian 폴더 트리를 보고, LLM은 이 파일을 본다.

멱등적으로 동작한다. 기존 파일은 내용이 같으면 건드리지 않는다.

사용법:
  python3 scripts/generate_folder_guides.py            # 전체 생성
  python3 scripts/generate_folder_guides.py --dry-run  # 계획만 출력
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FIELDS: dict[str, tuple[str, str, list[str]]] = {
    "00_graph-interfaces": (
        "그래프 인터페이스",
        "Obsidian Graph View에 실제 노드로 보이는 구조 레이어. 단계(stages), 브리지(bridges), "
        "과목 인터페이스(courses), 2026 GraphRAG 아카이브(archive-kg)의 커뮤니티·개념·근거 노드가 있다. "
        "여기 있는 노트는 강의 내용이 아니라 **다른 노트를 연결하는 구조물**이다. "
        "특정 주제의 실제 내용을 찾을 때가 아니라, 아카이브의 지도를 볼 때 읽는다.",
        ["graph", "structure", "moc"],
    ),
    "01_programming-foundations": (
        "프로그래밍 기초",
        "프로그래밍 언어와 구현 기초. 코딩기초(아두이노·컴퓨팅 사고), Python, 자료구조, "
        "코딩테스트, Java. 문법과 자료구조 구현이 주된 내용이다.",
        ["programming", "foundations"],
    ),
    "02_math-theory": (
        "수학과 이론",
        "AI·CS의 수학적 기초. 확률통계, 이산수학, 최적화수학, 수리논리학. "
        "증명과 수식이 중심이라 LaTeX 표기가 많다.",
        ["math", "theory"],
    ),
    "03_ai-ml-data": (
        "AI · ML · 데이터",
        "인공지능, 머신러닝, 뉴럴네트워크, 컴퓨터비전, LLM, 빅데이터분석, AI시스템설계, "
        "생성형 AI 파인튜닝, 양자 ML. 이 아카이브에서 가장 두꺼운 영역이고, "
        "노트와 실습 코드(ipynb)가 섞여 있다.",
        ["ai", "ml", "data"],
    ),
    "04_systems-infrastructure": (
        "시스템과 인프라",
        "리눅스, 컴퓨터구조, 운영체제, 컴퓨터네트워크, 병렬·분산처리(CUDA·MPI), "
        "컨테이너 오케스트레이션. 하드웨어에서 클러스터까지의 계층을 다룬다.",
        ["systems", "infrastructure"],
    ),
    "05_software-engineering": (
        "소프트웨어 엔지니어링",
        "웹 프로그래밍, 데이터베이스, 오픈소스 소프트웨어, 프로그래밍언어론, "
        "AIOSS 오픈소스 delivery. 실제 개발·협업·운영 워크플로가 중심이다.",
        ["software-engineering"],
    ),
    "06_algorithms-graphics": (
        "알고리즘과 그래픽스",
        "알고리즘 설계와 분석(분할정복·탐욕법·DP·NP), 컴퓨터 그래픽스. "
        "복잡도 분석과 증명이 많다.",
        ["algorithms", "graphics"],
    ),
    "07_professional-humanities": (
        "전문 교양",
        "지식재산권, 창의적 글쓰기, 고전 읽기, 인간환경과이해, 학점 포트폴리오. "
        "전공 외 교양 및 졸업 요건 관련 자료.",
        ["humanities", "general-education"],
    ),
}

# course-folder -> (표시 제목, 설명, 학기)
COURSES: dict[str, tuple[str, str, str]] = {
    # 01 programming-foundations
    "coding-basics": ("코딩 기초와 문제해결", "컴퓨팅 사고, 정보의 표현, 아두이노 실습.", "1-2"),
    "python-programming": ("Python 프로그래밍", "파이썬 문법과 객체지향. 지뢰찾기 등 구현 과제 포함.", "2-1"),
    "data-structures": ("자료구조", "리스트·스택·큐·트리·정렬. 주차별 구현 과제가 함께 있다.", "2-1"),
    "coding-test": ("코딩 테스트", "알고리즘 문제 풀이 정리. 강의자료 없이 문제 풀이 노트만 있다.", "extracurricular"),
    "java-programming": ("Java 프로그래밍", "자바 문법과 객체지향 정리.", "extracurricular"),
    # 02 math-theory
    "probability-statistics": ("확률과 통계", "확률·통계 기초와 데이터 해석. 이 저장소에서 노트가 가장 많은 수학 과목.", "2-1"),
    "discrete-mathematics": ("이산수학", "집합·논리·관계·그래프 이론. 알고리즘과 수리논리학의 선수 과목.", "2-2"),
    "optimization-math": ("최적화 수학", "경사하강·볼록 최적화 등 학습 알고리즘의 수학적 배경.", "3-2"),
    "mathematical-logic": ("수리논리학", "명제·술어 논리와 증명. 신호 처리 실습(STFT·MFCC) 자료가 섞여 있다.", "3-1"),
    # 03 ai-ml-data
    "artificial-intelligence": ("인공지능", "퍼셉트론·역전파·CNN. CIFAR10 CNN 실습이 포함된다.", "2-1"),
    "machine-learning": ("머신러닝", "회귀, SVM, RNN, Transformer.", "3-1"),
    "ml-projects": ("머신러닝 프로젝트", "SKLearn·Pandas·LangChain 기반 프로젝트 과제.", "3-1"),
    "neural-networks": ("뉴럴네트워크", "심화 신경망 아키텍처.", "3-2"),
    "computer-vision": ("컴퓨터비전", "영상처리, 기하변환, 특징점(SIFT·ORB), 스테레오.", "4-1"),
    "large-language-models": ("LLM 이해와 활용", "대규모 언어모델의 구조와 활용. RAG·프롬프팅 포함.", "extracurricular"),
    "big-data-analysis": ("빅데이터 분석", "데이터 레이크, 분석 도구, MLFlow 과제.", "3-2"),
    "ai-system-design": ("AI 시스템 설계", "MLOps와 아키텍처 설계. 카페 주문 시스템 프로젝트.", "3-1"),
    "generative-ai-fine-tuning": ("생성형 AI 파인튜닝", "생성형 모델 파인튜닝 실습 자료.", "extracurricular"),
    "quantum-lecture": ("양자컴퓨팅 특강", "양자 게이트·Braket·Grover/Shor·VQA·QML 2일 특강. pptx와 ipynb 실습이 함께 있다.", "extracurricular"),
    "quantum-ml": ("양자 머신러닝", "Quantum Reframing Challenge 등 양자 ML 연구·대회 자료.", "extracurricular"),
    # 04 systems-infrastructure
    "linux": ("리눅스 시스템", "셸·편집기·권한·프로세스·도커·REST 등 서버 기초.", "2-1"),
    "computer-architecture": ("컴퓨터 구조", "논리 게이트부터 CPU·기억장치·파이프라이닝까지.", "2-1"),
    "operating-systems": ("운영체제", "프로세스·스레드·스케줄링·동기화·메모리·파일시스템. 스케줄러 구현 과제 포함.", "2-2"),
    "computer-networks": ("컴퓨터 네트워크", "계층 모델, 신호 처리, LAN, IP·라우팅, 전송 계층, 보안.", "2-2"),
    "parallel-distributed-computing": (
        "병렬 · 분산처리",
        "Stanford CS149 기반. 병렬성의 동기, 멀티코어 아키텍처, ISPC, 작업 분배와 스케줄링, "
        "지역성·통신·경합, GPU/CUDA, MPI.",
        "3-1",
    ),
    "container-orchestration": ("컨테이너 오케스트레이션", "도커와 쿠버네티스 기초. 강의자료 없이 정리 노트만 있다.", "extracurricular"),
    # 05 software-engineering
    "web-programming": ("웹 프로그래밍", "HTML/CSS/JS와 Spring Boot 기초.", "2-1"),
    "database-systems": ("데이터베이스", "SQL, 정규화, 데이터 모델링.", "2-2"),
    "open-source-software": ("오픈소스 소프트웨어", "JS 이벤트·객체·DOM 등 클라이언트 기초.", "2-2"),
    "programming-languages": ("프로그래밍 언어론", "언어 설계 원리, 타입 체계, 실행 모델.", "3-1"),
    "aioss-open-source-delivery": (
        "AIOSS 오픈소스 delivery",
        "메트릭·계획·협업·비동기 작업·GitHub Actions 등 오픈소스 개발 운영. "
        "주차별 강의(md/Week0~6)와 실증 프로젝트 기여 기록.",
        "4-1",
    ),
    # 06 algorithms-graphics
    "algorithm-design-analysis": (
        "알고리즘 설계와 분석",
        "복잡도 분석, 억지기법·완전탐색, 축소정복, 분할정복, 공간으로 시간 벌기, "
        "동적계획법, 탐욕적 기법, 백트래킹·분기한정, NP완전과 근사 알고리즘. Pop Quiz 풀이 포함.",
        "4-1",
    ),
    "computer-graphics": ("컴퓨터 그래픽스", "렌더링과 그래픽스 기초.", "3-2"),
    # 07 professional-humanities
    "intellectual-property": ("지식재산권", "특허·저작권·상표 등 지식재산 제도.", "3-1"),
    "creative-writing": ("창의적 글쓰기", "MECE·로직트리·문단 구성 등 테크니컬 라이팅.", "4-1"),
    "classics-reading": ("고전 읽기", "고전 텍스트 독해와 토론.", "extracurricular"),
    "degree-portfolio": ("학점 포트폴리오", "졸업 요건과 이수 계획 정리.", "extracurricular"),
    "인간환경과이해": ("인간환경과이해", "환경과 인간의 상호작용을 다루는 교양.", "extracurricular"),
}

PDF_GUIDE = (
    "이 폴더의 PDF는 **교수가 배포한 원본 강의자료**다 (Git LFS로 관리). "
    "직접 읽지 말고 `scripts/pdf_lecture_extract.py` 로 추출 번들을 만든 뒤 그것을 근거로 삼는다. "
    "정리문서는 이 폴더가 아니라 **상위 수업 폴더 루트**에 만든다. "
    "파일명이 ` 2.pdf` 로 끝나는 것은 macOS 중복 사본이므로 무시한다."
)


def yaml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def render(title: str, description: str, tags: list[str]) -> str:
    lines = [
        "# 이 파일은 OpenKnowledge가 에이전트에게 폴더의 의미를 알려주는 가이드다.",
        "# 사람은 Obsidian 폴더 트리를 보고, LLM은 이 설명을 본다.",
        "# scripts/generate_folder_guides.py 로 생성됨 — 손으로 고쳐도 되지만 스크립트도 같이 갱신할 것.",
        f'title: "{yaml_escape(title)}"',
        f'description: "{yaml_escape(description)}"',
        "tags:",
    ]
    lines += [f"  - {t}" for t in tags]
    return "\n".join(lines) + "\n"


def write(path: Path, content: str, dry: bool, changed: list[str]) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    changed.append(str(path.relative_to(ROOT)))
    if not dry:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    changed: list[str] = []

    cs = ROOT / "ComputerScience"

    write(
        cs / ".ok" / "frontmatter.yml",
        render(
            "Computer Science & AI 커리큘럼",
            "학부 전 과정의 강의 아카이브. 8개 분야 인터페이스 폴더 아래에 수업 폴더가 있고, "
            "각 수업 폴더는 원본 강의자료(`pdf/`)와 그것을 근거로 쓴 정리문서(폴더 루트 `.md`)로 구성된다. "
            "학년/학기는 폴더가 아니라 각 노트의 `semester` 프론트매터에 기록된다. "
            "노트 작성 규격은 `docs/knowledge-schema.md` 를 따른다.",
            ["curriculum", "archive", "obsidian"],
        ),
        args.dry_run,
        changed,
    )

    for field_dir, (title, desc, tags) in FIELDS.items():
        p = cs / field_dir
        if not p.is_dir():
            continue
        write(p / ".ok" / "frontmatter.yml", render(title, desc, tags), args.dry_run, changed)

    for field_dir in FIELDS:
        fp = cs / field_dir
        if not fp.is_dir():
            continue
        for course in sorted(x for x in fp.iterdir() if x.is_dir() and not x.name.startswith(".")):
            meta = COURSES.get(course.name)
            if meta is None:
                continue
            ctitle, cdesc, sem = meta
            full = (
                f"{cdesc} 학기: {sem}. "
                "원본 강의자료는 `sources/` 에, 슬라이드 렌더 이미지는 `assets/` 에 있고, "
                "그것을 근거로 쓴 정리문서가 `notes/` 에 있다."
            )
            write(
                course / ".ok" / "frontmatter.yml",
                render(ctitle, full, ["course", f"semester/{sem}"]),
                args.dry_run,
                changed,
            )
            pdf_dir = course / "pdf"
            if pdf_dir.is_dir():
                write(
                    pdf_dir / ".ok" / "frontmatter.yml",
                    render(f"{ctitle} — 원본 강의자료", PDF_GUIDE, ["source", "pdf", "immutable"]),
                    args.dry_run,
                    changed,
                )

    verb = "생성 예정" if args.dry_run else "생성/갱신"
    print(f"{verb}: {len(changed)}개")
    for c in changed:
        print(f"  {c}")


if __name__ == "__main__":
    main()
