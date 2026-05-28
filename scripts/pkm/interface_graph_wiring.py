#!/usr/bin/env python3
"""Create real graph nodes for field interface folders and wire course anchors.

Obsidian Graph View renders files and links, not folders. The field interface
folders therefore need explicit interface notes, and course anchor notes need to
point to those notes so the visual graph clusters by domain.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

VAULT = Path("/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu")
FIELD_RE = re.compile(r"^(?:up|siblings|related|prerequisites|next|central)::\s*.*$", re.MULTILINE)
FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.S)


@dataclass(frozen=True)
class CourseAnchor:
    label: str
    rel: str


@dataclass(frozen=True)
class Domain:
    key: str
    title: str
    path: str
    summary: str
    courses: tuple[CourseAnchor, ...]
    prerequisites: tuple[str, ...] = ()
    related: tuple[str, ...] = ()


DOMAINS: tuple[Domain, ...] = (
    Domain(
        key="programming",
        title="프로그래밍 기초 인터페이스",
        path="ComputerScience/01_programming-foundations/프로그래밍 기초 인터페이스.md",
        summary="프로그래밍 언어 문법, 자료구조, 코딩 테스트 구현력을 묶는 진입 인터페이스입니다.",
        courses=(
            CourseAnchor("코딩 기초", "ComputerScience/01_programming-foundations/coding-basics/중간고사.md"),
            CourseAnchor("Python 프로그래밍", "ComputerScience/01_programming-foundations/python-programming/1. 변수와 자료형.md"),
            CourseAnchor("자료구조", "ComputerScience/01_programming-foundations/data-structures/5. 정렬/정렬.md"),
            CourseAnchor("코딩 테스트", "ComputerScience/01_programming-foundations/coding-test/자료구조/1. 배열과 리스트.md"),
            CourseAnchor("Java 프로그래밍", "ComputerScience/01_programming-foundations/java-programming/1. Hello Java.md"),
        ),
        related=("math", "software"),
    ),
    Domain(
        key="math",
        title="수학 이론 인터페이스",
        path="ComputerScience/02_math-theory/수학 이론 인터페이스.md",
        summary="확률통계, 이산수학, 최적화, 논리학을 AI와 알고리즘의 이론 기반으로 연결합니다.",
        courses=(
            CourseAnchor("확률통계", "ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability.md"),
            CourseAnchor("이산수학", "ComputerScience/02_math-theory/discrete-mathematics/4. 그래프/그래프.md"),
            CourseAnchor("최적화 수학", "ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix.md"),
            CourseAnchor("수리논리학", "ComputerScience/02_math-theory/mathematical-logic/논리학 개론.md"),
        ),
        related=("programming", "ai", "algorithms"),
    ),
    Domain(
        key="ai",
        title="AI ML 데이터 인터페이스",
        path="ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스.md",
        summary="인공지능, 머신러닝, 신경망, 컴퓨터비전, LLM, 빅데이터, 생성형 AI 프로젝트를 연결합니다.",
        courses=(
            CourseAnchor("인공지능", "ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation.md"),
            CourseAnchor("머신러닝", "ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념.md"),
            CourseAnchor("ML 프로젝트", "ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약.md"),
            CourseAnchor("뉴럴네트워크", "ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리.md"),
            CourseAnchor("빅데이터분석", "ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템.md"),
            CourseAnchor("컴퓨터비전", "ComputerScience/03_ai-ml-data/computer-vision/중간고사_컴퓨터비전_정밀분석_정리.md"),
            CourseAnchor("LLM", "ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG.md"),
            CourseAnchor("AI 시스템 설계", "ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발.md"),
            CourseAnchor("생성형 AI 파인튜닝", "ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 파인튜닝 프로젝트 주제.md"),
        ),
        prerequisites=("programming", "math"),
        related=("systems", "software"),
    ),
    Domain(
        key="systems",
        title="시스템 인프라 인터페이스",
        path="ComputerScience/04_systems-infrastructure/시스템 인프라 인터페이스.md",
        summary="Linux, 컴퓨터구조, 운영체제, 네트워크, CUDA/MPI 분산처리, 컨테이너 운영을 묶습니다.",
        courses=(
            CourseAnchor("Linux", "ComputerScience/04_systems-infrastructure/linux/1. 리눅스의 기본.md"),
            CourseAnchor("컴퓨터구조", "ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/과제_CacheFriendly코딩실습.md"),
            CourseAnchor("운영체제", "ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/프로세스와 프로세스 관리.md"),
            CourseAnchor("컴퓨터네트워크", "ComputerScience/04_systems-infrastructure/computer-networks/14. TCP와 소켓 프로그래밍/TCP와 소켓 프로그래밍.md"),
            CourseAnchor("병렬 분산처리", "ComputerScience/04_systems-infrastructure/parallel-distributed-computing/쿠다.md"),
            CourseAnchor("컨테이너 오케스트레이션", "ComputerScience/04_systems-infrastructure/container-orchestration/도커 기초.md"),
        ),
        prerequisites=("programming",),
        related=("software", "ai"),
    ),
    Domain(
        key="software",
        title="소프트웨어 엔지니어링 인터페이스",
        path="ComputerScience/05_software-engineering/소프트웨어 엔지니어링 인터페이스.md",
        summary="웹, 데이터베이스, OSS, 프로그래밍언어론, AIOSS 오픈소스 delivery workflow를 연결합니다.",
        courses=(
            CourseAnchor("웹 프로그래밍", "ComputerScience/05_software-engineering/web-programming/3. Spring Boot 기초/Spring Boot 기초 실습.md"),
            CourseAnchor("데이터베이스", "ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/데이터 베이스 언어 SQL.md"),
            CourseAnchor("오픈소스 소프트웨어", "ComputerScience/05_software-engineering/open-source-software/3. 문서 객체 모델/문서 객체 모델(DOM).md"),
            CourseAnchor("프로그래밍언어론", "ComputerScience/05_software-engineering/programming-languages/필기/3. 구문론.md"),
            CourseAnchor("AIOSS 오픈소스 delivery", "ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week0 - Orientation.md"),
        ),
        prerequisites=("programming", "systems"),
        related=("ai", "professional"),
    ),
    Domain(
        key="algorithms",
        title="알고리즘 그래픽스 인터페이스",
        path="ComputerScience/06_algorithms-graphics/알고리즘 그래픽스 인터페이스.md",
        summary="알고리즘 설계/분석과 컴퓨터그래픽스를 문제 해결 및 시각화 역량으로 연결합니다.",
        courses=(
            CourseAnchor("알고리즘 설계와 분석", "ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리.md"),
            CourseAnchor("컴퓨터그래픽스", "ComputerScience/06_algorithms-graphics/computer-graphics/컴퓨터그래픽스-시험대비.md"),
        ),
        prerequisites=("programming", "math"),
        related=("ai",),
    ),
    Domain(
        key="professional",
        title="전문 교양 인터페이스",
        path="ComputerScience/07_professional-humanities/전문 교양 인터페이스.md",
        summary="지식재산, 창의적 글쓰기, 고전 읽기, 졸업/포트폴리오 자료를 학습 산출물 맥락으로 연결합니다.",
        courses=(
            CourseAnchor("지식재산", "ComputerScience/07_professional-humanities/intellectual-property/2. 저작권제도와 등록요건/저작권 제도와 등록요건.md"),
            CourseAnchor("창의적 글쓰기", "ComputerScience/07_professional-humanities/creative-writing/중간고사_창의적글쓰기_정리.md"),
            CourseAnchor("고전 읽기", "ComputerScience/07_professional-humanities/classics-reading/멋진신세계.md"),
            CourseAnchor("학점 포트폴리오", "ComputerScience/07_professional-humanities/degree-portfolio/졸업학점.md"),
        ),
        related=("software", "ai"),
    ),
)


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def link(rel: str, alias: str | None = None) -> str:
    target = rel.removesuffix(".md")
    if alias:
        return f"[[{target}|{alias}]]"
    return f"[[{target}]]"


def write_note(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = nfc(text.rstrip() + "\n")
    if path.exists() and nfc(path.read_text(encoding="utf-8")) == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def replace_field_line(body: str, field: str, value: str) -> str:
    pattern = re.compile(rf"^{field}::\s*.*$", re.MULTILINE)
    if pattern.search(body):
        return pattern.sub(f"{field}:: {value}", body, count=1)
    return f"{field}:: {value}\n" + body.lstrip("\n")


def split_frontmatter(text: str) -> tuple[str, str]:
    match = FRONTMATTER_RE.match(text)
    if match:
        return text[: match.end()].rstrip() + "\n\n", text[match.end() :].lstrip("\n")
    return "", text


def ensure_anchor_up(anchor: CourseAnchor, domain: Domain) -> bool:
    path = VAULT / anchor.rel
    if not path.exists():
        raise FileNotFoundError(anchor.rel)
    text = nfc(path.read_text(encoding="utf-8"))
    fm, body = split_frontmatter(text)
    body = replace_field_line(body, "up", link(domain.path, domain.title))
    out = fm + body.lstrip("\n")
    if not out.endswith("\n"):
        out += "\n"
    if out != text:
        path.write_text(nfc(out), encoding="utf-8")
        return True
    return False


def domain_note(domain: Domain, by_key: dict[str, Domain]) -> str:
    prereq_links = [link(by_key[k].path, by_key[k].title) for k in domain.prerequisites]
    related_links = [link(by_key[k].path, by_key[k].title) for k in domain.related]
    course_links = [link(course.rel, course.label) for course in domain.courses]
    all_related = related_links + course_links

    lines = [
        "---",
        "aliases: []",
        f"course: {domain.key}",
        "created: '2026-05-28'",
        "date: '2026-05-28'",
        "semester: meta",
        "source: ''",
        "status: evergreen",
        "tags:",
        "- type/interface",
        "- pkm/domain",
        f"title: {domain.title}",
        "type: interface",
        "updated: '2026-05-28'",
        "---",
        "",
    ]
    if prereq_links:
        lines.append("prerequisites:: " + ", ".join(prereq_links))
    if all_related:
        lines.append("related:: " + ", ".join(all_related))
    lines.extend(
        [
            "",
            f"# {domain.title}",
            "",
            domain.summary,
            "",
            "## 과목 인터페이스",
            "",
        ]
    )
    for course in domain.courses:
        lines.append(f"- {link(course.rel, course.label)}")
    if prereq_links:
        lines.extend(["", "## 선수 분야", ""])
        for key in domain.prerequisites:
            target = by_key[key]
            lines.append(f"- {link(target.path, target.title)}")
    if related_links:
        lines.extend(["", "## 연결 분야", ""])
        for key in domain.related:
            target = by_key[key]
            lines.append(f"- {link(target.path, target.title)}")
    return "\n".join(lines)


def main() -> None:
    by_key = {domain.key: domain for domain in DOMAINS}
    created_or_updated = 0
    anchor_updates = 0
    for domain in DOMAINS:
        if write_note(VAULT / domain.path, domain_note(domain, by_key)):
            created_or_updated += 1
        for anchor in domain.courses:
            anchor_updates += int(ensure_anchor_up(anchor, domain))

    print(
        "interface_graph_wiring "
        f"interface_notes={len(DOMAINS)} updated_interfaces={created_or_updated} "
        f"course_anchor_updates={anchor_updates}"
    )


if __name__ == "__main__":
    main()
