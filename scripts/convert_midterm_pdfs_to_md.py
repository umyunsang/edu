#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CourseConfig:
    name: str
    folder: Path
    out_dir: Path
    sources: list[Path]
    skipped: list[Path]


ALGORITHM_DIR = ROOT / "ComputerScience" / "4-1_algorithm"
CV_DIR = ROOT / "ComputerScience" / "4-1_computer-vision"


COURSES: list[CourseConfig] = [
    CourseConfig(
        name="알고리즘",
        folder=ALGORITHM_DIR,
        out_dir=ALGORITHM_DIR / "markdown_midterm",
        sources=[
            ALGORITHM_DIR / "01장.알고리즘개요-파알.pdf",
            ALGORITHM_DIR / "02장.알고리즘효율성분석-파알.pdf",
            ALGORITHM_DIR / "03장-억지기법과완전탐색-파알.pdf",
            ALGORITHM_DIR / "04장-축소정복기법-수정판.pdf",
            ALGORITHM_DIR / "05장-분할정복기법-파알.pdf",
            ALGORITHM_DIR / "06장-공간으로시간벌기-수정판.pdf",
            ALGORITHM_DIR / "Pop Quiz 풀이" / "PopQuiz-25-1-풀이.pdf",
            ALGORITHM_DIR / "Pop Quiz 풀이" / "Alg-PopQuiz#2-풀이-2.pdf",
            ALGORITHM_DIR / "PopQuiz" / "Pop Quiz 1 풀이.pdf",
        ],
        skipped=[
            ALGORITHM_DIR / "07-동적계획법-보충-수정판.pdf",
            ALGORITHM_DIR / "07장-동적계획법-수정판.pdf",
            ALGORITHM_DIR / "08장-탐욕적기법-수정판.pdf",
            ALGORITHM_DIR / "09장-백트래킹과분기한정-수정판-v2.pdf",
            ALGORITHM_DIR / "10장-NP완전과근사알고리즘-파알.pdf",
        ],
    ),
    CourseConfig(
        name="컴퓨터비전",
        folder=CV_DIR,
        out_dir=CV_DIR / "markdown_midterm",
        sources=[
            CV_DIR / "Computer_Vision_1_overview_v1.1.pdf",
            CV_DIR / "Computer_Vision_2_2D_Image_Processing_v1.1.pdf",
            CV_DIR / "Computer_Vision_3_2D_Image_Processing_2_v1.1.pdf",
            CV_DIR / "Computer_Vision_4_Feature_Extraction_and_Matching_v1.5.pdf",
            CV_DIR / "Computer_Vision_5_Stereo_Vision_v1.0.pdf",
        ],
        skipped=[
            CV_DIR / "Computer_Vision_2_2D_Image_Processing_v1.0.pdf",
            CV_DIR / "Computer_Vision_3_2D_Image_Processing_2_v1.0.pdf",
            CV_DIR / "Computer_Vision_4_Feature_Extraction_and_Matching_v1.1.pdf",
            CV_DIR / "Computer_Vision_4_Feature_Extraction_and_Matching_v1.2.pdf",
            CV_DIR / "Computer_Vision_4_Feature_Extraction_and_Matching_v1.3.pdf",
            CV_DIR / "Computer_Vision_4_Feature_Extraction_and_Matching_v1.4.pdf",
        ],
    ),
]


PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")
MULTI_BLANK_RE = re.compile(r"\n{3,}")


def run_text_command(*args: str) -> str:
    completed = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout


def pdf_page_count(pdf_path: Path) -> int:
    output = run_text_command("pdfinfo", str(pdf_path))
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError(f"Could not parse page count for {pdf_path}")


def normalize_line(line: str) -> str:
    line = line.rstrip()
    line = line.replace("\u2022", "- ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def clean_page(page_text: str) -> str:
    kept_lines: list[str] = []
    for raw_line in page_text.splitlines():
        line = normalize_line(raw_line)
        if PAGE_NUMBER_RE.match(line):
            continue
        if not line:
            kept_lines.append("")
            continue
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines).strip()
    cleaned = MULTI_BLANK_RE.sub("\n\n", cleaned)
    return cleaned or "(텍스트 없음)"


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    raw_text = run_text_command("pdftotext", "-layout", str(pdf_path), "-")
    pages = [clean_page(page) for page in raw_text.split("\f")]
    return [page for page in pages if page]


def pdf_to_markdown(pdf_path: Path) -> str:
    relative_source = pdf_path.relative_to(ROOT)
    page_count = pdf_page_count(pdf_path)
    pages = extract_pdf_pages(pdf_path)
    lines = [
        f"# {pdf_path.stem}",
        "",
        f"- Source PDF: `{relative_source}`",
        f"- Total pages: {page_count}",
        "",
    ]
    for index, page in enumerate(pages, start=1):
        lines.append(f"## Page {index}")
        lines.append("")
        lines.append(page)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_index(course: CourseConfig) -> str:
    lines = [
        f"# {course.name} 중간고사 PDF -> Markdown 변환 목록",
        "",
        f"- Output folder: `{course.out_dir.relative_to(ROOT)}`",
        "",
        "## Converted PDFs",
        "",
    ]
    for pdf_path in course.sources:
        lines.append(f"- `{pdf_path.name}`")
    if course.skipped:
        lines.extend(
            [
                "",
                "## Skipped PDFs",
                "",
            ]
        )
        for pdf_path in course.skipped:
            lines.append(f"- `{pdf_path.name}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    for course in COURSES:
        course.out_dir.mkdir(parents=True, exist_ok=True)
        for pdf_path in course.sources:
            markdown = pdf_to_markdown(pdf_path)
            out_path = course.out_dir / f"{pdf_path.stem}.md"
            out_path.write_text(markdown, encoding="utf-8")
        index_path = course.out_dir / "INDEX.md"
        index_path.write_text(build_index(course), encoding="utf-8")


if __name__ == "__main__":
    main()
