domain:: [[ComputerScience/07_professional-humanities/전문 교양 인터페이스|전문 교양 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/6단계 전문 확장 인터페이스|6단계 전문 확장 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/창의적 글쓰기 인터페이스|창의적 글쓰기 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/산출물 포트폴리오 브리지|산출물 포트폴리오 브리지]]
up:: [[ComputerScience/07_professional-humanities/creative-writing/중간고사_창의적글쓰기_정리|중간고사_창의적글쓰기_정리]]
related:: [[ComputerScience/07_professional-humanities/creative-writing/AGENTS|AGENTS]], [[ComputerScience/07_professional-humanities/creative-writing/docs/CODEX_RAG_NOTES|CODEX_RAG_NOTES]], [[ComputerScience/07_professional-humanities/creative-writing/templates/writing-assignment|writing-assignment]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/self_intro_assignment2_starl_skeleton|self_intro_assignment2_starl_skeleton]], [[ComputerScience/07_professional-humanities/creative-writing/knowledge/source_map|source_map]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/self_intro_assignment2_full_draft|self_intro_assignment2_full_draft]], [[ComputerScience/07_professional-humanities/creative-writing/pdf/퀴즈|퀴즈]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/deco_extracurricular_evidence|deco_extracurricular_evidence]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/self_intro_complexity_hardest_experience|self_intro_complexity_hardest_experience]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/self_intro_ax_archive_draft|self_intro_ax_archive_draft]], [[ComputerScience/07_professional-humanities/degree-portfolio/PDF_인쇄_완전가이드|PDF_인쇄_완전가이드]], [[ComputerScience/07_professional-humanities/degree-portfolio/GovOn 온프레미스 AI 발표 스크립트|GovOn 온프레미스 AI 발표 스크립트]], [[ComputerScience/07_professional-humanities/degree-portfolio/졸업학점|졸업학점]], [[ComputerScience/07_professional-humanities/classics-reading/멋진신세계|멋진신세계]], [[ComputerScience/07_professional-humanities/intellectual-property/3. 상표제도 등록요건/상표제도 및 등록요건|상표제도 및 등록요건]], [[ComputerScience/07_professional-humanities/intellectual-property/6. 특허 명세서/특허 명세서 작성법|특허 명세서 작성법]], [[ComputerScience/07_professional-humanities/intellectual-property/5. 특허/특허 제도|특허 제도]], [[ComputerScience/07_professional-humanities/intellectual-property/4. 디자인 제도 및 등록요건/디자인 제도의 목적과 개념|디자인 제도의 목적과 개념]], [[ComputerScience/07_professional-humanities/intellectual-property/기출문제/processed/특허_processed|특허_processed]]

kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/창의적 글쓰기 지식그래프|창의적 글쓰기]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/창의적 글쓰기 근거 인덱스|창의적 글쓰기 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/creative-writing/창의적글쓰기 강의자료 중핵교양|창의적글쓰기 강의자료 중핵교양]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/creative-writing/source id|source id]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/creative-writing/pdf path|pdf path]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/creative-writing/page|page]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/creative-writing/creative writing|creative writing]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Creative Generation|Creative Generation]]

# 창의적글쓰기 과제 작업 절차

## 1. 자료 구조

- 원본 강의자료: `pdf/26-1창의적글쓰기 강의자료_중핵교양.pdf`
- 기존 중간 정리: `중간고사_창의적글쓰기_정리.md`
- 기존 실습 예시: `pdf/퀴즈.md`
- PDF 페이지별 추출 텍스트: `knowledge/source_text/pages/`
- PDF 출처 요약: `knowledge/source_map.md`

## 2. 과제 시작 루틴

1. 과제 프롬프트에서 `주제`, `형식`, `분량`, `제출 조건`, `강의 범위`를 뽑는다.
2. 핵심어를 3-5개로 나눈다.
3. 아래처럼 근거를 검색한다.

```bash
python3 tools/search_sources.py "Hi Five"
python3 tools/search_sources.py "5문단"
python3 tools/search_sources.py "MECE"
```

4. 필요한 페이지 원문을 직접 확인한다.

```bash
sed -n '1,220p' knowledge/source_text/pages/26-1창의적글쓰기_강의자료_중핵교양/page-024.txt
```

Apple Notes 확인 결과, 창의적글쓰기 과제와 직접 연결되는 실질 단서는 `과제` 메모의 `헤더 100퍼`였다. 따라서 과제 산출물은 항상 제목/헤더를 먼저 분명하게 잡고, 본문 구조보다 위에 배치한다.

## 3. 글 설계

수업 자료상 이 과목은 감동 위주의 `impress`보다 전달 중심의 `express` 글쓰기를 강하게 다룬다. 과제도 기본적으로 독자가 바로 이해하는 구조를 우선한다.

기본 골격:

- 제목: 주장이 드러나게 쓴다.
- 형식: `why` 또는 `how` 중 하나로 먼저 결정한다.
- 서론: 필요하면 미괄식으로 문제 상황을 깔고 마지막에 주장을 둔다.
- 본론 1-3: 각 문단 첫 문장에 핵심 주장을 둔다.
- 결론: 본론을 압축하고 마지막 문장으로 주제를 강조한다.
- 암기 키워드 흐름: 제출 요구가 없어도 발표/암기용으로 유지하면 좋다.

문장 힘 번호:

- `(1)`: 주제 또는 핵심 주장
- `(2)`: 이유, 방법, 전개 근거
- `(3)`: 구체 자료, 예시, 사실
- `(4)`: 주제 강조 또는 결론

## 4. 근거 사용 규칙

- 강의 개념은 반드시 페이지 근거를 붙인다.
- 기존 Markdown 정리는 빠른 길잡이로 쓰되, 중요한 개념은 원본 PDF 페이지와 대조한다.
- 외부 기술/시사 사실은 강의자료 근거가 아니다. 필요한 경우 웹 검증 출처를 별도로 둔다.
- 자료에 없는 교수 의도는 추정하지 않는다.

## 5. 수정 체크리스트

- 제목/헤더가 선명한가?
- 3C: 정확하고, 간결하고, 명료한가?
- Hi Five: 분석-주제-근거-증명-강조 흐름이 보이는가?
- MECE: 본론 3개가 겹치거나 빠진 축 없이 나뉘었는가?
- 두괄식/미괄식 배치가 의도와 맞는가?
- 모든 강의 개념과 핵심 사실에 근거가 있는가?

초안 파일 검토:

```bash
python3 tools/check_grounding.py assignments/drafts/<draft>.md
```
