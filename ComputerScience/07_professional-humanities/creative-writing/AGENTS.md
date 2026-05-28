domain:: [[ComputerScience/07_professional-humanities/전문 교양 인터페이스|전문 교양 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/6단계 전문 확장 인터페이스|6단계 전문 확장 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/창의적 글쓰기 인터페이스|창의적 글쓰기 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/산출물 포트폴리오 브리지|산출물 포트폴리오 브리지]]
up:: [[ComputerScience/07_professional-humanities/creative-writing/중간고사_창의적글쓰기_정리|중간고사_창의적글쓰기_정리]]
related:: [[ComputerScience/07_professional-humanities/creative-writing/docs/WRITING_WORKFLOW|WRITING_WORKFLOW]], [[ComputerScience/07_professional-humanities/creative-writing/knowledge/source_map|source_map]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/self_intro_assignment2_starl_skeleton|self_intro_assignment2_starl_skeleton]], [[ComputerScience/07_professional-humanities/creative-writing/templates/writing-assignment|writing-assignment]], [[ComputerScience/07_professional-humanities/creative-writing/docs/CODEX_RAG_NOTES|CODEX_RAG_NOTES]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/self_intro_ax_archive_draft|self_intro_ax_archive_draft]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/self_intro_assignment2_full_draft|self_intro_assignment2_full_draft]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/deco_extracurricular_evidence|deco_extracurricular_evidence]], [[ComputerScience/07_professional-humanities/creative-writing/pdf/퀴즈|퀴즈]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/self_intro_complexity_hardest_experience|self_intro_complexity_hardest_experience]], [[ComputerScience/07_professional-humanities/degree-portfolio/PDF_인쇄_완전가이드|PDF_인쇄_완전가이드]], [[ComputerScience/07_professional-humanities/degree-portfolio/GovOn 온프레미스 AI 발표 스크립트|GovOn 온프레미스 AI 발표 스크립트]], [[ComputerScience/07_professional-humanities/classics-reading/멋진신세계|멋진신세계]], [[ComputerScience/07_professional-humanities/degree-portfolio/졸업학점|졸업학점]]

# Creative Writing Course Instructions

이 폴더는 `창의적글쓰기` 수업 과제 작업 공간이다. 답변과 산출물은 기본적으로 한국어로 작성한다.

## Source Grounding

- 과제 답안은 강의자료 PDF와 이 폴더의 기존 정리/퀴즈 자료를 먼저 근거로 삼는다.
- 원본 우선순위는 `pdf/26-1창의적글쓰기 강의자료_중핵교양.pdf` > `중간고사_창의적글쓰기_정리.md` > `pdf/퀴즈.md` 순서다.
- 과제 작성 전 `python3 tools/search_sources.py "<핵심어>"`로 관련 페이지를 찾고, 필요한 경우 `knowledge/source_text/pages/.../page-###.txt`를 직접 읽는다.
- 강의 근거는 `[강의자료 p.24]`, `[정리 p.6]`처럼 페이지 또는 파일 위치를 표시한다. 근거가 불명확하면 단정하지 말고 `근거 확인 필요`로 남긴다.
- PDF/정리에서 확인되지 않은 외부 사실은 수업 개념 근거와 분리한다. 사용자가 최신 사실 검증을 원하거나 과제 주제가 외부 정보를 요구하면 웹 검증 후 출처를 별도로 표시한다.
- 교수 발언, 시험 출제 의도, 채점 기준은 자료에 명시된 경우에만 단정한다.

## Course Writing Shape

- 수업 핵심 개념: Express 중심 글쓰기, Technical Writing 3C(Correct, Concise, Clear), Prewriting/Writing/Revising, Power Writing, Hi Five 5원리, Why/How 형식, MECE, 5문단 구조.
- 기존 퀴즈 예시는 `제목` + `형식` + `1' 서론` + `2' 본론1/2/3` + `4' 결론` + `암기 핵심 키워드 흐름` 구조를 사용했다.
- 문장 번호는 수업 방식에 맞춰 `(1)` 주제, `(2)` 근거/방법, `(3)` 증명/자료, `(4)` 주제 강조로 쓴다.
- 문단은 두괄식/미괄식 배치를 의식한다. 서론과 결론은 미괄식이 필요한지 먼저 판단하고, 본론은 보통 두괄식으로 쓴다.
- 본론 3개 축은 서로 겹치지 않게 MECE로 나누고, 각 축은 수직 주종 관계가 드러나게 전개한다.

## Workflow

1. 과제 프롬프트, 요구 분량, 제출 형식, 주제 제한을 먼저 파악한다.
2. 관련 강의 개념을 `tools/search_sources.py`로 검색하고 근거 표를 만든다.
3. Pre-Writing 단계에서 제목, 주장, 독자, Why/How 형식, 1-2-3-4 문장 구조를 설계한다.
4. 초안을 작성할 때 과제용 본문에는 필요한 만큼만 근거 표시를 넣고, 별도 `근거 메모` 섹션에는 상세 출처를 보존한다.
5. Revising 단계에서 3C, MECE, 문장 번호, 헤더/제목, 근거 누락을 점검한다.
6. 초안 파일은 `assignments/drafts/`, 최종본은 `assignments/final/`에 둔다.

## Verification

- PDF 인덱스가 없거나 오래된 경우 `python3 tools/build_knowledge_base.py`를 실행한다.
- 초안 검토에는 `python3 tools/check_grounding.py <draft.md>`를 사용해 긴 무출처 문단을 찾는다.
- 완료 전 확인할 것: 제목/헤더 존재, 요구 형식 충족, 강의 개념 적용, 근거 출처 표시, 외부 사실 검증 여부, 불필요한 과장 제거.
