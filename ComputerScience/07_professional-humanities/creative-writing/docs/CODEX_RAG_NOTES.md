domain:: [[ComputerScience/07_professional-humanities/전문 교양 인터페이스|전문 교양 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/6단계 전문 확장 인터페이스|6단계 전문 확장 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/창의적 글쓰기 인터페이스|창의적 글쓰기 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/산출물 포트폴리오 브리지|산출물 포트폴리오 브리지]]
up:: [[ComputerScience/07_professional-humanities/creative-writing/중간고사_창의적글쓰기_정리|중간고사_창의적글쓰기_정리]]
related:: [[ComputerScience/07_professional-humanities/creative-writing/docs/WRITING_WORKFLOW|WRITING_WORKFLOW]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/self_intro_assignment2_starl_skeleton|self_intro_assignment2_starl_skeleton]], [[ComputerScience/07_professional-humanities/creative-writing/templates/writing-assignment|writing-assignment]], [[ComputerScience/07_professional-humanities/creative-writing/knowledge/source_map|source_map]], [[ComputerScience/07_professional-humanities/creative-writing/AGENTS|AGENTS]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/self_intro_ax_archive_draft|self_intro_ax_archive_draft]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/deco_extracurricular_evidence|deco_extracurricular_evidence]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/self_intro_complexity_hardest_experience|self_intro_complexity_hardest_experience]], [[ComputerScience/07_professional-humanities/creative-writing/pdf/퀴즈|퀴즈]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/self_intro_assignment2_full_draft|self_intro_assignment2_full_draft]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/07_professional-humanities/degree-portfolio/GovOn 온프레미스 AI 발표 스크립트|GovOn 온프레미스 AI 발표 스크립트]], [[ComputerScience/07_professional-humanities/degree-portfolio/PDF_인쇄_완전가이드|PDF_인쇄_완전가이드]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT API|ChatGPT API]], [[ComputerScience/07_professional-humanities/classics-reading/멋진신세계|멋진신세계]], [[ComputerScience/07_professional-humanities/degree-portfolio/졸업학점|졸업학점]]

kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/창의적 글쓰기 지식그래프|창의적 글쓰기]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/창의적 글쓰기 근거 인덱스|창의적 글쓰기 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/creative-writing/창의적글쓰기 강의자료 중핵교양|창의적글쓰기 강의자료 중핵교양]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/creative-writing/source id|source id]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/creative-writing/pdf path|pdf path]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/creative-writing/creative writing|creative writing]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/creative-writing/ai|ai]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Creative Generation|Creative Generation]]

# Codex/RAG 셋업 메모

이 폴더의 셋업은 OpenAI 공식 문서 기준으로 다음 선택을 했다.

## Codex 지침 계층

Codex는 작업 전 `AGENTS.md`를 읽고, 글로벌 지침과 프로젝트 지침을 계층적으로 합친다. 그래서 이 폴더 루트에 `AGENTS.md`를 두어 과제 작성 원칙, 근거 우선순위, 검증 명령을 지속 지침으로 고정했다.

참고:

- https://developers.openai.com/codex/guides/agents-md
- https://developers.openai.com/codex/learn/best-practices

## Skill 사용

공식 문서상 반복 워크플로우는 스킬로 포장할 수 있다. 이 수업은 매 수업 말 과제가 반복되므로 `.agents/skills/creative-writing-assignment/SKILL.md`를 만들었다. 다음부터 이 폴더에서 과제 작성 요청을 하면 Codex가 이 스킬을 사용할 수 있다.

참고:

- https://developers.openai.com/codex/skills

## 로컬 RAG 기본값

API 키 없이 바로 쓸 수 있도록 로컬 검색을 기본으로 했다.

- `tools/build_knowledge_base.py`: PDF를 페이지 단위 텍스트로 추출
- `tools/search_sources.py`: 페이지/노트 단위 로컬 검색
- `tools/check_grounding.py`: 초안의 긴 무출처 문단 탐지

이 방식은 의미 기반 검색보다 단순하지만, 과제에서 중요한 `페이지 근거 확인`과 `환각 방지`에는 효과적이다.

## OpenAI File Search로 업그레이드하는 경우

OpenAI File Search는 Responses API에서 사용할 수 있는 hosted tool이며, 업로드한 파일을 vector store로 검색한다. 공식 문서에 따르면 keyword/semantic search, query rewrite, reranking을 제공하고, 기본 chunk는 800 tokens, overlap은 400 tokens다. PDF와 Markdown 모두 지원 파일 형식에 포함된다.

다만 이 과제 폴더는 교수 필기나 슬라이드 이미지까지 중요한 경우가 있으므로, 이미지/도표 파싱 한계는 주의해야 한다. 현재 로컬 추출 텍스트에서 누락되는 시각 정보는 PDF 원본을 직접 열어 확인하는 절차를 유지한다.

참고:

- https://developers.openai.com/api/docs/guides/tools-file-search
- https://developers.openai.com/api/docs/guides/retrieval
- https://developers.openai.com/api/docs/assistants/tools/file-search#how-it-works
- https://developers.openai.com/codex/mcp
