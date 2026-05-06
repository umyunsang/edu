# 창의적글쓰기 과제 작업 공간

이 폴더는 강의자료 PDF를 근거로 창의적글쓰기 과제를 작성하기 위한 작업 공간입니다.

## 빠른 사용법

PDF 인덱스 재생성:

```bash
python3 tools/build_knowledge_base.py
```

근거 검색:

```bash
python3 tools/search_sources.py "MECE" -n 5
python3 tools/search_sources.py "5문단" --primary-only -n 5
python3 tools/search_sources.py "두괄식 미괄식" --pdf-only -n 8
```

초안 검토:

```bash
python3 tools/check_grounding.py assignments/drafts/<draft>.md
```

## 주요 파일

- `AGENTS.md`: Codex가 이 폴더에서 자동으로 따를 과제 작성 규칙
- `.agents/skills/creative-writing-assignment/SKILL.md`: 반복 과제용 Codex 스킬
- `docs/WRITING_WORKFLOW.md`: 과제 작성 절차
- `docs/CODEX_RAG_NOTES.md`: Codex/RAG 셋업 근거
- `templates/writing-assignment.md`: 과제 초안 템플릿
- `knowledge/source_map.md`: PDF 인덱스 출처 지도

## 기본 원칙

- 원본 강의자료 PDF를 최우선 근거로 사용합니다.
- 기존 정리 Markdown은 빠른 길잡이로 쓰되, 핵심 개념은 원본 PDF와 대조합니다.
- 과제 본문은 수업 형식인 Power Writing, Hi Five, MECE, 1'-2'-2'-2'-4' 구조를 우선 적용합니다.
- 자료에 없는 사실은 단정하지 않고 외부 검증이 필요한 내용으로 분리합니다.
