---
aliases: []
course: AI OSS
created: '2026-05-18'
date: '2026-05-18'
semester: 4-1
source: local setup
status: seedling
tags:
  - type/setup
  - topic/llmops
  - topic/rag
title: AIOSS LLMOps RAG Setup
type: note
updated: '2026-05-18'
---

up:: [[Home MOC]]

# AIOSS LLMOps RAG Setup

> [!summary]
> AIOSS 실습시험 대비용 로컬 셋업이다. 목표는 PDF 수업자료, 샘플 문제, GitHub/CI/TDD 실습 자료를 Codex가 즉시 검색하고, 샘플 풀이를 자동 평가할 수 있게 만드는 것이다.

## 구성

- `AGENTS.md`: 이 폴더 전용 작업 규칙
- `tools/aioss_eval/build_rag_index.py`: PDF, `md/`, `sample/` 자료를 SQLite FTS5 색인으로 변환
- `tools/aioss_eval/rag_query.py`: 색인 질의와 근거 스니펫 반환
- `tools/aioss_eval/sample_eval.py`: 샘플 1~3 자동 평가
- `.aioss-rag/requirements.txt`: 최신 RAG/LLMOps 확장 패키지
- `.codex/hooks.json`: 세션 종료 시 최신 평가 상태 요약
- `.agents/skills/aioss-course-rag`: 수업자료 검색용 Codex skill
- `.agents/skills/aioss-exam-llmops`: 실습시험 루프용 Codex skill

## 실행 명령

```bash
python3 tools/aioss_eval/build_rag_index.py --root .
python3 tools/aioss_eval/rag_query.py "GitHub Actions CI testing"
python3 tools/aioss_eval/sample_eval.py --target sample/sample-solutions-minimal --label minimal
```

## 설치된 확장 패키지

`.aioss-rag/.venv`는 Python 3.12.11로 생성했다.

- `docling==2.94.0`
- `qdrant-client==1.18.0`
- `rank-bm25==0.2.2`
- `sentence-transformers==5.5.0`
- `ragas==0.4.3`
- `torch==2.12.0`

확인 결과 `docling`, `qdrant_client`, `ragas`, `rank_bm25`, `torch`, `sentence_transformers` import가 가능하다. `sentence_transformers`는 첫 import가 20초를 넘길 수 있으므로 시험 중 기본 검증 경로에는 넣지 않는다.

## 최신 방법론 반영

- 문서 파싱: PDF 텍스트만 믿지 않고 Docling 기반 레이아웃 파싱으로 확장 가능하게 설계
- 검색: 로컬 FTS5를 기본값으로 두고, 필요 시 dense+sparse hybrid retrieval과 reranking으로 확장
- 평가: functional correctness, CI reproducibility, shift-left testing, open source readiness, RAG readiness를 분리 평가
- 운영: 네트워크와 API 키 없이도 기본 실습 평가가 돌아가고, 외부 패키지는 `.aioss-rag/.venv`에 격리

## 현재 색인 상태

- 색인 단위: PDF, `md/`, `sample/`
- chunk 수: 250
- 저장소: `.aioss-rag/index/fts.sqlite`
- manifest: `.aioss-rag/manifest.json`

## 시험 적용 방식

1. 문제를 읽으면 먼저 관련 수업자료를 `rag_query.py`로 찾는다.
2. 구현 전 baseline 실패를 기록한다.
3. 최소 구현 후 `sample_eval.py`, `pytest`, `ruff`, `actionlint`를 실행한다.
4. 실패 원인을 기능, CI, 테스트, 문서, 근거 부족 중 하나로 분류한다.
5. 수정 후 다시 평가하고 증거를 남긴다.
