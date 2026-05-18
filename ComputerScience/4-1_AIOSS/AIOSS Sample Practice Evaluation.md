---
aliases: []
course: AI OSS
created: '2026-05-18'
date: '2026-05-18'
semester: 4-1
source: sample package
status: seedling
tags:
  - type/evaluation
  - topic/github-actions
  - topic/testing
title: AIOSS Sample Practice Evaluation
type: note
updated: '2026-05-18'
---

up:: [[AIOSS LLMOps RAG Setup]]

# AIOSS Sample Practice Evaluation

> [!summary]
> `sample/` 패키지를 베이스라인 실패 → 구현 → 평가 → 디버깅 루프로 진행했다. 미완성 풀이본과 완성 예시 모두 최종 8/8 checks 통과 상태다.

## 결과

| 대상 | 초기 결과 | 최종 결과 | 핵심 디버깅 |
| --- | ---: | ---: | --- |
| `sample-solutions-minimal` | 2/8 | 8/8 | greeting 구현, PR template 완성, GitHub Actions 실제 step 작성, calculator 테스트/구현 완료 |
| `sample-solutions` | 5/8 | 8/8 | PR template 평가 기준 보정, 테스트 의도 평가 보정, unused `pytest` import 제거 |

## 적용한 샘플 풀이

- 샘플 1: `get_greeting(name)` 구현과 PR 설명 완성
- 샘플 2: `actions/checkout@v6.0.2`, `actions/setup-python@v6.2.0`, Python 3.10, `ruff check .` 기반 CI 작성
- 샘플 3: `add`, `subtract` 테스트와 최소 구현 작성, TDD cycle 기록

## 검증 명령

```bash
python3 tools/aioss_eval/sample_eval.py --target sample/sample-solutions-minimal --label minimal-final
python3 tools/aioss_eval/sample_eval.py --target sample/sample-solutions --label solution-after-fix
ruff check sample/sample-solutions sample/sample-solutions-minimal tools/aioss_eval
actionlint sample/sample-solutions/sample-2-ci-basics/.github/workflows/ci.yml sample/sample-solutions-minimal/sample-2-ci-basics/.github/workflows/ci.yml
```

## 평가 인사이트

- 제공된 완성본도 검증 대상이다. 실제로 unused import와 평가 기준 불일치가 발견됐다.
- CI YAML은 일반 YAML 파서보다 `actionlint`로 검증해야 한다. GitHub Actions의 `on` 키는 일반 YAML 파서에서 boolean으로 잘못 해석될 수 있다.
- 샘플 문제의 본질은 코드 정답보다 증거 체인이다. baseline 실패, 수정 내용, verification command, rollback plan을 남기는 습관이 점수에 직접 연결된다.
- RAG는 시험 중 답을 생성하는 도구가 아니라 근거를 찾는 도구로 써야 한다. 먼저 수업자료 chunk를 찾고, 최신 도구 버전이나 생태계 동향은 공식 릴리스와 primary source로 보강한다.

## 다음 확장

- Docling 기반 PDF layout 파서로 기존 FTS index의 품질을 높인다.
- Qdrant hybrid retrieval을 붙여 dense+sparse 검색과 reranking을 비교한다.
- RAGAS 또는 TruLens로 answer relevance, context relevance, groundedness를 샘플 리포트에 추가한다.
- 실제 시험 문제 5개를 가정해 `sample_eval.py`의 check registry를 확장한다.
