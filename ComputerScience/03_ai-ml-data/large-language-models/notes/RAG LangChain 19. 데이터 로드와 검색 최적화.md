---
title: RAG LangChain 19. 데이터 로드와 검색 최적화
description: 데이터 로드, 텍스트 분할, 인덱싱, Retrieval, Rerank를 검색 품질을 만드는 연쇄 단계로 정리한다.
type: lecture
tags:
  - rag
  - retrieval
  - rerank
course: large-language-models
semester: extracurricular
status: draft
created: '2026-08-29'
updated: '2026-08-29'
---

> [!abstract] 핵심 질문
> 검색 결과가 나쁘면 모델을 바꾸기 전에, 데이터가 어떻게 들어오고 어떻게 나뉘고 어떤 후보가 살아남는지부터 살펴야 한다.

강의의 RAG 실습은 데이터 로더, 텍스트 분할, 벡터 인덱싱, Retriever, Rerank를 연속된 결정으로 다룬다. 한 단계의 선택이 다음 단계의 입력을 바꾸므로, 검색 품질은 단일 파라미터가 아니라 파이프라인의 산물이다.

```mermaid
flowchart LR
  A["문서 원천"] --> B["Data Loader"]
  B --> C["Text Splitter"]
  C --> D["Embedding과 Index"]
  D --> E["Retriever"]
  E --> F["Rerank"]
  F --> G["생성 문맥"]
```

## 단계마다 남길 질문

| 단계 | 선택 | 품질에 미치는 영향 | 관찰 지표 |
| :-- | :-- | :-- | :-- |
| Data Load | 원천과 메타데이터 | 누락·중복·형식 오류 | 로드 수·문서 내용 |
| Text Split | 조각 크기와 겹침 | 문맥 보존·검색 정밀도 | 조각의 완결성 |
| Index | 표현과 저장 구조 | 비교 가능한 후보 공간 | 검색 가능 여부 |
| Retrieval | 후보 수와 방식 | 관련 문서 회수 | 상위 결과의 관련성 |
| Rerank | 후보 재정렬 | 문맥의 우선순위 | 최종 문맥 품질 |

> [!note] 분할은 압축이 아니다
> 텍스트 분할은 긴 자료를 작게 만드는 작업이 아니라, 질문에 필요한 문맥이 한 조각 안에 남도록 경계를 정하는 작업이다.

```html preview h=180
<div style="font-family:system-ui,sans-serif;padding:18px;color:var(--foreground)">
  <div style="font-weight:700;margin-bottom:12px">후보는 넓게 찾고, 문맥은 좁게 고른다</div>
  <div style="display:flex;gap:7px;align-items:end;height:78px">
    <div style="flex:1;height:90%;background:var(--chart-4);border-radius:var(--radius) var(--radius) 0 0"></div>
    <div style="flex:1;height:72%;background:var(--chart-3);border-radius:var(--radius) var(--radius) 0 0"></div>
    <div style="flex:1;height:55%;background:var(--chart-2);border-radius:var(--radius) var(--radius) 0 0"></div>
    <div style="flex:1;height:35%;background:var(--border);border-radius:var(--radius) var(--radius) 0 0"></div>
    <div style="flex:1;height:20%;background:var(--border);border-radius:var(--radius) var(--radius) 0 0"></div>
  </div>
  <div style="font-size:12px;color:var(--muted-foreground);margin-top:7px">검색 후보 → 재정렬 후 상위 문맥</div>
</div>
```

## 최적화의 초점

<Tabs>
<Tab label="회수율">

정답에 필요한 조각이 후보에 들어오는지를 본다. 너무 좁은 검색은 관련 문서를 처음부터 놓칠 수 있다.

</Tab>
<Tab label="정밀도">

상위 후보가 질문과 얼마나 직접 관련되는지 본다. 불필요한 문맥이 많으면 생성이 핵심을 놓칠 수 있다.

</Tab>
<Tab label="다양성">

유사한 조각만 반복해서 가져오지 않도록 조절한다. 서로 다른 근거를 함께 보게 하는 검색 전략이 필요할 수 있다.

</Tab>
</Tabs>

<details>
<summary>Rerank를 별도 단계로 두는 이유</summary>

초기 검색은 빠르게 넓은 후보를 찾는 데 초점을 둔다. Rerank는 그 후보 안에서 질문과의 관련성을 더 세밀하게 비교해, 제한된 생성 문맥에 어떤 조각을 넣을지 결정한다.

</details>

> [!tip] 실패 사례를 보관한다
> 검색 실패를 “답이 틀림”으로 한 줄 기록하지 말고, 로딩 누락·분할 경계·검색 누락·재정렬 실패 중 어디였는지 붙여 둔다.

## 정리

- 데이터 로드·분할·인덱싱·검색·재정렬은 하나의 품질 사슬이다.
- ==Rerank는 넓게 찾은 후보를 생성 문맥에 맞게 다시 고르는 단계==다.
- 검색 최적화는 정답률 하나보다 실패 단계의 식별에서 시작한다.

> [!warning] 측정의 함정
> 상위 몇 개 결과가 자연스러워 보여도 질문 유형이 바뀌면 실패할 수 있다. 다양한 질문과 경계 사례로 검색 결과를 확인한다.
