---
template:
  title: 과목 인덱스
  description: 한 과목의 진입점. 학습 경로 다이어그램·노트 목록·원본 자료 대응표를 둔다. 파일명은 `00. 인덱스.md`.
  tags:
    - index
    - openknowledge
title: ""
description: ""
type: course-index
tags: []
course: ""
semester: ""
status: draft
aliases: []
created: "{{date}}"
updated: "{{date}}"
---

> [!abstract] {이 과목이 다루는 것 한 줄}

## 학습 경로

```mermaid
flowchart LR
    N1["01. {주제}"] --> N2["02. {주제}"]
    N2 --> N3["03. {주제}"]
```

## 노트 목록

| # | 노트 | 다루는 것 |
| :-- | :-- | :-- |
| 01 | {노트 링크} | {한 줄 요약} |

## 원본 자료

| 파일 | 페이지 | 대응 노트 |
| :-- | --: | :-- |
| `pdf/{파일명}.pdf` | {n} | {노트 링크} |

## 이 과목이 연결되는 곳

- **prerequisite** — {선수 과목 링크} : {왜 먼저인가}
- **applies-to** — {응용 과목 링크} : {어디에 쓰이는가}
