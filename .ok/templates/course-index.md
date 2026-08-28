---
template:
  title: 과목 인덱스
  description: 실제 정리문서·근거 범위·학습 순서를 연결하는 프로젝트 공통 과목 인덱스.
  tags:
    - index
    - openknowledge
    - slides
title: "{과목명}"
description: "{과목의 범위와 학습 목표}"
type: course-index
tags: []
course: "{과목명}"
semester: "{학기}"
status: draft
aliases: []
slides: true
created: "{{date}}"
updated: "{{date}}"
---

> [!abstract]
> {이 과목이 다루는 범위와 도달 목표}

## 학습 경로

{실제 선수 관계가 있으면 공식 Mermaid로 표시하고, 없으면 번호 목록으로 정리}

## 정리문서

| 번호 | 문서 | 핵심 질문 | 근거 상태 |
| :-- | :-- | :-- | :-- |
| {번호} | {표준 Markdown 상대 링크} | {한 줄 질문} | {완료·근거 부족·제외} |

## 근거 범위

| 원본 식별자 | 담당 문서 | 상태 |
| :-- | :-- | :-- |
| {파일명} | {문서명} | {반영·중복·근거 부족} |

> [!warning]
> {비어 있는 source·중복 원본·추출 실패 등 실제 누락이 있을 때만 유지}

## 학습 순서 요약

- **먼저:** {선수 개념}
- **다음:** {핵심 흐름}
- **마지막:** {응용 또는 종합}
