---
template:
  title: 과목 인덱스
  description: 실제 학습 경로와 문서·원본 coverage를 공식 컴포넌트로 정리하는 과목 인덱스.
  tags:
    - index
    - openknowledge
    - visual
    - slides
title: "{과목명}"
description: "{과목 범위와 학습 목표}"
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

> [!NOTE]
> {과목 범위와 도달 목표}

---

## 학습 경로

{source-backed prerequisite or learning-path visual}

---

## 정리문서

| 번호 | 문서 | 핵심 질문 | 근거 상태 |
| :-- | :-- | :-- | :-- |
| {번호} | {표준 Markdown 상대 링크} | {한 줄 질문} | {완료·근거 부족·제외} |

---

## 범위 시각화

{source-backed coverage visual}

---

## 근거 범위

| 원본 식별자 | 담당 문서 | 상태 |
| :-- | :-- | :-- |
| {파일명} | {문서명} | {반영·중복·근거 부족} |

> [!warning]
> {비어 있는 source·중복 원본·추출 실패가 있을 때만 유지}
