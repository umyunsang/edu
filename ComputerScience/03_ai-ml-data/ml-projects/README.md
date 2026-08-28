---
title: 머신러닝 프로젝트
description: Python·고전 머신러닝·RAG 실습을 연결하는 학습 노트
type: course-index
tags:
- course
- machine-learning
course: ml-projects
semester: 3-1
status: draft
---

> [!abstract]
> 이 과목은 Python 데이터 처리, 고전 머신러닝 모형, 점진 학습, 결정 트리, RAG 챗봇을 하나의 프로젝트 학습 흐름으로 정리한다.

## 학습 노트

| 원본 | 노트 | 핵심 질문 |
| :-- | :-- | :-- |
| 01_K-NN | [K-최근접 이웃 분류](<./notes/01. K-최근접 이웃 분류.md>) | 가까운 이웃의 다수결은 어떻게 분류를 만드는가? |
| 02_Regression_K-NN | [K-최근접 이웃 회귀](<./notes/02. K-최근접 이웃 회귀.md>) | 이웃의 타깃 평균은 어떤 수치를 예측하는가? |
| 03_Regression_Lin&poly | [선형 회귀와 다항 특성](<./notes/03. 선형 회귀와 다항 특성.md>) | 입력 표현을 넓히면 선형 모형은 어떻게 달라지는가? |
| 04_Regression_Multiple | [다중 회귀와 특성 공학](<./notes/04. 다중 회귀와 특성 공학.md>) | 여러 특성과 조합 특성은 어떻게 쓰는가? |
| 05_Regression_Logistic | [로지스틱 회귀와 클래스 확률](<./notes/05. 로지스틱 회귀와 클래스 확률.md>) | 점수는 어떻게 확률 분포가 되는가? |
| 06_Regression_Incremental learning | [점진 학습과 확률적 경사 하강법](<./notes/06. 점진 학습과 확률적 경사 하강법.md>) | 도착하는 데이터로 모형을 어떻게 갱신하는가? |
| 07_DecisionTree_before | [결정 트리와 가지치기](<./notes/07. 결정 트리와 가지치기.md>) | 질문 기반 분류기를 어떻게 일반화하는가? |
| 2024세미나자료 | [Python 데이터와 평균제곱오차](<./notes/01. Python 데이터와 평균제곱오차.md>) | 데이터를 이름으로 다루고 오차를 어떻게 읽는가? |
| LangChain_RAG | [LangChain 기반 RAG 챗봇](<./notes/01. LangChain 기반 RAG 챗봇.md>) | 검색 문맥을 답변에 어떻게 연결하는가? |
| 인공지능특강 | [퍼셉트론과 벡터화](<./notes/01. 퍼셉트론과 벡터화.md>) | 가중합 분류와 배열 연산은 어떻게 이어지는가? |

## 학습 경로

Python과 데이터 → 분류와 회귀 → 특성 공학과 검증 → 점진 학습과 트리 → RAG 챗봇

> [!note]
> 각 노트는 원본 강의의 개념을 독립적으로 재구성했다. 원본 PDF와 슬라이드 이미지는 노트에 삽입하지 않는다.

## 원본 자료 상태

- `2024세미나자료 2.pdf`, `인공지능특강 2.pdf`는 같은 자료의 macOS 복제본으로 별도 노트를 만들지 않았다.
- `Python_Machine_Learning_Certificate.pdf`는 개인 이수 증명서이며 학습용 강의 자료가 아니므로 노트에서 제외했다.
