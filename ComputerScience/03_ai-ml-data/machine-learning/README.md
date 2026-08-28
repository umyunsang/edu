---
title: "Machine Learning 학습 노트"
description: "회귀, 분류, 시퀀스, 이미지, Transformer를 다루는 학습 노트 모음"
type: lecture
course: machine-learning
semester: "2025-1"
status: stable
created: "2026-08-29"
updated: "2026-08-29"
tags:
  - machine-learning
---

이 과목은 회귀·분류에서 시퀀스 모델과 이미지·Transformer까지, 입력 표현·학습 규칙·검증 관점을 연결해 정리한다. 각 노트는 독립적으로 읽을 수 있고, 같은 주제 안에서는 아래 순서로 학습할 수 있다.

## 회귀 분석

- [01. 선형 회귀 이론](<./notes/01. 선형 회귀 이론.md>)
- [01. 단일 선형 회귀 - LSM과 경사 하강법](<./notes/01. 단일 선형 회귀 - LSM과 경사 하강법.md>)
- [01. 선형 회귀 실습 설계](<./notes/01. 선형 회귀 실습 설계.md>)
- [01. 다중 선형 회귀](<./notes/01. 다중 선형 회귀.md>)
- [01. 우버 요금 다중 선형 회귀](<./notes/01. 우버 요금 다중 선형 회귀.md>)

## 분류와 거리 기반 학습

- [01. SVM의 마진과 경사 하강법](<./notes/01. SVM의 마진과 경사 하강법.md>)
- [01. QP 기반 SVM과 마진](<./notes/01. QP 기반 SVM과 마진.md>)
- [01. SVM 실습과 결정 경계](<./notes/01. SVM 실습과 결정 경계.md>)
- [01. 엔트로피, 결정 트리와 KNN](<./notes/01. 엔트로피, 결정 트리와 KNN.md>)
- [01. 결정 트리와 KNN 실습](<./notes/01. 결정 트리와 KNN 실습.md>)
- [01. 머신러닝 대비 문제](<./notes/01. 머신러닝 대비 문제.md>)
- [01. SVM과 KNN 대비 문제](<./notes/01. SVM과 KNN 대비 문제.md>)

## 시퀀스와 언어 표현

- [01. RNN과 LSTM 기초](<./notes/01. RNN과 LSTM 기초.md>)
- [01. RNN·LSTM 판서 자료의 텍스트 추출 한계](<./notes/01. RNN·LSTM 판서 자료의 텍스트 추출 한계.md>)
- [01. Word2Vec과 순환 신경망 리뷰](<./notes/01. Word2Vec과 순환 신경망 리뷰.md>)
- [01. RNN·LSTM 실습](<./notes/01. RNN·LSTM 실습.md>)

## 이미지 모델

- [01. CNN과 U-Net 실습](<./notes/01. CNN과 U-Net 실습.md>)
- [01. CNN 기반 초해상도](<./notes/01. CNN 기반 초해상도.md>)

## Transformer

- [01. Transformer 언어 모델링](<./notes/01. Transformer 언어 모델링.md>)
- [01. Transformer Self-Attention](<./notes/01. Transformer Self-Attention.md>)
- [01. Transformer 예시 자료의 텍스트 추출 한계](<./notes/01. Transformer 예시 자료의 텍스트 추출 한계.md>)

## 읽기 기준

- 모델을 볼 때는 입력 표현, 학습 규칙, 평가 기준을 함께 확인한다.
- 수치 결과는 데이터 분할과 전처리 조건을 분리해 해석한다.
- 본문 텍스트가 충분하지 않은 자료는 추정하지 않고 텍스트 추출 한계 노트로 구분한다.
