---
title: "인공지능 학습 노트"
description: "퍼셉트론, 최적화, CNN과 이미지 분류를 다루는 학습 노트"
type: course-index
course: artificial-intelligence
semester: "2024-1"
status: stable
created: "2026-08-29"
updated: "2026-08-29"
tags:
  - artificial-intelligence
---

이 과목은 퍼셉트론에서 시작해 다층 신경망, 역전파, 최적화, 일반화, CNN 기반 이미지 분류까지를 연결한다. 각 문서는 독립적으로 읽을 수 있으며 아래 순서는 개념의 의존 관계를 고려한 학습 경로다.

## 퍼셉트론·MLP·역전파

- [01. Perceptron 논리 게이트 실습](<./notes/01. Perceptron 논리 게이트 실습.md>)
- [01. Perceptron 이론 - 신경망 구성](<./notes/01. Perceptron 이론 - 신경망 구성.md>)
- [01. Perceptron 이론 - 활성화와 최적화](<./notes/01. Perceptron 이론 - 활성화와 최적화.md>)
- [01. MLP 이론 - 다층 표현](<./notes/01. MLP 이론 - 다층 표현.md>)
- [01. MLP 실습 - 모델 구성](<./notes/01. MLP 실습 - 모델 구성.md>)
- [01. MLP 실습 - 학습과 평가](<./notes/01. MLP 실습 - 학습과 평가.md>)
- [01. Backpropagation의 연쇄 법칙](<./notes/01. Backpropagation의 연쇄 법칙.md>)
- [01. Backpropagation의 가중치 갱신](<./notes/01. Backpropagation의 가중치 갱신.md>)
- [01. Vanishing Gradient Effect](<./notes/01. Vanishing Gradient Effect.md>)
- [01. Vanishing Gradient 완화](<./notes/01. Vanishing Gradient 완화.md>)

## 최적화와 일반화

- [01. Optimization 이론 - 손실과 경사](<./notes/01. Optimization 이론 - 손실과 경사.md>)
- [01. Optimizer - 모멘텀](<./notes/01. Optimizer - 모멘텀.md>)
- [01. Optimizer - 적응적 학습률](<./notes/01. Optimizer - 적응적 학습률.md>)
- [01. Overfitting - 일반화 진단](<./notes/01. Overfitting - 일반화 진단.md>)
- [01. Overfitting - 규제와 조기 종료](<./notes/01. Overfitting - 규제와 조기 종료.md>)

## 이미지 분류와 CNN

- [01. CIFAR-10 분류 실습](<./notes/01. CIFAR-10 분류 실습.md>)
- [01. CNN 분류 실습](<./notes/01. CNN 분류 실습.md>)
- [01. CNN의 합성곱 원리](<./notes/01. CNN의 합성곱 원리.md>)
- [01. CNN의 공간 크기와 채널](<./notes/01. CNN의 공간 크기와 채널.md>)
- [01. CNN Backpropagation - AlexNet 구조](<./notes/01. CNN Backpropagation - AlexNet 구조.md>)
- [01. CNN Backpropagation - AlexNet 정규화](<./notes/01. CNN Backpropagation - AlexNet 정규화.md>)
- [01. CNN 설계 모듈 실습 - 연결 구조](<./notes/01. CNN 설계 모듈 실습 - 연결 구조.md>)
- [01. CNN 설계 모듈 실습 - Conv2d 매개변수](<./notes/01. CNN 설계 모듈 실습 - Conv2d 매개변수.md>)
- [01. CNN 주요 설계 모듈 이론](<./notes/01. CNN 주요 설계 모듈 이론.md>)
- [01. VGGNet 실습](<./notes/01. VGGNet 실습.md>)

## 실습 과제와 자료 한계

- [01. 중간고사 - MLP 기반 CIFAR-10 분류](<./notes/01. 중간고사 - MLP 기반 CIFAR-10 분류.md>)
- [01. AI 아바타 만들기 실습](<./notes/01. AI 아바타 만들기 실습.md>)
- [01. 필기 과제 자료의 텍스트 추출 한계](<./notes/01. 필기 과제 자료의 텍스트 추출 한계.md>)
- [01. 필기 과제 2 텍스트 추출 한계](<./notes/01. 필기 과제 2 텍스트 추출 한계.md>)

## 읽기 기준

- 입력 표현, 모델 구조, 손실과 평가 기준을 함께 확인한다.
- 실습 결과는 데이터 분할과 하이퍼파라미터를 분리해 기록한다.
- 읽을 수 있는 본문이 충분하지 않은 자료는 추정하지 않고 텍스트 추출 한계로 구분한다.
