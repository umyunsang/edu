---
title: 양자컴퓨팅 특강
description: 양자 게이트·Braket·Grover/Shor·VQA·QML을 PDF 원본과 렌더 슬라이드로 재구성한 8편의 강의 정리
type: course-index
tags:
  - course
  - extracurricular
course: quantum-lecture
semester: extracurricular
status: draft
created: '2026-08-28'
updated: '2026-08-28'
---

> [!abstract] 이 과목은
> 양자 상태와 게이트에서 출발해 클라우드 실행, 알고리즘, 물리·선형대수, 화학·머신러닝 응용, 하드웨어까지 이어지는 2일 특강이다. 각 문서는 해당 PDF의 흐름을 다시 설계하고, 확인 가능한 슬라이드 렌더만 이미지 근거로 넣었다.

## 정리문서

PDF 강의자료 8개를 주제의 의존 순서와 학습 목적에 맞춰 나눴다.

| 문서 | 다루는 내용 |
| :-- | :-- |
| [01. 양자 기초 게이트 설명](<./notes/01. 양자 기초 게이트 설명.md>) | 블로흐 구면, 단일·다중 큐비트 게이트, 회로 해석 |
| [01-1. 양자 알고리즘 소개 — Grover와 Shor](<./notes/01-1. 양자 알고리즘 소개 — Grover와 Shor.md>) | 진폭 증폭과 주기 추출을 회로·복잡도로 비교 |
| [02. 양자클라우드 Braket 기초 사용법](<./notes/02. 양자클라우드 Braket 기초 사용법.md>) | AWS Braket 콘솔, 디바이스 선택, 노트북 실행·종료 |
| [03. 양자컴퓨팅을 위한 물리 및 선형대수](<./notes/03. 양자컴퓨팅을 위한 물리 및 선형대수.md>) | 상태 벡터, 고윳값, 유니터리, 텐서곱 |
| [03-1. 화학 알고리즘 소개 — VQA](<./notes/03-1. 화학 알고리즘 소개 — VQA.md>) | 변분 원리, VQE, 분자 바닥상태 에너지 |
| [04. 양자 머신러닝 알고리즘 소개 — QML](<./notes/04. 양자 머신러닝 알고리즘 소개 — QML.md>) | 데이터 인코딩, 양자 회로 학습, QuGAN 사례 |
| [05. 양자컴퓨팅 하드웨어에 대한 이해](<./notes/05. 양자컴퓨팅 하드웨어에 대한 이해.md>) | 초전도·이온 트랩·어닐러·중성 원자 플랫폼 |
| [05-1. 하이브리드 알고리즘 소개 — SQD](<./notes/05-1. 하이브리드 알고리즘 소개 — SQD.md>) | 샘플링과 고전 대각화를 결합한 전자 구조 계산 |

## 원본 자료

`sources/`의 PDF 9개가 출처 자산이다. 이 중 수료증 PDF는 행정 자료라 강의 정리문서에서 제외했고, 나머지 8개를 정리했다. 슬라이드 근거는 `assets/`에 실제로 존재하는 `.webp` 렌더만 사용했다.

- [1. 양자 기초 게이트 설명_Jun.2026.pdf](<./sources/1. 양자 기초 게이트 설명_Jun.2026.pdf>)
- [1. 양자 알고리즘 소개-Grover Shor Algorithm_Jun.2026.pdf](<./sources/1. 양자 알고리즘 소개-Grover Shor Algorithm_Jun.2026.pdf>)
- [2. 양자클라우드 Bracket 기초 사용법_Jun.2026.pdf](<./sources/2. 양자클라우드 Bracket 기초 사용법_Jun.2026.pdf>)
- [3. 양자컴퓨팅을 위한 물리 및 선형대수_Jun.2026.pdf](<./sources/3. 양자컴퓨팅을 위한 물리 및 선형대수_Jun.2026.pdf>)
- [3. 화학 알고리즘 소개-VQA_Jun.2026.pdf](<./sources/3. 화학 알고리즘 소개-VQA_Jun.2026.pdf>)
- [4. 양자 머신러닝 알고리즘 소개-QML_Jun.2026.pdf](<./sources/4. 양자 머신러닝 알고리즘 소개-QML_Jun.2026.pdf>)
- [5. 양자컴퓨팅 하드웨어에 대한 이해 _Jun.2026.pdf](<./sources/5. 양자컴퓨팅 하드웨어에 대한 이해 _Jun.2026.pdf>)
- [5. 하이브리드 알고리즘 소개-SQD_Jun.2026.pdf](<./sources/5. 하이브리드 알고리즘 소개-SQD_Jun.2026.pdf>)
- [MEGAZONE_Quantum_Computing_Certificate.pdf](<./sources/MEGAZONE_Quantum_Computing_Certificate.pdf>) — 행정 자료, 정리 대상에서 제외

## 실습

실행 완료된 노트북 7개와 산출물이 `work/`에 있다. 노트북 결과는 강의 설명을 보조하는 실습 자산으로만 사용하고, 중앙 fidelity ledger가 없는 자료의 수치 주장은 인용 근거로 승격하지 않았다.

| 종류 | 개수 |
| :-- | --: |
| 실행 완료 `.ipynb` | 7 |

## 관련 과목

> [!note] 관계 타입은 후속 단계에서 부여
> 이 과목의 문서 작성이 끝난 뒤 지식그래프 설계의 4계층·5가지 관계 타입·6개 검증 규칙에 따라 관계를 검토한다. 현재 문서에는 아직 관계 타입을 연결하지 않는다.
