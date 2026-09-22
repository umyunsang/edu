---
title: 인공지능
description: "동아대 컴퓨터AI공학부 2024년 1학기 인공지능 강의 덱을 원본 PDF만을 근거로 재작성하는 중인 정리문서. 현재 Perceptron 이론·실습과 MLP 이론 세 덱(92쪽)이 네 편으로 정리되어 있고, 남은 열여섯 덱(약 533쪽)은 대기 중이다."
type: course-index
tags:
  - course
  - artificial-intelligence
  - deep-learning
  - perceptron
  - neural-network
course: artificial-intelligence
semester: "2024-1"
status: draft
created: "2026-09-22"
updated: "2026-09-22"
slides: true
---

> [!warning] 진행 중인 과목이다
> `sources/` 의 PDF 29편은 파일명이 `… 2.pdf` 로 끝나는 **중복 열 편**(바이트 크기가 원본과 같다)을 걷어내면 **실제 19편(약 625쪽)** 이다. 2026-09-22 회차가 그중 셋(`Perceptron (이론)` 61쪽 · `Perceptron (실습)` 8쪽 · `MLP(이론)` 23쪽)을 읽었다. 남은 열여섯 편의 목록은 [05. 남은 열여섯 덱 (재작성 대기)](<./notes/05. 남은 열여섯 덱 [재작성 대기].md>)에 있다.

> [!abstract] 이 과목의 원본에 대해
> 모든 덱의 저자가 같다 — `Dongsan Jun (dsjun@dau.ac.kr), Image Signal Processing Laboratory, Dept. of AI, Dong-A University`. 그리고 **모든 덱의 두 번째 장이 같은 그림**이다(`Overall Architecture of Deep Learning`). 그 한 장이 `sources/` 의 덱 목록과 거의 그대로 대응하고, 덱이 진행될수록 꼬리표가 붙는다 — `MLP(이론)` 판에는 `BN` 과 `Vanishing Gradient (Activation Function)` 이 추가되어 있다.
>
> **텍스트 추출이 잘 작동한다.** `Perceptron (이론)` 은 61쪽 중 52쪽에서 쪽당 180자 안팎이 잡히고, `MLP(이론)` 은 23쪽 중 22쪽이 잡힌다. 예외는 `필기 과제.pdf` 하나로, 10쪽 전체가 손글씨라 추출이 의미를 이루지 않는다.
>
> **쪽 꼬리표가 총 쪽수를 넘는 덱이 둘 있다** — `Perceptron (이론)` 은 61쪽인데 꼬리표가 `/59`(마지막 두 장이 `60/59` · `61/59`), `Perceptron (실습)` 은 8쪽인데 `/7`. `MLP(이론)` 은 2번 슬라이드 하나만 `2/39` 로, 다른 덱의 두 번째 장을 가져오면서 꼬리표가 따라온 것으로 보인다.

## 정리문서

→ **[정리문서 목록](<./notes/index.md>)**

| 순서 | 문서 | 원본 |
| --- | --- | --- |
| 01 | [빌려 온 그림이 양쪽을 다 찌른다](<./notes/01. 빌려 온 그림이 양쪽을 다 찌른다.md>) | `Perceptron (이론)` 1~17 |
| 02 | [예측을 먼저 보여 주고, 그 예측이 나오지 않는 모델을 보여 준다](<./notes/02. 예측을 먼저 보여 주고, 그 예측이 나오지 않는 모델을 보여 준다.md>) | `Perceptron (이론)` 18~40 |
| 03 | [다섯 장에 걸쳐 XOR을 쌓는 동안 이름이 두 번 바뀌고 부호가 한 번 틀린다](<./notes/03. 다섯 장에 걸쳐 XOR을 쌓는 동안 이름이 두 번 바뀌고 부호가 한 번 틀린다.md>) | `Perceptron (이론)` 41~61 · `Perceptron (실습)` |
| 04 | [목록은 길고 고르는 기준은 한 줄이다](<./notes/04. 목록은 길고 고르는 기준은 한 줄이다.md>) | `MLP (이론)` 전체 |
| 05 | [남은 열여섯 덱 (재작성 대기)](<./notes/05. 남은 열여섯 덱 [재작성 대기].md>) | — |

## 원본 자료

중복을 걷어낸 19편. ✅ 는 이번 회차에 읽은 것이다.

| | 파일 | 쪽 |
| --- | --- | ---: |
| ✅ | `Perceptron (이론).pdf` | 61 |
| ✅ | `Perceptron (실습).pdf` | 8 |
| ✅ | `MLP(이론).pdf` | 23 |
| ⬜ | `MLP(실습).pdf` | 37 |
| ⬜ | `Backpropagation (이론).pdf` | 85 |
| ⬜ | `필기 과제.pdf` — 전 쪽 손글씨 | 10 |
| ⬜ | `Optimization(이론).pdf` | 21 |
| ⬜ | `Optimizer.pdf` | 19 |
| ⬜ | `Overfitting.pdf` | 37 |
| ⬜ | `Vanishing Gradient Effect.pdf` | 49 |
| ⬜ | `CNN (이론).pdf` | 74 |
| ⬜ | `CNN (실습).pdf` | 27 |
| ⬜ | `CNN Backpropagation(이론).pdf` | 34 |
| ⬜ | `CNN 주요설계모듈(이론).pdf` | 36 |
| ⬜ | `CNN 주요설계모듈(실습).pdf` | 33 |
| ⬜ | `VGGNet (실습).pdf` | 34 |
| ⬜ | `CIFAR10 (실습).pdf` | 8 |
| ⬜ | `AI 아바타 만들기 (실습).pdf` | 24 |
| ⬜ | `24_인공지능_중간고사.pdf` | 5 |

전체 슬라이드는 저장하지 않는다. 정리문서가 인용하는 장면만 `assets/` 에 잘라 두었다.
