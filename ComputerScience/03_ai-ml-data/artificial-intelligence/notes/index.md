---
title: 인공지능 — 정리문서 목록
description: 원본 강의 PDF만을 근거로 재작성한 네 편(01~04)과, 아직 읽지 않은 열여섯 덱의 자리 표시(05). 2026-09-22 회차가 Perceptron 이론·실습과 MLP 이론 세 덱(92쪽)을 읽었다.
type: index
tags:
  - artificial-intelligence
  - index
course: artificial-intelligence
semester: 2024-1
status: stable
created: 2026-09-22
updated: 2026-09-22
---

> [!warning] 진행 중인 과목이다
> `sources/` 의 PDF 29편에서 중복 열 편을 걷어낸 **실제 19편 중 셋**을 읽었다. 남은 열여섯 편(약 533쪽)은 [05. 남은 열여섯 덱 (재작성 대기)](<./05. 남은 열여섯 덱 [재작성 대기].md>)에 목록으로 적어 두었다.

## 1회차 — 퍼셉트론과 MLP (2026-09-22)

| 문서 | 원본 | 척추 |
| --- | --- | --- |
| [01. 빌려 온 그림이 양쪽을 다 찌른다](<./01. 빌려 온 그림이 양쪽을 다 찌른다.md>) | `Perceptron (이론)` 1~17 | 기호주의를 무너뜨리려고 빌려 온 CS231N 슬라이드의 네 난제가, 세 장 뒤에 나오는 퍼셉트론에도 그대로 해당한다 |
| [02. 예측을 먼저 보여 주고, 그 예측이 나오지 않는 모델을 보여 준다](<./02. 예측을 먼저 보여 주고, 그 예측이 나오지 않는 모델을 보여 준다.md>) | `Perceptron (이론)` 18~40 | 22번이 65를 예측하고 23번의 $y=8x+10$ 은 50을 낸다. 29번과 30번의 학습 데이터도 한 장 만에 바뀐다 |
| [03. 다섯 장에 걸쳐 XOR을 쌓는 동안 이름이 두 번 바뀌고 부호가 한 번 틀린다](<./03. 다섯 장에 걸쳐 XOR을 쌓는 동안 이름이 두 번 바뀌고 부호가 한 번 틀린다.md>) | `Perceptron (이론)` 41~61 + `Perceptron (실습)` | 58번이 NAND의 편향을 본문에 −0.7, 그림에 0.7로 인쇄한다. 본문 값을 쓰면 XOR이 네 입력 모두 0을 낸다 |
| [04. 목록은 길고 고르는 기준은 한 줄이다](<./04. 목록은 길고 고르는 기준은 한 줄이다.md>) | `MLP (이론)` 전체 | 활성화 함수 여덟 개 중 용도가 적힌 것은 셋. 그리고 인쇄된 tanh 식은 tanh가 아니라 tanh(x/2)다 |
| [05. 남은 열여섯 덱 [재작성 대기]](<./05. 남은 열여섯 덱 [재작성 대기].md>) | — | 남은 원본 목록과, 앞 네 노트가 넘긴 빚 |

## 이 과목의 지도

모든 덱의 두 번째 장이 같다 — `Overall Architecture of Deep Learning`. 그 한 장이 `sources/` 의 덱 목록과 거의 그대로 대응하고, 덱이 진행될수록 꼬리표가 조금씩 붙는다(`MLP (이론)` 판에는 `BN` 과 `(Activation Function)` 이 추가되어 있다).

```text
Training Input → Network(FC) → Activation(Step·Sigmoid·ReLU·PReLU) → Drop-Out
              → Loss(MAE·MSE) → Optimization(GD·Momentum·Adam)
              ↑ Backward (Backpropagation) ← Vanishing Gradient
Test Input → Trained Network → Evaluation(PSNR·SSIM·Total Memory)
Overfitting
```
