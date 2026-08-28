---
title: 00. 신경망 인덱스
description: 퍼셉트론의 한계에서 출발해 학습을 실제로 동작시키는 기술까지, 신경망 과정 5편의 진입점.
type: course-index
tags:
  - deep-learning
  - index
course: neural-networks
semester: 3-1
status: stable
aliases:
  - 신경망 인덱스
  - Neural Networks MOC
  - AIE309
created: 2026-08-28
updated: 2026-08-28
---
> [!abstract] 이 과정이 다루는 것
> 직선 하나로 XOR을 못 가르는 데서 시작해,
> 층을 쌓고 · 손실을 정의하고 · 기울기를 효율적으로 구하고 · 실제로 돌아가게 만드는 까지.

## 학습 경로

```mermaid
flowchart LR
    N1["01. 퍼셉트론<br/>한계를 본다"] --> N2["02. 인공신경망과<br/>활성화 함수"]
    N2 --> N3["03. 신경망 학습<br/>손실과 기울기"]
    N3 --> N4["04. 오차역전파법<br/>빠르게 구하기"]
    N4 --> N5["05. 학습 기술들<br/>실제로 돌아가게"]
```

다섯 편이 하나의 질문을 순서대로 밀고 나간다 — 각 편이 앞 편이 남긴 문제를 받는다.

| # | 노트 | 남기는 문제 |
| :-- | :-- | :-- |
| 01 | [퍼셉트론](<./notes/01. 퍼셉트론.md>) | 직선 하나로는 XOR을 못 가른다 |
| 02 | [인공신경망과 활성화 함수](<./notes/02. 인공신경망과 활성화 함수.md>) | 층은 쌓았는데, 무엇을 기준으로 학습하나 |
| 03 | [신경망 학습](<./notes/03. 신경망 학습.md>) | 기울기를 수치미분으로 구하면 3억 번 계산이다 |
| 04 | [오차역전파법](<./notes/04. 오차역전파법.md>) | 기울기는 구했는데 학습이 잘 안 된다 |
| 05 | [학습 기술들](<./notes/05. 학습 기술들.md>) | — |

## 원본 자료

```html preview
<div style="font-family:system-ui,sans-serif;padding:20px;color:var(--foreground)">
  <h3 style="margin:0 0 4px;font-size:15px;font-weight:600">강의별 슬라이드 분량</h3>
  <p style="margin:0 0 16px;font-size:13px;color:var(--muted-foreground)">
    뒤로 갈수록 분량이 늘어난다 — 05번은 01번의 일곱 배다.
  </p>
  <div id="pg" style="display:flex;align-items:flex-end;gap:16px;height:160px"></div>
  <script>
    var d = [['01', 7], ['02', 30], ['03', 30], ['04', 34], ['05', 51]];
    var mx = 51;
    document.getElementById('pg').innerHTML = d.map(function (x, i) {
      return '<div style="flex:1;display:flex;flex-direction:column;align-items:center;' +
        'gap:6px;height:100%;justify-content:flex-end">' +
        '<span style="font-size:13px;font-weight:700">' + x[1] + 'p</span>' +
        '<div style="width:100%;height:' + (x[1] / mx * 100) + '%;' +
        'background:var(--chart-' + (i + 1) + ');' +
        'border-radius:var(--radius) var(--radius) 0 0"></div>' +
        '<span style="font-size:12px;color:var(--muted-foreground)">' + x[0] + '</span>' +
        '</div>';
    }).join('');
  </script>
</div>
```

<details>
<summary>노트–PDF 대응표</summary>

| 노트 | 원본 파일 | 페이지 |
| :-- | :-- | --: |
| 01 | `pdf/2장_퍼셉트론.pdf` | 7 |
| 02 | `pdf/3장_인공신경망.pdf` | 30 |
| 03 | `pdf/AIE309_4장_신경망학습.pdf` | 30 |
| 04 | `pdf/AIE309_5장_오차역전파.pdf` | 34 |
| 05 | `pdf/AIE309_6장_학습기술들.pdf` | 51 |

모든 강의가 『밑바닥부터 시작하는 딥러닝』(사이토 고키, 한빛미디어)를 자료 출처로 말한다.

</details>

> [!note] 원본 PDF 메타데이터 이상
> 다섯 개 원본 PDF 전부 제목 메타데이터가 `The Effect of Mo or W on TiC Coarsening in HSLA Steel` 로 박혀 있다.
> 내용과 무관한 다른 문서의 제목이며, 템플릿 재사용 흔적으로 보인다. 본문 내용에는 영향이 없다.

## 이 과정을 관통하는 두 숫자

| 숫자 | 출처 | 의미 |
| --: | :-- | :-- |
| **39,760** | 03번 | 2층 신경망(784→50→10)의 학습 변수 개수 |
| **397,600,000** | 03번 | 수치미분으로 10,000 epoch 를 돌릴 때 필요한 계산 횟수 |

두 번째 숫자가 04번(역전파)이 존재하는 이유다.

## 이 과정이 연결되는 곳

- **applies-to** — [양자 ML 인덱스](<../quantum-ml/README.md>) : Variational Circuit 의 학습 구조가 여기 신경망과 같은 틀이다
- **contrasts** — [04. Quantum Feature Space](<../quantum-ml/notes/04. Quantum Feature Space.md>) : 같은 XOR 한계를 층이 아니라 공간 확장으로 푸는 접근
