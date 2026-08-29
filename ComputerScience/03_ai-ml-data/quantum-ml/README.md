---
title: 00. 양자 ML 인덱스
description: 표현력의 한계에서 첫 QML 회로까지, 상태·회로·측정의 학습 경로를 안내한다.
type: course-index
tags: []
course: quantum-ml
semester: summer
source: ""
source_pages: 0
status: draft
aliases: []
created: "2026-08-29"
updated: "2026-08-29"
---

> [!abstract] 한 줄 요약
> 양자 ML은 표현, 상태, 회로, 측정을 하나의 하이브리드 흐름으로 읽는다.

## 강의 흐름 지도

[00. quantum-ml 강의 흐름 지도](<./notes/00. quantum-ml 강의 흐름 지도.md>)

## 이 과정의 지도

```mermaid
flowchart TB
    subgraph Course[학습 흐름]
        direction LR
        A[기초] --> B[상태]
    end
    B --> C[회로]
    C --> D[응용]
```

## 학습 경로

<Tabs>
  <Tab label="기초">표현력·양자 계산 배경·큐비트를 점검한다.</Tab>
  <Tab label="상태">특징 공간·게이트·상태 변화를 회로 관점으로 읽는다.</Tab>
  <Tab label="응용">측정값을 손실과 QML 회로로 연결한다.</Tab>
</Tabs>

```html preview
<div style="font-family:system-ui,sans-serif;padding:20px">
  <div id="cards" style="display:flex;gap:14px;flex-wrap:wrap"></div>
  <script>
    var stats = [['기초', '1', '표현과 배경', 'var(--chart-1)'], ['회로', '2', '게이트와 측정', 'var(--chart-2)'], ['응용', '3', '하이브리드 설계', 'var(--chart-3)']];
    document.getElementById('cards').innerHTML = stats.map(function (s) {
      return '<div style="flex:1;min-width:150px;padding:16px;background:var(--card);' +
        'color:var(--card-foreground);border:1px solid var(--border);' +
        'border-radius:var(--radius)">' +
        '<div style="font-size:13px;color:var(--muted-foreground)">' + s[0] + '</div>' +
        '<div style="font-size:26px;font-weight:700;margin-top:4px">' + s[1] + '</div>' +
        '<div style="font-size:12px;font-weight:600;margin-top:4px;color:' + s[3] + '">' +
        s[2] + '</div>' +
        '</div>';
    }).join('');
  </script>
</div>
```

<details>
<summary>과정 사용법</summary>

각 노트에서는 학습 질문을 먼저 읽고, 상태·회로·측정 중 무엇이 바뀌는지 확인한 뒤 비교표와 점검 질문으로 돌아온다.

</details>

> [!tip] 순서의 이유
> ==상태 표현==을 이해한 뒤 게이트와 회로를 읽으면, 측정값을 단순한 숫자가 아니라 설계 결과로 해석할 수 있다.

## 노트 목록

- [표현력의 한계](<./notes/01. 표현력의 한계.md>)
- [왜 양자 컴퓨팅인가](<./notes/02. 왜 양자 컴퓨팅인가.md>)
- [Bit와 Qubit](<./notes/03. Bit와 Qubit.md>)
- [Quantum Feature Space](<./notes/04. Quantum Feature Space.md>)
- [QML에서 Quantum의 역할](<./notes/05. QML에서 Quantum의 역할.md>)
- [Hadamard Gate](<./notes/06. Hadamard Gate.md>)
- [상태변화 분석](<./notes/07. 상태변화 분석.md>)
- [Quantum Gate 개념](<./notes/08. Quantum Gate 개념.md>)
- [Quantum Circuit](<./notes/09. Quantum Circuit.md>)
- [Quantum Circuit과 QML](<./notes/10. Quantum Circuit과 QML.md>)
- [양자 ML 과정](<./notes/양자 ML 과정.md>)

| 구간 | 주된 질문 | 다음 단계 |
| :-- | :-- | :-- |
| 기초 | 무엇을 표현하는가 | 상태 |
| 상태 | 무엇이 바뀌는가 | 회로 |
| 응용 | 무엇을 측정하는가 | 평가 |

> [!question]- 스스로 점검
> **Q.** 다음 노트로 넘어가기 전에 적어둘 한 문장은 무엇인가?
>
> **A.** 현재 노트에서 입력·변환·관측 중 무엇이 핵심인지다.

## 출처

- 공개 노트에는 원본 강의 자료, 자산 이미지, 페이지 단위 근거를 포함하지 않는다.
