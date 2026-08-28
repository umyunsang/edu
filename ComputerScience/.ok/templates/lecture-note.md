---
template:
  title: 강의 정리문서
  description: 강의 PDF를 근거로 쓰는 수업 정리문서. 콜아웃·Mermaid·KaTeX·html preview
    차트·Accordion·Tabs 를 쓴다. 상세 규칙은 ComputerScience 폴더 설명 참고.
  tags:
    - lecture
    - openknowledge
title: ""
description: ""
type: lecture
tags: []
course: ""
semester: ""
source: ""
source_pages: 0
status: draft
aliases: []
prerequisite: []
created: "{{date}}"
updated: "{{date}}"
---

> [!abstract] 한 줄 요약
> {이 강의가 답하는 질문 하나}

## 이 노트의 지도

```mermaid
flowchart LR
    A["{출발점}"] --> B["{핵심 개념}"]
    B --> C["{도달점}"]
```

## 1. {첫 번째 주제}

> [!quote] 슬라이드 근거
> `pdf/{파일명}.pdf` p.{n}

![슬라이드 {n}](./assets/{slug}-p{nnn}.png)

{설명}. =={핵심 용어}== 는 {정의}.

## 2. {두 번째 주제}

<details>
<summary>유도 과정</summary>

$$
{수식}
$$

</details>

## 데이터로 보기

%% 정량 데이터가 없는 강의면 이 섹션을 통째로 지운다 %%

```html preview
<div style="font-family:system-ui,sans-serif;padding:20px;color:var(--foreground)">
  <h3 style="margin:0 0 14px;font-size:15px;font-weight:600">{차트 제목}</h3>
  <div id="bars" style="display:flex;align-items:flex-end;gap:14px;height:170px"></div>
  <script>
    var data = [['{항목1}', 0], ['{항목2}', 0]];
    var max = Math.max.apply(null, data.map(function (d) { return d[1]; })) || 1;
    document.getElementById('bars').innerHTML = data.map(function (d, i) {
      return '<div style="flex:1;display:flex;flex-direction:column;align-items:center;' +
        'gap:6px;height:100%;justify-content:flex-end">' +
        '<span style="font-size:12px;font-weight:600">' + d[1] + '</span>' +
        '<div style="width:100%;height:' + (d[1] / max * 100) + '%;' +
        'background:var(--chart-' + (i + 1) + ');' +
        'border-radius:var(--radius) var(--radius) 0 0"></div>' +
        '<span style="font-size:12px;color:var(--muted-foreground)">' + d[0] + '</span>' +
        '</div>';
    }).join('');
  </script>
</div>
```

## 핵심 정리

| 개념 | 정의 | 왜 중요한가 |
| :-- | :-- | :-- |
| {개념} | {정의} | {이유} |

## 관련 개념

- **prerequisite** — [{노트}](./{경로}.md) : {왜 먼저 봐야 하는가}
- **uses** — [{노트}](./{경로}.md) : {무엇에 도구로 쓰는가}

> [!question]- 스스로 점검
> **Q.** {질문}
>
> **A.** {답}

## 출처

- `pdf/{파일명}.pdf` ({n}p)
