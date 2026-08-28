---
template:
  title: 강의 정리문서
  description: 강의 PDF를 근거로 쓰는 수업 정리문서. 증거는 렌더된 슬라이드 페이지 이미지로만 남긴다.
    콜아웃·Mermaid·KaTeX·html preview 차트·Accordion 을 쓴다.
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

%% 증거 규칙 — 반드시 지킬 것
   슬라이드 근거는 **렌더된 페이지 이미지**로만 남긴다. `p.12` 같은 페이지 번호 표기나
   PDF 링크만 적는 것은 근거가 아니다. 절차는 이렇다—
   1. `.ok/local/pdf-extract/<도메인>__<과목>__<slug>/pages/pNNN.png` 에서 해당 페이지를 고른다
   2. `<과목>/assets/<slug>-pNNN.png` 로 복사한다 (인용하는 페이지만 복사한다)
   3. 아래처럼 임베드하고, alt 텍스트에 그림이 무엇을 보이는지 적는다
   한 노트당 5–8장을 목표로 한다. %%

## 1. {첫 번째 주제}

{설명}. =={핵심 용어}== 는 {정의}.

![{그림 번호·제목 — 이 슬라이드가 보이는 것}](<../assets/{slug}-p{NNN}.png>)

{그림에서 눈여겨볼 점 하나}.

## 2. {두 번째 주제}

![{그림 번호·제목 — 이 슬라이드가 보이는 것}](<../assets/{slug}-p{NNN}.png>)

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

%% 라벨은 5종만 쓴다: prerequisite · elaborates · contrasts · applies · evidences
   반대 방향은 적지 않는다 (backlink 가 자동 계산된다). contrasts 만 양쪽에 적는다.
   속한 과목은 폴더 경로가 말하므로 간선으로 적지 않는다. %%

- **prerequisite** — [{선수 노트}](<./{선수 노트}.md>) : {왜 먼저 봐야 하는가}
- **applies** — [{응용 대상 노트}](<./{응용 대상 노트}.md>) : {무엇에 쓰는가}

> [!question]- 스스로 점검
> **Q.** {질문}
>
> **A.** {답}

## 출처

- [{파일명}.pdf](<../sources/{파일명}.pdf>) — {n}쪽. 본문에 인용한 슬라이드는 위 이미지로 남겼다.
