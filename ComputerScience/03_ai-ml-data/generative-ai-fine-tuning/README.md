---
title: 생성형 AI와 파인튜닝
description: 생성형 AI 도구에서 LoRA·지시 튜닝·제조 생성 설계·Physical AI까지 이어지는 학습 경로
type: course-index
tags:
  - generative-ai
  - fine-tuning
  - multimodal
  - design
course: generative-ai-fine-tuning
semester: extracurricular
status: draft
aliases: []
slides: true
created: '2026-08-29'
updated: '2026-08-29'
---

> [!ABSTRACT]
> ==생성 도구를 나열하는 과정이 아니라, 모델·제작 흐름·경량 튜닝·공학 설계를 하나의 학습 경로로 연결한다.==

- **출발점:** 생성형 AI와 멀티모달 도구가 무엇을 바꾸었는가
- **전환점:** 디자인 제어와 LoRA·지시 튜닝은 어떻게 목적에 맞는 출력을 만드는가
- **도착점:** 생성 모델과 강화학습을 공학 설계·Physical AI에 어떻게 연결하는가

---

## 학습 경로

```mermaid
flowchart LR
    A["생성형 AI와 멀티모달"] --> B["디자인 탐색과 제어"]
    B --> C["LoRA와 지시 튜닝"]
    C --> D["이미지 학습 데이터 설계"]
    D --> E["제조 생성 설계"]
    E --> F["Physical AI와 DRL"]
```

> [!TIP]
> 도구 이름보다 **입력 → 제어 → 학습 → 평가 → 적용**의 연결을 먼저 잡으면 뒤의 사례를 같은 틀로 비교할 수 있다.

---

## 정리문서

| 번호 | 문서 | 핵심 질문 | 근거 상태 |
| :-- | :-- | :-- | :-- |
| 01 | [생성형 AI와 파인튜닝](notes/01.%20생성형%20AI와%20파인튜닝.md) | 생성형 AI의 확산과 모델 규모·국가 전략은 어떻게 연결되는가 | 반영 |
| 15 | [멀티모달 생성 워크플로](notes/15.%20멀티모달%20생성%20워크플로.md) | 이미지·영상·음향 도구를 하나의 제작 공정으로 어떻게 엮는가 | 반영 |
| 26 | [디자인 탐색과 제어](notes/26.%20디자인%20탐색과%20제어.md) | 비용 충격과 디자인 플랫폼·제어 도구는 탐색 방식을 어떻게 바꾸는가 | 반영 |
| 40 | [LoRA와 파라미터 효율 미세조정](notes/40.%20LoRA와%20파라미터%20효율%20미세조정.md) | 소형 어댑터는 배포·합성·스타일 학습을 어떻게 단순화하는가 | 반영 |
| 46 | [지시문 데이터와 대화 파인튜닝](notes/46.%20지시문%20데이터와%20대화%20파인튜닝.md) | Self-Instruct와 비전 LoRA는 학습 데이터를 어떻게 확장하는가 | 반영 |
| 50 | [이미지 LoRA 학습과 적용](notes/50.%20이미지%20LoRA%20학습과%20적용.md) | 가중치·데이터 수·체크포인트 선택을 어떻게 관리하는가 | 반영 |
| 60 | [제조 생성 설계와 성능 평가](notes/60.%20제조%20생성%20설계와%20성능%20평가.md) | 생성 설계를 novelty·강성·설명 가능성으로 어떻게 평가하는가 | 반영 |
| 72 | [강화학습과 생성 설계의 기초](notes/72.%20강화학습과%20생성%20설계의%20기초.md) | Physical AI와 DRL을 위해 어떤 선수 지식과 벤치마크가 필요한가 | 반영 |

---

## 문서·원본 구조

```html preview
<div style="font-family:system-ui,sans-serif;padding:20px">
  <div id="cards" style="display:flex;gap:14px;flex-wrap:wrap"></div>
  <script>
    var stats = [
      ['학습 노트', '8', '서로 겹치지 않는 주제 구간', 'var(--chart-2)'],
      ['원본 파일', '2', '서로 다른 파일 해시', 'var(--chart-1)'],
      ['고유 본문', '1', '동일한 추출 페이지 스트림', 'var(--chart-5)']
    ];
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

- **분할 원칙:** 한 노트는 하나의 독립 주제 흐름만 담당한다.
- **중복 처리:** 두 PDF를 별도 강의로 이중 계산하지 않는다.
- **번호 원칙:** 파일 번호는 담당 구간이 시작되는 원본 슬라이드 번호를 따른다.

---

## 범위 시각화

```html preview
<div style="font-family:system-ui,sans-serif;padding:20px;display:flex;align-items:center;gap:20px;color:var(--foreground)">
  <svg width="120" height="120" viewBox="0 0 120 120" role="img" aria-label="82 of 82 source slides covered">
    <circle cx="60" cy="60" r="46" stroke-width="14" style="fill: none; stroke: var(--border)" />
    <circle cx="60" cy="60" r="46" stroke-width="14"
      stroke-linecap="round" stroke-dasharray="289" stroke-dashoffset="0"
      transform="rotate(-90 60 60)" style="fill: none; stroke: var(--chart-1)" />
    <text x="60" y="67" text-anchor="middle" font-size="22" font-weight="700"
      style="fill: var(--foreground)">100%</text>
  </svg>
  <div>
    <div style="font-weight:600;font-size:15px">고유 추출 범위 반영</div>
    <div style="font-size:13px;color:var(--muted-foreground);margin-top:2px">82개 중 82개 슬라이드가 8개 노트에 배정됨</div>
  </div>
</div>
```

<details>
<summary>coverage 판정 기준</summary>

- 추출 본문이 같은 두 파일은 하나의 고유 페이지 스트림으로 계산한다.
- 각 슬라이드는 정확히 한 노트 구간에만 배정한다.
- 텍스트가 희박하거나 잘린 슬라이드는 내용을 추정하지 않고 해당 노트의 경고로 남긴다.

</details>

---

## 근거 범위

| 원본 식별자 | 역할 | 판정 |
| :-- | :-- | :-- |
| `_GenAI_FineTuning.pdf` | 8개 노트의 canonical source | 고유 추출 스트림 |
| `_GenAI_FineTuning 2.pdf` | 동일 범위의 두 번째 파일 | duplicate-content, 별도 노트 미생성 |

> [!WARNING]
> 두 파일은 해시가 다르지만 추출 페이지 본문은 동일하다. 따라서 **파일 수 2**와 **고유 강의 내용 1**을 혼동하지 않는다. 원문 철자 오류·잘린 문장·근거가 희박한 페이지는 각 담당 노트에서 별도로 표시한다.
