---
title: OpenAI API 128. Fine-Tuning
description: Fine-Tuning의 목적, 데이터 형식, 작업 단계, 상태 확인, 한계를 프롬프트 적응과 비교해 정리한다.
type: lecture
tags:
  - fine-tuning
  - llm
  - training-data
course: large-language-models
semester: extracurricular
status: draft
created: '2026-08-29'
updated: '2026-08-29'
---

> [!abstract] 한 줄 요약
> Fine-Tuning은 좋은 예시를 많이 모으는 일만이 아니라, 안정된 작업 정의를 데이터 형식·검증·평가까지 연결하는 과정이다.

강의는 Fine-Tuning을 특정 업무의 입력과 출력 패턴에 모델을 맞추는 방법으로 제시한다. 여기서 핵심 산출물은 학습 호출 코드보다 **일관된 학습 데이터와 실패를 판별할 평가 기준**이다.

```mermaid
flowchart LR
  A["작업 정의"] --> B["예시 데이터 수집"]
  B --> C["형식·품질 검증"]
  C --> D["학습 작업 생성"]
  D --> E["상태와 이벤트 확인"]
  E --> F["평가와 비교"]
  F --> G["배포 또는 개선"]
```

## 단계별 산출물

| 단계 | 산출물 | 검토 질문 |
| :-- | :-- | :-- |
| 작업 정의 | 입력·출력 계약 | 무엇을 잘하게 만들 것인가 |
| 데이터 준비 | 일관된 학습 예시 | 예시가 같은 기준을 따르는가 |
| 형식 검증 | 유효한 데이터 파일 | 누락·구문·역할 오류가 없는가 |
| 학습 실행 | 작업 식별자와 상태 | 실패·대기·완료를 구분했는가 |
| 평가 | 기준선과 비교 결과 | 프롬프트만 쓴 경우보다 나은가 |
| 운영 | 모니터링 계획 | 드리프트와 비용을 어떻게 볼 것인가 |

> [!important] 데이터가 모델 행동을 정의한다
> 애매하거나 서로 충돌하는 예시는 모델의 불안정한 출력으로 이어질 수 있다. 학습 예시의 정답 형식과 제외 규칙을 먼저 합의해야 한다.

```html preview h=180
<div style="font-family:system-ui,sans-serif;padding:18px;color:var(--foreground)">
  <div style="font-weight:700;margin-bottom:12px">Fine-Tuning의 병목은 코드보다 검증 가능한 데이터다</div>
  <div style="display:flex;gap:9px;align-items:center">
    <div style="flex:1;padding:12px;background:var(--card);border:1px solid var(--border);border-radius:var(--radius)">예시<br><span style="font-size:12px;color:var(--muted-foreground)">입력·출력 쌍</span></div>
    <span style="color:var(--muted-foreground)">→</span>
    <div style="flex:1;padding:12px;background:var(--card);border:1px solid var(--border);border-radius:var(--radius)">검증<br><span style="font-size:12px;color:var(--muted-foreground)">형식·일관성</span></div>
    <span style="color:var(--muted-foreground)">→</span>
    <div style="flex:1;padding:12px;background:var(--card);border:1px solid var(--border);border-radius:var(--radius);color:var(--chart-2)">평가<br><span style="font-size:12px;color:var(--muted-foreground)">기준선 비교</span></div>
  </div>
</div>
```

## 적용 판단

<Tabs>
<Tab label="적합한 경우">

입력·출력 형식이 반복되고, 업무 정의가 안정적이며, 품질 기준을 담은 예시를 계속 확보할 수 있을 때 후보가 된다.

</Tab>
<Tab label="먼저 할 일">

프롬프트·예시·검색 기반 문맥으로 해결되는지 먼저 비교한다. 데이터가 적거나 요구가 자주 바뀌면 학습 조정이 오히려 부담이 될 수 있다.

</Tab>
<Tab label="평가">

훈련 예시만 잘 맞는지 보지 말고, 보지 않은 입력에서 형식·정확성·안전 조건을 지키는지 확인한다.

</Tab>
</Tabs>

<details>
<summary>학습 데이터 검사 목록</summary>

역할과 메시지 구조가 일관적인지, 필수 필드가 빠지지 않았는지, 같은 입력에 상충하는 답이 없는지, 민감 정보가 포함되지 않았는지, 평가용 사례가 따로 있는지를 점검한다.

</details>

> [!tip] 기준선을 남긴다
> Fine-Tuning 전후를 비교하려면 동일한 테스트 입력, 프롬프트 기준선, 평가 항목을 기록해야 한다. “좋아 보인다”만으로는 조정의 효과를 판단하기 어렵다.

## 정리

- Fine-Tuning은 ==반복되는 작업 계약을 학습 데이터로 표현==하는 방식이다.
- 성공은 학습 작업 완료가 아니라 보지 않은 입력에서의 기준선 대비 개선으로 판단한다.
- 데이터 형식 오류와 작업 상태 관리는 학습 품질과 별개로 반드시 확인해야 한다.

> [!warning] 소스 오류 처리
> 강의는 데이터 파일 형식 오류와 작업 실행 오류를 따로 다룬다. 오류 메시지를 무시하고 재시도하기보다, 데이터 검증 실패와 원격 작업 실패를 분리해 기록한다.
