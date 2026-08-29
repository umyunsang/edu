---
title: 인공지능
description: 퍼셉트론에서 CNN 응용까지 원본 강의 순서로 연결한 강의 색인.
course: artificial-intelligence
type: lecture-index
status: stable
tags:
  - course
  - artificial-intelligence
slides: true
---

## 인공지능 강의 흐름

원본 PDF의 진행 순서를 따라 **퍼셉트론 → MLP → 최적화·일반화 → CNN → 응용·평가**로 읽습니다. 같은 PDF를 두 관점으로 나눈 노트는 번호를 유지해 강의 중 질문과 복습 위치를 빠르게 찾을 수 있습니다.

## 1. 표현의 출발점

- [01. Perceptron 이론 - 신경망 구성](./notes/01.%20Perceptron%20이론%20-%20신경망%20구성.md)
- [02. Perceptron 이론 - 활성화와 최적화](./notes/02.%20Perceptron%20이론%20-%20활성화와%20최적화.md)
- [03. Perceptron 논리 게이트 실습](./notes/03.%20Perceptron%20논리%20게이트%20실습.md)
- [04. MLP 이론 - 다층 표현](./notes/04.%20MLP%20이론%20-%20다층%20표현.md)
- [05. MLP 실습 - 모델 구성](./notes/05.%20MLP%20실습%20-%20모델%20구성.md)
- [06. MLP 실습 - 학습과 평가](./notes/06.%20MLP%20실습%20-%20학습과%20평가.md)

## 2. 학습 규칙과 안정화

- [07. Optimization 이론 - 손실과 경사](./notes/07.%20Optimization%20이론%20-%20손실과%20경사.md)
- [08. Optimizer - 모멘텀](./notes/08.%20Optimizer%20-%20모멘텀.md)
- [09. Optimizer - 적응적 학습률](./notes/09.%20Optimizer%20-%20적응적%20학습률.md)
- [10. Overfitting - 일반화 진단](./notes/10.%20Overfitting%20-%20일반화%20진단.md)
- [11. Overfitting - 규제와 조기 종료](./notes/11.%20Overfitting%20-%20규제와%20조기%20종료.md)
- [12. Backpropagation의 연쇄 법칙](./notes/12.%20Backpropagation의%20연쇄%20법칙.md)
- [13. Backpropagation의 가중치 갱신](./notes/13.%20Backpropagation의%20가중치%20갱신.md)
- [14. Vanishing Gradient Effect](./notes/14.%20Vanishing%20Gradient%20Effect.md)
- [15. Vanishing Gradient 완화](./notes/15.%20Vanishing%20Gradient%20완화.md)

## 3. CNN 구조와 설계

- [16. CNN의 합성곱 원리](./notes/16.%20CNN의%20합성곱%20원리.md)
- [17. CNN의 공간 크기와 채널](./notes/17.%20CNN의%20공간%20크기와%20채널.md)
- [18. CNN 주요 설계 모듈 이론](./notes/18.%20CNN%20주요%20설계%20모듈%20이론.md)
- [19. CNN 설계 모듈 실습 - Conv2d 매개변수](./notes/19.%20CNN%20설계%20모듈%20실습%20-%20Conv2d%20매개변수.md)
- [20. CNN 설계 모듈 실습 - 연결 구조](./notes/20.%20CNN%20설계%20모듈%20실습%20-%20연결%20구조.md)
- [21. CNN 분류 실습](./notes/21.%20CNN%20분류%20실습.md)
- [22. CNN Backpropagation - AlexNet 구조](./notes/22.%20CNN%20Backpropagation%20-%20AlexNet%20구조.md)
- [23. CNN Backpropagation - AlexNet 정규화](./notes/23.%20CNN%20Backpropagation%20-%20AlexNet%20정규화.md)
- [24. VGGNet 실습](./notes/24.%20VGGNet%20실습.md)
- [25. CIFAR-10 분류 실습](./notes/25.%20CIFAR-10%20분류%20실습.md)

## 4. 응용과 평가

- [26. AI 아바타 만들기 실습](./notes/26.%20AI%20아바타%20만들기%20실습.md)
- [27. 중간고사 - MLP 기반 CIFAR-10 분류](./notes/27.%20중간고사%20-%20MLP%20기반%20CIFAR-10%20분류.md)

## 5. 텍스트 추출 한계 기록

- [28. 필기 과제 자료의 텍스트 추출 한계](./notes/28.%20필기%20과제%20자료의%20텍스트%20추출%20한계.md)
- [29. 필기 과제 2 텍스트 추출 한계](./notes/29.%20필기%20과제%202%20텍스트%20추출%20한계.md)

> [!note] 원본 범위
> `sources/`의 29개 PDF를 확인했습니다. 동일 내용의 “2” 사본은 같은 주제 노트에 함께 반영했고, 텍스트 추출이 빈약한 필기 자료는 한계를 별도 기록했습니다.

## 읽는 순서 인터랙티브

```html preview
<div style="font-family:system-ui,sans-serif;padding:20px;color:var(--foreground)">
  <label for="ai-course-step" style="font-size:14px;font-weight:600">강의 구간</label>
  <div id="ai-course-out" style="font-size:26px;font-weight:700;color:var(--chart-1);margin:6px 0">표현의 출발점</div>
  <input id="ai-course-step" type="range" min="1" max="5" step="1" value="1" style="width:100%;accent-color:var(--primary)" />
  <p style="font-size:13px;color:var(--muted-foreground)">슬라이더를 움직여 원본 강의 흐름의 다섯 구간을 확인합니다.</p>
  <script>
    var aiCourseStep = document.getElementById('ai-course-step');
    var aiCourseOut = document.getElementById('ai-course-out');
    var aiCourseLabels = ['표현의 출발점', '학습 규칙과 안정화', 'CNN 구조와 설계', '응용과 평가', '추출 한계 기록'];
    aiCourseStep.addEventListener('input', function () {
      aiCourseOut.textContent = aiCourseLabels[Number(aiCourseStep.value) - 1];
    });
  </script>
</div>
```
