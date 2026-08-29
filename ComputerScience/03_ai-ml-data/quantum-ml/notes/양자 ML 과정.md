---
title: "양자 ML 과정 - 160시간 양자 머신러닝 집중 과정 종합 정리 (Quantum ML 160H Curriculum)"
description: "고전 표현력 한계 극복부터 큐비트 대수, 양자 임베딩, 아다마르/벨 상태 전이, 범용 회로 합성, VQC 및 Barren Plateaus 극복까지 160시간 양자 머신러닝 전 과정을 총괄 정리한다."
type: lecture
tags:
  - lecture
  - quantum-ml
  - 160h-curriculum
  - summary
course: quantum-ml
semester: extracurricular
source: "ICTIS_AI_Quantum_Computing_160H_Certificate.pdf"
source_pages: 50
status: draft
review_state: active
authority: primary
slides: true
created: "2026-08-29"
updated: "2026-08-29"
---

본 문서는 **160시간 ICTIS 인공지능 & 양자 컴퓨팅(Quantum Machine Learning) 집중 교육 과정**의 전 단원 핵심 수학적 이론과 실습 체계를 총괄 정리한다.

---

## 1. 160시간 양자 머신러닝 4대 핵심 역량 마일스톤

```html preview
<div style="font-family:system-ui,-apple-system,sans-serif;padding:16px;background:var(--card);color:var(--card-foreground);border:1px solid var(--border);border-radius:var(--radius)">
  <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(130px, 1fr));gap:8px">
    <div style="padding:10px;background:var(--background);border:1px solid var(--border);border-radius:var(--radius)">
      <div style="font-size:11px;font-weight:800;color:var(--chart-1)">1. 이론 및 대수 기반</div>
      <div style="font-size:10px;color:var(--muted-foreground);margin-top:4px">
        • $2^n$ 복소 힐베르트 공간<br/>
        • 디랙 브라-켓 & 텐서곱 연산<br/>
        • 블로흐 구 $(\theta, \phi)$ 극좌표
      </div>
    </div>
    <div style="padding:10px;background:var(--background);border:1px solid var(--border);border-radius:var(--radius)">
      <div style="font-size:11px;font-weight:800;color:var(--chart-2)">2. 양자 데이터 임베딩</div>
      <div style="font-size:10px;color:var(--muted-foreground);margin-top:4px">
        • 각도/진폭/IQP 특징 매핑<br/>
        • 양자 커널 $K(\mathbf{x}_i, \mathbf{x}_j)$<br/>
        • QSVM 고차원 분리 초평면
      </div>
    </div>
    <div style="padding:10px;background:var(--background);border:1px solid var(--border);border-radius:var(--radius)">
      <div style="font-size:11px;font-weight:800;color:var(--chart-3)">3. 회로 및 얽힘 공학</div>
      <div style="font-size:10px;color:var(--muted-foreground);margin-top:4px">
        • 아다마르(H) & CNOT 얽힘<br/>
        • 4대 벨 상태 $|\Phi^\pm\rangle, |\Psi^\pm\rangle$<br/>
        • 범용 게이트 셋 $\{H, S, T, \text{CNOT}\}$
      </div>
    </div>
    <div style="padding:10px;background:var(--background);border:1px solid var(--border);border-radius:var(--radius)">
      <div style="font-size:11px;font-weight:800;color:var(--primary)">4. VQC & 실전 최적화</div>
      <div style="font-size:10px;color:var(--muted-foreground);margin-top:4px">
        • PQC 가중치 안사츠 설계<br/>
        • 파라미터 시프트 룰 역전파<br/>
        • Barren Plateaus 완화 기법
      </div>
    </div>
  </div>
</div>
```

---

## 2. 인터랙티브 QML 4대 역량 자가 점검기

각 역량 영역을 클릭하여 본 과정에서 다룬 핵심 수식과 알고리즘 구현 숙련도를 확인한다.

```html preview
<div style="font-family:system-ui,-apple-system,sans-serif;padding:20px;background:var(--card);color:var(--card-foreground);border:1px solid var(--border);border-radius:var(--radius);max-width:100%">
  <!-- 버튼 바 -->
  <div style="display:flex;gap:6px;align-items:center;margin-bottom:14px;flex-wrap:wrap">
    <button id="qcur-btn-1" style="padding:6px 10px;background:var(--primary);color:#fff;border:none;border-radius:var(--radius);font-size:11px;font-weight:700;cursor:pointer">1. 힐베르트 대수</button>
    <button id="qcur-btn-2" style="padding:6px 10px;background:transparent;border:1px solid var(--border);color:var(--foreground);border-radius:var(--radius);font-size:11px;font-weight:700;cursor:pointer">2. 양자 특징 사상</button>
    <button id="qcur-btn-3" style="padding:6px 10px;background:transparent;border:1px solid var(--border);color:var(--foreground);border-radius:var(--radius);font-size:11px;font-weight:700;cursor:pointer">3. 게이트 & 회로</button>
    <button id="qcur-btn-4" style="padding:6px 10px;background:transparent;border:1px solid var(--border);color:var(--foreground);border-radius:var(--radius);font-size:11px;font-weight:700;cursor:pointer">4. VQC & 최적화</button>
  </div>

  <!-- 세부 명세 박스 -->
  <div style="background:var(--background);border:2px dashed var(--border);border-radius:var(--radius);padding:14px">
    <div id="qcur-title" style="font-size:14px;font-weight:800;color:var(--primary);margin-bottom:6px">1. 힐베르트 대수 & 큐비트 기초</div>
    <div id="qcur-desc" style="font-size:12px;color:var(--foreground);line-height:1.6">
      2준위 양자계, 블로흐 구 위도/경도 각도, 복소 내적과 Borns Rule 측정 확률.
    </div>
  </div>
</div>

<script>
(function() {
  var data = [
    { title: '1. 힐베르트 대수 & 큐비트 기초 (Notes 01-03)', desc: '2준위 양자계, 블로흐 구 위도/경도 각도, 복소 내적과 Borns Rule 측정 확률.' },
    { title: '2. 양자 특성 공간과 데이터 임베딩 (Notes 04-05)', desc: '각도/진폭/IQP 특징 맵, 양자 커널 추정기, 하이브리드 양자-고전 신경망.' },
    { title: '3. 양자 게이트와 상태 전이 분석 (Notes 06-08)', desc: '아다마르 중첩, 4대 벨 상태 얽힘 생성, 단일 큐비트 3축 회전과 범용 게이트 셋.' },
    { title: '4. 양자 회로 설계 및 VQC 최적화 (Notes 09-10)', desc: '회로 깊이/충실도, 파라미터 시프트 룰 역전파, Barren Plateaus 극복 전략.' }
  ];

  var t = document.getElementById('qcur-title');
  var d = document.getElementById('qcur-desc');

  function render(idx) {
    for (var i = 0; i < 4; i++) {
      var b = document.getElementById('qcur-btn-' + (i+1));
      if (i === idx) {
        b.style.background = 'var(--primary)'; b.style.color = '#fff';
      } else {
        b.style.background = 'transparent'; b.style.color = 'var(--foreground)';
      }
    }
    t.textContent = data[idx].title;
    d.textContent = data[idx].desc;
  }

  for (var k = 0; k < 4; k++) {
    (function(idx) {
      document.getElementById('qcur-btn-' + (idx+1)).addEventListener('click', function() { render(idx); });
    })(k);
  }
})();
</script>
```
