---
title: Slidev 호환 계층 임시 검증
description: 프로젝트 루트 addon의 깊은 경로 상속만 검증하는 임시 문서
type: lecture
tags: []
course: compatibility-probe
semester: extracurricular
source: synthetic-contract-fixture
source_pages: 0
status: draft
aliases: []
slides: true
created: '2026-08-29'
updated: '2026-08-29'
---

# 깊은 경로 상속

> [!ABSTRACT]+ 호환 계층
> 프로젝트 루트 addon이 깊은 과목 경로에도 적용되어야 한다.

==공식 강조==와 $E = mc^2$를 함께 확인한다.

---

## Tabs

<Tabs id="deep-probe-tabs">
  <Tab label="첫째">첫 번째 패널</Tab>
  <Tab label="둘째">두 번째 패널</Tab>
</Tabs>

---

## Mermaid

```mermaid
flowchart LR
    A["공식 문법"] --> B["Slidev addon"]
    B --> C["CLI export"]
```

---

## HTML preview

```html preview h=220px
<div style="font-family:system-ui,sans-serif;padding:20px">
  <div id="cards" style="display:flex;gap:14px;flex-wrap:wrap"></div>
  <script>
    var stats = [
      ['Active users', '12,480', '+8.2% MoM', 'var(--chart-2)'],
      ['Revenue', '$48.2k', '+3.1% MoM', 'var(--chart-1)'],
      ['Churn', '2.4%', '-0.5% MoM', 'var(--chart-5)']
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
