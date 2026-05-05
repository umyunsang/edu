---
aliases:
- 홈
- 지식 지도
course: cross-curriculum
created: 2026-05-05
date: 2026-05-05
semester: all
source: ''
status: evergreen
tags:
- type/MOC
- type/index
title: Home MOC
type: MOC
updated: 2026-05-05
---

central:: [[Home MOC]]
children:: [[Machine Learning MOC]], [[Deep Learning MOC]], [[Algorithms MOC]], [[Systems MOC]], [[Computer Vision MOC]], [[LLM & NLP MOC]], [[AI Open Source MOC]], [[Math Foundations MOC]], [[Database MOC]], [[Cloud & Containers MOC]], [[Security MOC]], [[Software Engineering MOC]], [[Certifications MOC]], [[Portfolio MOC]], [[Open Questions MOC]]

# Home MOC

> 단일 진입점. 모든 도메인 MOC가 여기서 출발한다.

## CS Core
- [[Machine Learning MOC]]
- [[Deep Learning MOC]]
- [[Algorithms MOC]]
- [[Systems MOC]]
- [[Computer Vision MOC]]
- [[LLM & NLP MOC]]
- [[AI Open Source MOC]]
- [[Database MOC]]
- [[Cloud & Containers MOC]]
- [[Security MOC]]
- [[Software Engineering MOC]]

## Foundations
- [[Math Foundations MOC]]

## Outputs
- [[Portfolio MOC]]
- [[Certifications MOC]]
- [[Open Questions MOC]]

## All MOCs (auto)
```dataview
LIST FROM #type/MOC
SORT file.name ASC
```

## Recently updated
```dataview
TABLE WITHOUT ID file.link as Note, type, status, file.mtime as updated
FROM "" WHERE file.path != this.file.path
SORT file.mtime DESC
LIMIT 15
```
