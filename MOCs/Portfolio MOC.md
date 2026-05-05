---
aliases:
- 포트폴리오
course: cross-curriculum
created: 2026-05-05
date: 2026-05-05
semester: all
source: ''
status: evergreen
tags:
- type/MOC
- meta/portfolio
title: Portfolio MOC
type: MOC
updated: 2026-05-05
---

up:: [[Home MOC]]
central:: [[Portfolio MOC]]

# Portfolio MOC

> evergreen 노트와 portfolio 후보 자동 수집

## Featured Projects
- 

## Evergreen Concepts
- 

## Writing Pipeline (status:budding → evergreen)

## All portfolio candidates (auto)
```dataview
TABLE status, type, file.mtime as updated
FROM #meta/portfolio OR #type/permanent
SORT status DESC, file.mtime DESC
LIMIT 50
```
