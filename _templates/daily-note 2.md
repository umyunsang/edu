---
aliases: []
created: <% tp.date.now("YYYY-MM-DD") %>
date: <% tp.date.now("YYYY-MM-DD") %>
status: seedling
tags:
  - type/index
title: <% tp.date.now("YYYY-MM-DD") %>
type: index
updated: <% tp.date.now("YYYY-MM-DD") %>
---

# <% tp.date.now("YYYY-MM-DD dddd") %>

## 오늘 해야 할 일
- [ ] 

## 강의
- 

## 캡처

## 시험 큐
```dataview
LIST FROM #meta/exam
SORT file.mtime DESC
LIMIT 10
```
