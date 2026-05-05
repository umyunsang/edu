<%*
const folder = tp.file.folder();
const semMatch = folder.match(/^(\d-\d)/);
const semester = semMatch ? semMatch[1] : (folder.startsWith("elective") ? "elective" : "extracurricular");
const course = folder.replace(/^(?:\d-\d|elective)_/, "");
-%>
---
aliases: []
course: <% course %>
created: <% tp.date.now("YYYY-MM-DD") %>
date: <% tp.date.now("YYYY-MM-DD") %>
semester: "<% semester %>"
source: ""
status: seedling
tags:
  - type/lecture
title: <% tp.file.title %>
type: lecture
updated: <% tp.date.now("YYYY-MM-DD") %>
---

# <% tp.file.title %>

## 핵심 개념

## 정리

## 질문 / TODO
- 

## 관련
- 
