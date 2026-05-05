<%*
const folder = tp.file.folder();
const semMatch = folder.match(/^(\d-\d)/);
const semester = semMatch ? semMatch[1] : "elective";
const course = folder.replace(/^(?:\d-\d|elective)_/, "");
const src = await tp.system.prompt("Source (저자, 제목, URL)");
-%>
---
aliases: []
course: <% course %>
created: <% tp.date.now("YYYY-MM-DD") %>
date: <% tp.date.now("YYYY-MM-DD") %>
semester: "<% semester %>"
source: "<% src %>"
status: seedling
tags:
  - type/literature
title: <% tp.file.title %>
type: literature
updated: <% tp.date.now("YYYY-MM-DD") %>
---

# <% tp.file.title %>

> Source: <% src %>

## 요약

## 핵심 인용

## 내 생각

## 연결
- 
