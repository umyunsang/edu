<%*
const folder = tp.file.folder();
const semMatch = folder.match(/^(\d-\d)/);
const semester = semMatch ? semMatch[1] : "elective";
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
  - type/project
title: <% tp.file.title %>
type: project
updated: <% tp.date.now("YYYY-MM-DD") %>
---

# <% tp.file.title %>

## 목표

## 요구사항

## 진행

## 결과

## 회고
