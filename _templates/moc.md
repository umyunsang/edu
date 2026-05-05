---
aliases: []
course: cross-curriculum
created: <% tp.date.now("YYYY-MM-DD") %>
date: <% tp.date.now("YYYY-MM-DD") %>
semester: "all"
source: ""
status: evergreen
tags:
  - type/MOC
title: <% tp.file.title %>
type: MOC
updated: <% tp.date.now("YYYY-MM-DD") %>
---

up:: [[Home MOC]]
central:: [[<% tp.file.title %>]]

# <% tp.file.title %>

## Foundations

## Core Topics

## Open Questions

## All notes (auto)
```dataview
TABLE status, file.mtime as updated
FROM "<% tp.file.folder() %>"
WHERE type != "MOC"
SORT file.mtime DESC
LIMIT 50
```
