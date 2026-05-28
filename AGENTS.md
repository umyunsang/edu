---
aliases: []
course: uncategorized
created: '2026-04-06'
date: '2026-04-06'
semester: extracurricular
source: ''
status: seedling
tags:
- type/lecture
title: AGENTS.md
type: lecture
updated: '2026-05-05'
---





related:: [[CLAUDE|CLAUDE]], [[GEMINI|GEMINI]]

# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What This Is

An Obsidian vault containing a 3-year Computer Science & AI undergraduate curriculum archive. ~430 markdown notes organized by semester, plus certifications and extracurricular activities.

## Vault Structure

- `ComputerScience/` — All course notes, named as `학기_영문약어` (e.g., `3-1_machine-learning/`, `4-1_AIOSS/`, `elective_docker-k8s/`)
- `LGAimer/` — LG Aimers program materials
- `certifications/` — 정보처리기사, 빅데이터분석 등 자격증 준비 자료
- `image/` — Shared image assets referenced from notes
- `.obsidian/` — Vault configuration, plugins, themes

## Obsidian Conventions

- **Internal links**: Use `[[wikilink]]` syntax, not standard markdown links
- **Image embeds**: Use `![[3-2_neural-network__image.png]]` format; images are stored in `/image/`
- **Math**: LaTeX via `$...$` and `$$...$$` (Quick LaTeX + Extended MathJax plugins installed)
- **Diagrams**: Mermaid code blocks (Mermaid Tools plugin installed)
- **Properties**: YAML frontmatter when present
- **Callouts**: `> [!type]` syntax for info/warning/note blocks
- **Language**: Notes are primarily in Korean (한국어)

## Installed Plugins

**Community (active):** Better Export PDF, Obsidian Git (60s auto-commit/push), Terminal
**Community (installed):** Mermaid Tools, PDF++, Ink, LaTeX (Extended MathJax), Quick LaTeX
**Core (notable enabled):** Bases, Canvas, Daily Notes, Templates, Slides, Graph

## Git Workflow

- Auto-backup via Obsidian Git plugin: commits every 60s, pushes every 60s
- Commit message format: `vault backup: {{date}}`
- Remote: `https://github.com/umyunsang/edu.git`
- Pull-before-push strategy is active

## Working with This Vault

- When creating or editing notes, always use Obsidian Flavored Markdown (wikilinks, callouts, embeds)
- Directory naming: `학기_영문약어` (no spaces, no brackets) — e.g., `3-1_machine-learning`, `elective_LLM`
- Images should go in `/image/` and be referenced with `![[filename]]`
- PDF export is configured for A4, no margins, no title — respect these settings when formatting
- Notes may contain code blocks in Python, Java, JavaScript, C (CUDA), SQL, and shell
