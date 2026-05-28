---
aliases: []
course: uncategorized
created: '2026-04-06'
date: '2026-04-06'
kg_graph_size: 60
kg_layer_label: L4 support
kg_level: 4
kg_role: support
semester: extracurricular
source: ''
status: seedling
tags:
- type/lecture
title: AGENTS.md
type: lecture
updated: '2026-05-05'
---

graph:: [[ComputerScience/00_graph-interfaces/지식그래프 허브|지식그래프 허브]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/아카이브 운영 브리지|아카이브 운영 브리지]]
related:: [[CLAUDE|CLAUDE]]

# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What This Is

An Obsidian vault containing a 3-year Computer Science & AI undergraduate curriculum archive. ~430 markdown notes are organized by content domain, with semester history preserved in frontmatter.

## Vault Structure

- `ComputerScience/01_programming-foundations/` — coding basics, Python, data structures, coding test, Java
- `ComputerScience/00_graph-interfaces/` — real Obsidian graph nodes for stages, bridges, course modules, relationship ontology, research trends, tech stacks, ecosystems, and competencies
- `ComputerScience/02_math-theory/` — probability/statistics, discrete math, optimization math, mathematical logic
- `ComputerScience/03_ai-ml-data/` — AI, ML, neural networks, computer vision, LLM, big data, AI system design
- `ComputerScience/04_systems-infrastructure/` — Linux, computer architecture, OS, networks, distributed/CUDA/MPI, containers
- `ComputerScience/05_software-engineering/` — web, database, OSS, programming languages, AIOSS delivery workflow
- `ComputerScience/06_algorithms-graphics/` — algorithm design/analysis and computer graphics
- `ComputerScience/07_professional-humanities/` — intellectual property, creative writing, classics, degree/portfolio notes
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
- Graph relationships should prefer explicit interface nodes and wikilink fields: `domain::`, `stage::`, `module::`, `bridge::`, `schema::`, `source_model::`, `relation_type::`, `tech_stack::`, `research::`, `ecosystem::`, `competency::`
- Directory naming: field interface first, canonical course folder second — e.g., `03_ai-ml-data/machine-learning`, `04_systems-infrastructure/operating-systems`
- Images should go in `/image/` and be referenced with `![[filename]]`
- PDF export is configured for A4, no margins, no title — respect these settings when formatting
- Notes may contain code blocks in Python, Java, JavaScript, C (CUDA), SQL, and shell
