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

This file provides guidance to AI agents (Claude Code, Codex, OpenCode, Pi) working in this repository.

## What This Is

An Obsidian vault containing a full undergraduate Computer Science & AI curriculum archive.
Markdown notes are organized by content domain, with semester history preserved in frontmatter.

The vault is **dual-readable by design**: humans read it in Obsidian, agents read it through
frontmatter, folder guides, the source ledger, and the OpenKnowledge MCP server.

> **Read `docs/knowledge-schema.md` first.** It is the authoritative spec for frontmatter,
> visualization rules, folder guides, and the verification pipeline. This file is the short version.

## Layer Model

| Layer | Where | Who reads it |
|:--|:--|:--|
| Source of record | `<course>/pdf/*.pdf` (Git LFS) | extraction script only |
| Study notes | `<course>/NN. 제목.md` | humans in Obsidian |
| Graph structure | `ComputerScience/00_graph-interfaces/` | Obsidian Graph View |
| Agent metadata | `.ok/frontmatter.yml`, `wiki/meta/ledgers/`, `.agents/skills/` | LLMs |

## Tooling

| Tool | Role here | Key paths |
|:--|:--|:--|
| **OpenKnowledge** (`ok`) | agent navigation, search, lint, MCP server `open-knowledge` | `.ok/config.yml`, `.okignore`, `.mcp.json`, per-folder `.ok/frontmatter.yml` |
| **claude-obsidian** | provenance and vault linting | `.claude-obsidian.json`, `wiki/`, `inbox/`, `.raw/` |

`wiki/` holds **ledgers and operation logs only** — course notes stay in their course folders.
Do not migrate notes into `wiki/sources/` or `external-sources/`.

```bash
npx -y @inkeep/open-knowledge@latest start --open   # web editor + MCP
npx -y @inkeep/open-knowledge@latest lint "ComputerScience"
```

## Repository Scripts

| Script | Purpose |
|:--|:--|
| `scripts/pdf_lecture_extract.py` | lecture PDF → extraction bundle (`text.md`, `meta.json`, page PNGs) |
| `scripts/register_pdf_sources.py` | register PDFs in `wiki/meta/ledgers/source-ledger.json` (reads sha256 from LFS pointers, no full download needed) |
| `scripts/generate_folder_guides.py` | (re)generate per-folder `.ok/frontmatter.yml` agent guides — idempotent |
| `scripts/check_mermaid.mjs` | validate every ` ```mermaid ` block with the real Mermaid parser |

## Writing a Lecture Note From a PDF

Follow `.agents/skills/lecture-pdf-to-note/SKILL.md`. Summary:

1. `git lfs pull --include="<course>/pdf/*"`
2. `python3 scripts/pdf_lecture_extract.py "<course>/pdf/01_Foo.pdf" --render none`
3. Write `<course>/NN. 제목.md` from the bundle — Korean, restructured, never copied verbatim
4. Mix **at least 4 kinds of visualization** (tables, ≥3 distinct Mermaid types, callouts, LaTeX, PDF page embeds)
5. `node scripts/check_mermaid.mjs "<note>"` until zero failures
6. `python3 scripts/register_pdf_sources.py`

Never invent content that is not in the slides. Never create `![[image.png]]` embeds for images
that do not exist — use `![[pdf/file.pdf#page=N]]` with a text explanation instead.

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
- Keep `related::` to **8 links maximum**. Some legacy notes have dozens injected by an old script;
  that is a defect, not a pattern to copy — it wrecks Graph View and wastes LLM context
- Directory naming: field interface first, canonical course folder second — e.g., `03_ai-ml-data/machine-learning`, `04_systems-infrastructure/operating-systems`
- Images should go in `/image/` and be referenced with `![[filename]]`
- PDF export is configured for A4, no margins, no title — respect these settings when formatting
- Notes may contain code blocks in Python, Java, JavaScript, C (CUDA), SQL, and shell
- Prefer Mermaid over ` ```html preview ` embeds: the latter renders in OpenKnowledge but shows as
  a raw code block in Obsidian, and Obsidian is the primary human interface

## Verification Before Committing

```bash
node scripts/check_mermaid.mjs --dir "ComputerScience"
npx -y @inkeep/open-knowledge@latest lint "ComputerScience"
python3 scripts/register_pdf_sources.py
```
