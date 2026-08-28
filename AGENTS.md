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
status: stable
tags:
  - agent-guidance
  - openknowledge
title: AGENTS.md
type: lecture
updated: '2026-08-29'
---

This file provides guidance to AI agents (Claude Code, Codex, OpenCode, Pi) working in this repository.

## What This Is

An Obsidian vault containing a full undergraduate Computer Science & AI curriculum archive.
Markdown notes are organized by content domain, with semester history preserved in frontmatter.

The vault is **dual-readable by design**: humans read it in Obsidian, agents read it through
frontmatter, folder guides, the source ledger, and the OpenKnowledge MCP server.

> **Read `ComputerScience/.ok/frontmatter.yml` and the applicable inherited template in `.ok/templates/` first.**
> Together they are the current authority for source scope, visualization rules, and the verification pipeline.

## Layer Model

| Layer | Where | Who reads it |
| :-- | :-- | :-- |
| Source of record | `<course>/sources/*.pdf` (Git LFS) | extraction and evidence workflow only |
| Study notes | `<course>/notes/NN. 제목.md` | humans in OpenKnowledge and Obsidian |
| Graph structure | `ComputerScience/00_graph-interfaces/` | Obsidian Graph View |
| Agent metadata | `.ok/frontmatter.yml`, `wiki/meta/ledgers/`, `.agents/skills/` | LLMs |

## Tooling

| Tool | Role here | Key paths |
| :-- | :-- | :-- |
| **OpenKnowledge** (`ok`) | agent navigation, search, lint, MCP server `open-knowledge` | `.ok/config.yml`, `.okignore`, `.mcp.json`, per-folder `.ok/frontmatter.yml` |
| **claude-obsidian** | provenance and vault linting | `.claude-obsidian.json`, `wiki/`, `inbox/`, `.raw/` |

`wiki/` holds **ledgers and operation logs only** — course notes stay in their course folders.
Do not migrate notes into `wiki/sources/` or `external-sources/`.

```bash
npx -y @inkeep/open-knowledge@latest start          # MCP service; do not auto-open a browser
npx -y @inkeep/open-knowledge@latest lint "ComputerScience"
```

## Repository Scripts

| Script | Purpose |
| :-- | :-- |
| `scripts/pdf_lecture_extract.py` | lecture PDF → extraction bundle (`text.md`, `meta.json`, page PNGs) |
| `scripts/register_pdf_sources.py` | register PDFs in `wiki/meta/ledgers/source-ledger.json` (reads sha256 from LFS pointers, no full download needed) |
| `scripts/generate_folder_guides.py` | (re)generate per-folder `.ok/frontmatter.yml` agent guides — idempotent |
| `scripts/check_mermaid.mjs` | validate every ` ```mermaid ` block with the real Mermaid parser |

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

## Authoring Conventions

- **Project templates**: Start every lecture, course index, and practice note from the inherited root templates in `.ok/templates/`.
- **Links**: Use standard Markdown relative links. Wikilinks and wiki embeds are prohibited in rewritten course notes.
- **Source privacy**: Do not embed source-slide images, PDFs, or visible page citations in public study notes.
- **Math**: Use `$...$` or `$$...$$` only when the source actually contains the equation.
- **Diagrams**: Use valid Mermaid code blocks for relationships, sequences, and workflows.
- **Visual findings**: Re-query the OpenKnowledge palette while authoring. Use the official chart, stat-cards, custom-svg, or interactive-control starter whenever the source supplies the required data shape.
- **Visual roles**: A lecture note needs at least two distinct information-bearing visual roles; a course index needs learning-path, document-map, and coverage roles; a practice note needs workflow and measured-result roles. HTML preview has no unconditional quota.
- **Density baseline**: Ten meaningful components per lecture note is the review floor. A lower count requires a source-sparsity explanation; duplicate or decorative components never satisfy the floor.
- **Template flexibility**: The root template is a starting skeleton. Add, delete, or reorder sections to match the source instead of preserving a common two-section shape.
- **Components**: Preserve official starter structure, theme tokens, and control flow. Replace only source-backed labels and data literals.
- **Delivery**: Combine compact conclusions, selective bullets, and visuals. Do not write an essay or reduce the whole document to one bullet list.
- **Language**: Notes are primarily in Korean (한국어).

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

- Read every in-scope Markdown file and perform every Markdown mutation through OpenKnowledge MCP.
- Reject all prior lane note bodies and gate results. For lecture notes and course indexes, derive structure and substance only from current source PDFs and extracts; use executed notebooks only for practice notes. Existing notes may be inspected only to locate defects, and deleted notes must never be read through Git history.
- Read the target folder and inherited template menu before writing.
- Call the OpenKnowledge palette while drafting every source-backed note. Quantitative or interactive findings must not remain prose-only when an official starter gate is satisfied.
- Use `chart` only for 2–5 comparable, non-negative values on one unit and axis; at least one value must be positive.
- Use `stat-cards` for independent source metrics, `custom-svg` for an exact part-to-whole ratio, and `interactive-control` only for a source-defined minimum, maximum, step, and default.
- Never invent source values, interpolate missing values, fetch external data, or leave palette example values in a note.
- Keep official HTML nodes, IDs, inline CSS, theme tokens, and JavaScript control flow intact; change only source-backed display text and data literals.
- Preserve source errors in a warning instead of silently correcting them.
- Do not use wikilinks, Dataview fields, hierarchical tags, original-PDF images, PDF links, or visible `p.N` citations in rewritten notes.
- Every lecture, course-index, and practice note keeps `slides: true` and uses `---` separators between logical slide units.
- Run OpenKnowledge lint with fixes immediately after template instantiation, then a scoped audit whose `ran` includes `markdownlint`, `frontmatter`, `okf`, and `links`.
- Use the pinned project-local Slidev toolchain and `slidev-addon-openknowledge`. Build/export success proves CLI compilation, component registration, static export, and the explicit test assertions only; viewport fit, visual fidelity, and interaction remain unverified. Never label them PASS without separately authorized visual testing.
- Directory naming remains field interface first, canonical course folder second — e.g., `03_ai-ml-data/machine-learning`, `04_systems-infrastructure/operating-systems`.
- Relationship typing is deferred until the source-backed rewrite is complete.

## Verification Before Committing

```bash
npm run test:slidev-compat
node scripts/check_mermaid.mjs --dir "ComputerScience"
npx -y @inkeep/open-knowledge@latest lint "ComputerScience"
./node_modules/.bin/slidev build "<note>.md" --out "/tmp/<note>-slidev-build"
./node_modules/.bin/slidev export "<note>.md" --output "/tmp/<note>-slidev.pdf"
```

Also run the OpenKnowledge MCP audit on the exact course path and confirm that `ran` includes `markdownlint`, `frontmatter`, `okf`, and `links`.

A gate fails when OpenKnowledge reports lint, link, OKF, frontmatter, or Mermaid problems; when a template placeholder or palette example value remains; when sibling notes reuse substantive prose, table rows, Mermaid edges, or visual payloads; when Slidev reports an unresolved component; or when build/export exits non-zero. A successful Slidev build/export is not viewport or component-fidelity evidence. Browser, preview URL, DOM, screenshot, and rsvg checks are not part of this workflow.
