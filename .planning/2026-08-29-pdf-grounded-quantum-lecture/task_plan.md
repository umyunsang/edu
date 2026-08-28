# Task Plan: original-visual-free edu rewrite

## Goal

Rebuild every lecture note in each edu course from its source PDFs and extracted text using only the official OpenKnowledge lecture-note template, native components, and palette starters.

All note content previously written by any delegated lane is rejected and may be inspected only to locate defects, never as an authoring anchor.

## Hard boundaries

- Never inspect deleted notes through Git history.
- Do not embed rendered source slides or any original-PDF-derived image in a note. Keep source PDFs and rendered assets private as evidence material, but do not delete those underlying source assets without separate authorization.
- Do not invent a visual component or hand-roll an SVG/CSS/JS pattern. Use only the official palette starters where a visual adds explanatory value, preserving their structure and theme tokens.
- Do not add Obsidian wikilinks, Dataview, or hierarchical tags.
- Preserve source errors and mark them as source errors when present.
- Do not commit or push until the user requests it; never stage unrelated or dot-directory changes.
- Every rewritten study note must carry `slides: true` in frontmatter so OpenKnowledge can open it as a Slidev job.
- Render validation is performed only with the Slidev CLI build path. Do not use browser, preview URL, screenshot, DOM, or rsvg checks.

## Phases

1. [in_progress] Reject every prior course output and prior gate result, including the thin and duplicated template-driven notes. Rebuild in place from source PDFs/extracts with the official OpenKnowledge skill, lecture-note template, palette starters, and project Mermaid skill as the component contract, while deriving each document's substance and section order from its own PDF segment.
2. [pending] Per course, read its sources and the official template before writing; apply only official markdown-native components, Tabs, and unmodified palette-starter structures populated with source-grounded data.
3. [pending] Keep the copyright override: omit source slide images and text-only page citations despite the template's evidence-image section.
4. [pending] Per course, validate through OpenKnowledge write/edit warnings, lint, audit, Mermaid write warnings, and Slidev CLI build. Do not open browsers, use preview URLs, take screenshots, or perform DOM/rsvg verification.
5. [pending] Report course component totals only after the official-template rewrite; no prior totals are accepted.
6. [pending] Leave `java-programming`, `coding-test`, `degree-portfolio`, `container-orchestration`, and `mathematical-logic` without reconstructed notes because their sources are empty.
7. [pending] After all rewrites, add graph relationship types in the separately authorized post-rewrite pass.

## Quality floor per lecture note

- Cover every substantive heading, example, procedure, source quantity, caveat, and error in the assigned PDF segment, or explicitly record why a sparse fragment cannot support prose.
- At least 10 meaningful components may be used where the source supports them, but component count never substitutes for explanation depth.
- Zero direct source-slide image embeds.
- Use an official palette visual starter only when its labels and values are directly grounded in the assigned source segment; otherwise omit it. Never invent numeric data for visual density.
- Component count is not a proxy for source coverage. Each note must preserve the distinct concepts, examples, procedures, equations, caveats, and narrative depth of its assigned PDF segment, with no copied generic body between sibling notes.
- Before a course gate, compare sibling notes for duplicated substantive prose, identical generic tables/questions, repeated visual data, and shared source facts. Any unexplained overlap rejects the entire course batch.
- Write for scanning rather than essay reading: short keyword-led bullets, numbered procedures, and compact lead sentences. Do not use long narrative paragraphs or turn the entire note into one bullet list.
- Choose official components by information shape: Mermaid for flow/relationships, Tabs and tables for comparison, stat-cards/charts for source quantities, interactive-control for source-backed parameter effects, details for optional depth, and callouts for errors/risks. Diversity must improve comprehension, not satisfy a count.
- Keep the document substantial by distributing source-backed information across compact bullets and multiple appropriate official visuals. Do not reduce source coverage to make the note shorter, and do not inflate length with narrative prose; each visual must carry information that prose does not merely repeat.
- Treat Slidev compilation as the render gate: a note is not render-valid merely because OpenKnowledge lint/audit is clean.

## 2026-08-29 Project-template and Slidev reset

- The project root had no `.ok/templates/`, so agents had no project-wide authoring contract. `ComputerScience/.ok/templates/` existed but its same-name templates overrode any future root template and still required slide images/PDF links, contradicting the current copyright rule.
- Authoring stays paused while root project templates are created and every same-name folder override is aligned or removed.
- All lecture/practice/index templates must include `slides: true` where the document is intended to open as a deck.
- OpenKnowledge's slides plugin is a user-scoped desktop preference and is not agent-settable through project config. Every eligible note still carries `slides: true`; render checking uses the installed Slidev CLI only.
- Slidev's exporter dependency is installed at `@slidev/cli/node_modules/playwright-chromium` (v1.61.0), with Chromium and headless-shell revision 1228 available in the Playwright cache.
- Current root authority is exactly three templates: `lecture-note`, `course-index`, and `practice-note`; no same-name `ComputerScience` override remains.
- Template probes pass OpenKnowledge lint/audit and raw Slidev build/export, but this is not a visual-fidelity PASS for OpenKnowledge-only components.
- Official OpenKnowledge 0.64.2 provides no Slidev adapter. `html preview` and `Tabs/Tab` are incompatible with raw Slidev 52.19.1, while Mermaid, KaTeX, and the five base GFM alerts are the confirmed common subset.
- Course authoring remains paused at this boundary: either Slidev is accepted as a compile/export-only gate for OpenKnowledge-native visuals, or a separately authorized compatibility layer is required. Do not invent that layer under the current official-components-only contract.
