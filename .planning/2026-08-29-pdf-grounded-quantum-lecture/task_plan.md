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

## Phases

1. [in_progress] Reject every prior course output and prior gate result, including the thin and duplicated template-driven notes. Rebuild in place from source PDFs/extracts with the official OpenKnowledge skill, lecture-note template, palette starters, and project Mermaid skill as the component contract, while deriving each document's substance and section order from its own PDF segment.
2. [pending] Per course, read its sources and the official template before writing; apply only official markdown-native components, Tabs, and unmodified palette-starter structures populated with source-grounded data.
3. [pending] Keep the copyright override: omit source slide images and text-only page citations despite the template's evidence-image section.
4. [pending] Per course, validate only through OpenKnowledge write/edit warnings, lint, audit, and Mermaid write warnings. Do not open browsers, use preview URLs, take screenshots, or perform DOM/render verification.
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
