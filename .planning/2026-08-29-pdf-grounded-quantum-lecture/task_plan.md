# Task Plan: PDF-grounded quantum-lecture rewrite

## Goal

Rebuild every `quantum-lecture` lecture note from its current PDF, extracted text, and rendered slide assets; make each note a high-quality visual learning artifact in OpenKnowledge.

## Hard boundaries

- Never inspect deleted notes through Git history.
- Use slide evidence only by embedding the matching rendered asset; do not retain textual `pdf p.N` evidence callouts.
- Do not add Obsidian wikilinks, Dataview, or hierarchical tags.
- Preserve source errors and mark them as source errors when present.
- Do not commit or push until the user requests it; never stage unrelated or dot-directory changes.

## Phases

1. [complete] Establish the OpenKnowledge component palette, source-to-note map, and quantum-ml density baseline.
2. [complete] Rebuild the first lecture note from PDF/extract/assets and validate rendering/lint.
3. [in_progress] Rebuild remaining quantum-lecture notes, saving each completed note through OpenKnowledge.
4. [pending] Count course-level components and slide embeds; fix any quality-floor gap.
5. [pending] Run scoped graph and markdown validation; report the course metrics and next course decision.

## Quality floor per lecture note

- At least 10 meaningful visual/interactive components across callouts, `html preview`, `details`, `Tabs`, Mermaid, highlights, math, and embedded slide images.
- At least one source-grounding rendered slide image when a claim requires slide evidence.
- A purpose-built SVG or interactive visual when it adds explanatory value; not decorative repetition.
