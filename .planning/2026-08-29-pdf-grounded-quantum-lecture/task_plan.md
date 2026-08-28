# Task Plan: original-visual-free edu rewrite

## Goal

Rebuild every lecture note in each edu course from its source PDFs and extracted text while replacing every direct original-PDF visual embed with an original SVG or HTML-preview learning visual.

## Hard boundaries

- Never inspect deleted notes through Git history.
- Do not embed rendered source slides or any original-PDF-derived image in a note. Keep source PDFs and rendered assets private as evidence material, but do not delete those underlying source assets without separate authorization.
- Replace each removed source visual with an original abstracted SVG or HTML-preview learning visual; do not reproduce the source slide's layout or text.
- Do not add Obsidian wikilinks, Dataview, or hierarchical tags.
- Preserve source errors and mark them as source errors when present.
- Do not commit or push until the user requests it; never stage unrelated or dot-directory changes.

## Phases

1. [complete] Establish the OpenKnowledge component palette, source-to-note map, and quantum-ml density baseline.
2. [complete] Inventory direct source-slide embeds in the existing quantum-lecture notes and define the original-visual replacement rule. Future inventory remains course-scoped.
3. [complete] Retrofit the five completed quantum-lecture notes, replacing every source slide embed with original SVG or HTML preview visuals.
4. [complete] Rebuild the remaining quantum-lecture notes with zero source-slide embeds.
5. [complete] Count quantum-lecture components and source-slide embeds; all notes clear the quality floor.
6. [complete] Run scoped markdown, frontmatter, OKF, and link validation; report the quantum-lecture metrics.
7. [complete] Render and repair inline SVG learning visuals in the completed courses. Confirmed clipping or horizontal overflow was fixed with 320px and 390px SVG renders; mobile typography is recorded separately rather than silently hidden.
8. [in_progress] Quality-reset every completed and future course against the official OpenKnowledge template, palette, component/preview guidance, KaTeX syntax, and Mermaid layout skill. Replace the production-line marker recipe with purpose-built themed HTML previews, charts, interactive controls, responsive SVG, and concise Mermaid. New quality-rebuild agents use `gpt-5.6-terra` at `xhigh`.
9. [pending] Leave `java-programming`, `coding-test`, `degree-portfolio`, `container-orchestration`, and `mathematical-logic` without reconstructed notes because their sources are empty.
10. [pending] After all rewrites, add graph relationship types in the separately authorized post-rewrite pass.

## Quality floor per lecture note

- At least 10 meaningful visual/interactive components across callouts, `html preview`, `details`, `Tabs`, Mermaid, highlights, math, and embedded slide images.
- Zero direct source-slide image embeds.
- At least one purpose-built SVG or HTML-preview visual when it adds explanatory value; not decorative repetition.
