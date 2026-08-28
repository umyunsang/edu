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

1. [in_progress] Reject every prior course output and prior gate result. Rebuild in place from source PDFs/extracts with the official OpenKnowledge skill, lecture-note template, palette starters, and project Mermaid skill as the exclusive component contract.
2. [pending] Per course, read its sources and the official template before writing; apply only official markdown-native components, Tabs, and unmodified palette-starter structures populated with source-grounded data.
3. [pending] Keep the copyright override: omit source slide images and text-only page citations despite the template's evidence-image section.
4. [pending] Per course, validate only through OpenKnowledge write/edit warnings, lint, audit, and Mermaid write warnings. Do not open browsers, use preview URLs, take screenshots, or perform DOM/render verification.
5. [pending] Report course component totals only after the official-template rewrite; no prior totals are accepted.
6. [pending] Leave `java-programming`, `coding-test`, `degree-portfolio`, `container-orchestration`, and `mathematical-logic` without reconstructed notes because their sources are empty.
7. [pending] After all rewrites, add graph relationship types in the separately authorized post-rewrite pass.

## Quality floor per lecture note

- At least 10 meaningful visual/interactive components across callouts, `html preview`, `details`, `Tabs`, Mermaid, highlights, math, and embedded slide images.
- Zero direct source-slide image embeds.
- At least one purpose-built SVG or HTML-preview visual when it adds explanatory value; not decorative repetition.
