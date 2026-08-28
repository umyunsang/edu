# Findings: PDF-grounded quantum-lecture rewrite

## Directly supported

- `quantum-lecture` has 8 lecture notes, a `sources/` directory, and an `assets/` directory.
- The inherited `lecture-note` template explicitly requires rendered-slide evidence and supports callouts, Mermaid, KaTeX, `html preview`, and Accordion.
- `quantum-ml/notes` contains 11 notes and is the retained density reference.
- OpenKnowledge's current palette supports the 15 callout variants, Tabs, Accordion/details, Mermaid, image components, theme-aware `html preview`, and inline SVG in an HTML preview.

## Decisions

- Start with `quantum-lecture` because it is the explicitly identified visual-quality gap and can be rebuilt without consulting deleted material.
- The original PDF plus its extracted text and matching rendered image assets are the sole evidence substrate for rewritten lecture notes.
- Defer every graph relationship label until all notes in this course have been rewritten.

## First-note evidence map

- `01. 양자 기초 게이트 설명` maps to the 31-page gate PDF and its matching extracted text bundle.
- Rendered assets for Pauli summary, S-dagger, H-Z-H basis change, parameterized rotation, and CNOT entanglement are present and visually verified.
- The first note will embed those assets directly from `../assets/`; no `pdf p.N` citation block will be carried over.

## Open questions

- Exact PDF-to-current-note mapping and per-PDF slide asset availability.
- Whether any source slide contains an error that needs a note-level caveat.
