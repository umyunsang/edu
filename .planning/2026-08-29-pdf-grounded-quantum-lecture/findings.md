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

## First-note verification

- `01. 양자 기초 게이트 설명.md` passed OpenKnowledge markdownlint, frontmatter, OKF, and link checks with zero findings.
- Its measured structure is 6 callouts, 2 HTML previews, 2 details blocks, 1 Tabs block, 1 Mermaid diagram, and 5 embedded rendered slides.
- Both HTML previews contain purpose-built inline SVG visuals: an interactive $R_y$ amplitude view and a CNOT control-target diagram.

## Second-note evidence map

- The Braket source is a 12-page console walkthrough, not a general AWS reference; every procedure in the rewrite will be limited to what the captured interface shows.
- The first two verified slides show the Braket dashboard and device list, including a region selector and availability states.
- The next verified slides show an IonQ hardware claim and the notebook-instance creation screen; current interface screenshots are treated as the source's captured state, not a claim about today's AWS console.
- The final verified workflow slide exposes the notebook action menu with a stop action. The source's GHZ slide is only a section marker, so it will not be used as evidence for a successful execution.

## Open questions

- Exact PDF-to-current-note mapping and per-PDF slide asset availability.
- Whether any source slide contains an error that needs a note-level caveat.
