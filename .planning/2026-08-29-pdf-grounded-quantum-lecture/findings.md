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
- User authorized repository-wide continuation with the same evidence and visual-quality contract; the five explicitly source-empty courses remain out of scope for note creation.

## Repository source inventory

- 33 of 38 course folders have source PDFs: 476 PDFs and 9,817 rendered WebP slide assets in total.
- `java-programming`, `coding-test`, `degree-portfolio`, `container-orchestration`, and `mathematical-logic` have no PDFs or assets and will retain no reconstructed notes.
- The five remaining quantum-lecture PDFs have usable asset coverage: Grover/Shor 40, VQA 39, QML 24, SQD 26, and hardware 37 rendered pages.

## Grover-Shor evidence map

- The Grover/Shor source is 59 pages, but rendered slide assets exist only through p042; p043 through p059 will not be used as image evidence.
- A verified operator slide shows controlled gates and the identity $HXH=Z$. The inspected p018 asset is only a Binary Sudoku section divider, so it will not carry a technical claim.
- A Grover iteration slide directly shows uniform superposition rotating toward a target state. A QFT slide describes extracting hidden periodicity and presents an asymptotic speed comparison; the rewrite will preserve this as a source claim rather than make a broad performance promise.
- The Grover oracle slide shows an XOR-based Binary Sudoku constraint structure. The inspected public/private-key slide provides cryptography context but not a Shor factoring procedure, so it will be used only as scoped context if at all.

## Grover-Shor verification

- `01-1. 양자 알고리즘 소개 — Grover와 Shor.md` passed OpenKnowledge markdownlint, frontmatter, OKF, and link checks with zero findings.
- Its measured structure is 4 callouts, 2 HTML previews, 2 details blocks, 1 Tabs block, 1 Mermaid diagram, and 4 embedded rendered slides.
- The note explicitly limits QFT's displayed speed comparison to the source's QFT stage and records the absent p043-p059 rendered evidence.

## VQA evidence map

- The VQA source identifies repeated measurements to select a maximum or minimum bounded value, VQE as a ground-state-energy problem, Ansatz families, and a molecular-energy workflow.
- The inspected p008 image is a bounded-problem example without a VQA mechanism, while p027 is a valid Ansatz visual. The next evidence set will use a direct VQE definition and flowchart.
- Direct VQE and VQE-flow slides define VQE as finding ground-state energy and show molecular inputs, an Ansatz, optimizer/measurement conditions, and molecular-energy analysis as an iterative workflow.

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

## Second-note verification

- `02. 양자클라우드 Braket 기초 사용법.md` passed OpenKnowledge markdownlint, frontmatter, OKF, and link checks with zero findings.
- Its measured structure is 5 callouts, 2 HTML previews, 2 details blocks, 1 Tabs block, 1 Mermaid diagram, and 5 embedded rendered slides.
- The note explicitly distinguishes source-captured interface state, source filename typo, and unverified GHZ execution.

## Third-note evidence map

- The physics and linear-algebra source explicitly links eigenvectors, unitary transformation, tensor products, quantum state, Schrödinger equation, and Rayleigh-Ritz.
- Verified slides define an eigenvector as a nonzero vector whose direction is maintained by a linear transformation, and describe unitary transformations as norm-preserving and reversible.
- The available evidence set also illustrates density loss with increasing dimension and states the Rayleigh-Ritz energy lower-bound inequality. The source has no p011 or p016 rendered asset, so those pages will not be cited as image evidence.
- The two Schrödinger slides expose the time-evolution equation, wavefunction-probability interpretation, time-independent form, and energy eigenvalue equation; both are eligible for embedded source grounding.

## Third-note verification

- `03. 양자컴퓨팅을 위한 물리 및 선형대수.md` passed OpenKnowledge markdownlint, frontmatter, OKF, and link checks with zero findings.
- Its measured structure is 5 callouts, 2 HTML previews, 2 details blocks, 1 Tabs block, 1 Mermaid diagram, and 6 embedded rendered slides.
- Missing physics render assets remained excluded; the note uses no substitute page-text citation for them.

## Open questions

- Exact PDF-to-current-note mapping and per-PDF slide asset availability.
- Whether any source slide contains an error that needs a note-level caveat.
