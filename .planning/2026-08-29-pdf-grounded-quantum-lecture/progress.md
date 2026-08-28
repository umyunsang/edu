# Progress: PDF-grounded quantum-lecture rewrite

## 2026-08-29

- Began the `quantum-lecture` course unit.
- Confirmed OpenKnowledge content root and read the target folder before writing.
- Loaded the official OpenKnowledge palette for visual authoring.
- Mapped all eight course PDFs to their current note filenames and extracted-text bundles.
- Visually inspected five content-bearing rendered slides for the first PDF; no source error requiring a caveat was found in the selected evidence.
- Did not inspect Git history or alter course content yet.
- Replaced `notes/01. 양자 기초 게이트 설명.md` through OpenKnowledge with a PDF-grounded, visually structured note.
- Corrected its two initial markdown lint warnings and verified a clean scoped audit, including asset-link integrity.
- Began the Braket source pass and recorded the first two visually verified console states.
- Verified the IonQ hardware and notebook-creation slides; the Braket note can now distinguish provider characteristics from the console workflow.
- Verified the notebook stop action and recorded that the source does not show a GHZ run result.
- Replaced `notes/02. 양자클라우드 Braket 기초 사용법.md`; fixed one blockquote-spacing warning and verified a clean scoped audit.
- Began the physics and linear-algebra source pass; verified eigenvector and unitary-transformation evidence slides.
- Verified rendered evidence for dimension growth and Rayleigh-Ritz; excluded missing render pages from the evidence plan.
- Verified both Schrödinger-equation slides for direct embedding in the third note.
- Replaced `notes/03. 양자컴퓨팅을 위한 물리 및 선형대수.md` and verified a clean scoped audit.
- User authorized extending the same workflow from quantum-lecture to all source-backed edu courses.
- User redirected the visual policy: remove every direct source-slide embed from notes and use original SVG or HTML-preview visuals instead.
- Completed a read-only non-Markdown source inventory for all 38 course folders and integrated the repository scope numbers.
- Began the Grover/Shor source pass; verified one operator evidence slide and rejected a section-divider slide as technical grounding.
- Verified the Grover amplitude-rotation and QFT-periodicity slides for direct evidence use.
- Verified an oracle-structure slide and restricted a public-key slide to contextual rather than procedural use.
- Replaced `notes/01-1. 양자 알고리즘 소개 — Grover와 Shor.md` and verified a clean scoped audit.
- Began the VQA source pass; separated a contextual bounded-problem example from a usable Ansatz visual.
- Verified direct VQE-definition and molecular-workflow evidence slides.
- Replaced `notes/03-1. 화학 알고리즘 소개 — VQA.md` and verified a clean scoped audit.
- A first broad direct-image grep identified five current quantum-lecture notes, but it crossed `sources/` and `work/` and hit an OpenKnowledge read-scope warning.
- That broad result is not treated as a repository-wide completion check; every future visual scan is constrained to the course `notes/` directory.
- Began retrofitting the five identified notes so all former slide locations become original SVG or HTML-preview learning visuals. Underlying PDFs and assets remain private evidence files.
- Retrofitted `notes/01. 양자 기초 게이트 설명.md`: replaced its five direct slide images with five independent SVG learning diagrams and verified a clean OpenKnowledge audit plus zero direct `../assets/` image matches in that note.
- Retrofitted the Braket, physics/linear algebra, Grover/Shor, and VQA notes in the same way, replacing 18 additional source-image embeds with independent SVG visualizations.
- Each retrofitted note passed its scoped OpenKnowledge audit. The authoritative course-scoped scan of `quantum-lecture/notes/` now returns zero direct `../assets/` image embeds.
- The remaining QML, SQD, and hardware notes will be authored directly under the source-free publication rule.
- Rewrote QML, SQD, and hardware from their extracted source text with original visual components only; each passed a clean scoped audit and contains zero direct asset image embeds.
- Completed the quantum-lecture course gate: 8 documents / 36 callouts / 46 HTML previews / 17 details / 10 Tabs / 0 direct source-slide embeds. All eight documents pass markdownlint, frontmatter, OKF, and link checks.
- Moving to the quantum-ml course for the next source-free inventory and retrofit pass.
- Completed the quantum-ml public-source conversion: replaced 10 direct original asset embeds across three notes, removed visible PDF page-only citation blocks, and added original SVGs where those removals would have dropped a note below the component floor.
- Rewrote the stale `양자 ML 과정.md` guide to remove 174 dead notebook links and present a source-free course map.
- Quantum-ml course gate passed: 11 documents / 66 callouts / 27 HTML previews / 12 details / 5 Tabs / 0 direct source-slide embeds. The scoped audit is clean and the notes contain no text-form slide-page citations.
- Next course: neural-networks.
- Completed neural-networks public-source conversion: it had no direct source-image embeds, but 26 visible text-only slide citation blocks and five PDF-page footer references were removed. Markdown auto-fix resolved the resulting blank lines.
- Neural-networks course gate passed: 5 documents / 34 callouts / 5 HTML previews / 6 details / 4 Tabs / 0 direct source-slide embeds. The scoped audit is clean and no visible PDF-page citations remain.
- Selected `ComputerScience/03_ai-ml-data/generative-ai-fine-tuning` as the next newly reconstructed course: it has two PDFs, 42 private rendered assets, and no notes directory yet. Its extracted text bundles were located; public notes will be authored without asset embeds.

## Errors

| Error | Attempt | Resolution |
| --- | --- | --- |
| JavaScript template literal ended at Markdown code fences before the OpenKnowledge write call | 1 | Assemble fence markers with a character variable so source text is transmitted unchanged. |
| OpenKnowledge exec rejected `nl`, which is outside its read-command allowlist | 1 | Use the structured lint result for line diagnostics instead. |
| OpenKnowledge exec rejected a grep pattern containing code-fence backticks | 1 | Count component markers with individual safe patterns instead. |
| Requested physics-slide render p011 does not exist | 1 | Enumerated the exact asset family; embed only available pages and avoid a textual page citation. |
| Verification script used an invalid quoted object shorthand before any tool call | 1 | Pass the path variable as a normal object property in the corrected read-only request. |
| Verification script's command template literal was rejected by the host parser | 1 | Build safe grep commands with ordinary string concatenation. |
