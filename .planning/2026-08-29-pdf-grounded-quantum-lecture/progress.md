# Progress: PDF-grounded quantum-lecture rewrite

## 2026-08-29

- Installed `playwright-chromium` 1.61.0 in Slidev's module-resolution path without replacing the existing global `playwright` executable.
- Installed Chromium and Chromium headless-shell revision 1228 under the Playwright cache.
- Verified `slidev export` completes without `--executable-path`.
- The dependency gate passes. The probe still reports an unresolved OpenKnowledge `<Tab>` component under raw Slidev, so this export does not establish document-component compatibility.

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
## 2026-08-29 SVG overflow gate

- Completed SVG-only rendering QA without opening whole note pages: quantum-lecture 46 SVGs, quantum-ml 20 SVGs, neural-networks 3 SVGs.
- Fixed confirmed geometry faults in 7 visuals across 6 notes: quantum-lecture 4, quantum-ml 2, neural-networks 1. Every repaired SVG was rerendered at 320px and 390px without clipping or horizontal overflow.
- Scoped OpenKnowledge audits are clean: quantum-lecture 8 docs, quantum-ml 11 docs, neural-networks 5 docs.
- Keep the separate risk visible: most quantum-lecture diagrams still use tiny projected mobile type; that is not an overflow defect and was not silently redesigned in this pass.
- Began three isolated course lanes: generative-ai-fine-tuning, large-language-models, and ai-system-design.

## 2026-08-29 Parallel course gates

- Completed `generative-ai-fine-tuning`: 8 notes plus README, 32 callouts / 16 HTML previews / 8 details / 8 Tabs / 0 source embeds; 9-document audit and lint clean. Preserved three source typos explicitly rather than correcting them.
- Completed `ai-system-design`: 5 notes plus README, 25 callouts / 12 HTML previews / 10 details / 5 Tabs / 9 Mermaid / 0 source embeds; 6-document audit and lint clean. Treated the identical week-4 extracts as one source without inventing sparse details.
- `large-language-models` is in final README and course-gate validation. Reused the free lane for `computer-vision` reconstruction.
- Completed `large-language-models`: 16 notes plus README, 64 callouts / 16 HTML previews / 16 details / 16 Tabs / 0 source embeds; 17-document audit and lint clean. Two source faults are visible warnings, not silent corrections.
- Completed `computer-vision`: 6 valid lecture notes plus README, 39 callouts / 12 HTML previews / 13 details / 6 Tabs / 6 Mermaid / 0 source embeds; 7-document audit and lint clean. Excluded declared duplicate PDFs and left three zero-text exam sources undocumented rather than infer content.
- Active parallel courses: machine-learning and big-data-analysis.
- Completed `machine-learning`: 21 notes plus README, 84 callouts / 21 HTML previews / 21 details / 21 Tabs / 21 Mermaid / 0 source embeds; 22-document audit and lint clean. Two extraction limits are explicit warnings with no invented reconstruction.
- Completed `big-data-analysis`: 10 notes plus README, 41 callouts / 20 HTML previews / 20 details / 10 Tabs / 10 Mermaid / 0 source embeds; 11-document audit and lint clean. Declared duplicate and out-of-scope problem-set PDFs remain undocumented.
- Active parallel courses: ml-projects and artificial-intelligence.
- Completed `ml-projects`: 10 notes plus README, 40 callouts / 20 HTML previews / 20 details / 10 Tabs / 10 Mermaid / 0 source embeds; 11-document audit and lint clean. Used no notebook evidence because no notebooks exist; excluded only the personal certificate and duplicate files.
- Active parallel courses: artificial-intelligence and operating-systems.
- Completed `artificial-intelligence`: 29 notes plus README, 116 callouts / 29 HTML previews / 29 details / 29 Tabs / 0 source embeds; 30-document audit and lint clean. Six source issues remain warnings rather than corrected material.
- Active parallel courses: operating-systems and database-systems.
- Completed `operating-systems`: 15 notes plus README, 60 callouts / 30 HTML previews / 30 details / 15 Tabs / 15 Mermaid / 0 source embeds; 16-document audit and lint clean. Valid assessment notes use 90–92; source-less MYBOX materials, an operational handout, and a duplicate are explicitly excluded.
- Active parallel courses: database-systems and linux.
- Completed `database-systems`: 17 notes plus README, 68 callouts / 17 HTML previews / 17 details / 17 Tabs / 17 Mermaid / 0 source embeds; 18-document audit and lint clean. Two exercise notes use the required 90–91 numbering; highlights were not used as a density substitute.
- Active parallel courses: linux and computer-networks.
- Completed `linux`: 15 notes plus README, 60 callouts / 30 HTML previews / 30 details / 15 Tabs / 15 Mermaid / 0 source embeds; 16-document audit and lint clean. The absent week 12, a duplicate week 4, and one week-label mismatch are explicit source issues.
- Active parallel courses: computer-networks and computer-architecture.
## 2026-08-29 Full rejection and official-template reset

- The user rejected every previous note rewrite, component count, SVG/render check, and course gate. None of the earlier completion entries are accepted.
- Do not delete the existing notes without a separate deletion instruction; rewrite them in place.
- Do not open browsers, resolve preview URLs, take screenshots, inspect DOM output, or run any rendering verification.
- Required contract for every rework: the official OpenKnowledge skill, lecture-note template, components-and-visuals guidance, palette starters, and project Mermaid skill. Use only official native components, Tabs, and palette-starter structures; do not infer, hand-roll, or add components.
- The user copyright policy still overrides only the template's source-image/page-citation examples: public notes omit source PDF/assets images and textual page citations.
- Root coordinates scope, source boundaries, and integration only. In-place rewrites are owned by Terra xhigh agents; no browser, preview, screenshot, DOM, or local-renderer operation is permitted.
- The first reset drafts exposed another rejected pattern: mechanically placing multiple palette starters in a note. Writers are now replacing those drafts with topic-relevant selections only, while retaining the 10 meaningful-component floor.
- `quantum-lecture` and `quantum-ml` have now been rewritten under the superseding contract. Their only recorded validation surface is OpenKnowledge write/edit warnings plus lint/audit and Mermaid write warnings; no rendering claim is attached. The next independent course lane is computer-vision.
- `neural-networks` has also completed the same rewrite and its per-note component inventory (all notes remain above the meaningful-component floor after excluding common template structure). Its next lane is ai-system-design.
- `computer-vision` has completed the official-template rewrite with one topic-relevant palette starter per document and OpenKnowledge static validation only. Its next lane is big-data-analysis.
- `ai-system-design` has completed the same per-note inventory and static validation. Its next lane is ml-projects.
- `generative-ai-fine-tuning` and `large-language-models` have completed the official-template rewrite and a 24-document individual component/recipe review. Their next lane is machine-learning.
- `big-data-analysis` has completed the same official-template rewrite with static OpenKnowledge validation only. Its next lane is operating-systems.
- `ml-projects` has completed its per-note inventory and static validation, documenting sparse or incorrect extraction rather than inferring it. Its next lane is linux.
- `machine-learning` has completed its 21-note rewrite and static validation, including two deliberately limited notes for source-text gaps. Its next lane is artificial-intelligence.
- `operating-systems` has completed a second per-note anti-repetition correction and static validation, retaining no note for the text-empty source. Its next lane is database-systems.
- All previous quality and progress metrics below are historical rejected output, not acceptance evidence.

## 2026-08-29 Source-depth and duplication reset

- User identified that generative-ai-fine-tuning notes 01, 15, and 26 contain the same substance despite different titles. This confirms that the previous component-count/template gate did not establish PDF coverage or note distinctness.
- User rejected every note produced by every previous lane, across all courses. All prior completion statements, source maps, component counts, lint/audit results, and claimed course gates are non-acceptance history only.
- Existing note files are not deleted without separate authorization; they may be read only to locate defects and must never anchor new structure, wording, depth, or visual choices.
- All Terra agents were interrupted. New delegated work uses `gpt-5.6-sol` with xhigh reasoning only.
- Three Sol lanes are now running read-only: exact duplicate forensics for notes 01/15/26, whole-course duplicate and missing-topic audit, and independent PDF-segment mapping for 01/15/26/40/46/50/62/72.
- No further course authoring is accepted until the generative-ai-fine-tuning source-depth gate is redesigned and demonstrated on these documents.
- The Sol audits rejected all 8 notes and README: source-specific visible density was only about 5% to 35%, all 8 visuals used invented or mismatched values, and the shared template scaffold dominated the documents.
- The two 82-page PDFs have different hashes but identical extracted page text, so they count as one content authority. The new independent segments are 01=1-14, 15=15-25, 26=26-39, 40=40-45, 46=46-49, 50=50-61, 62=62-71, 72=72-82.
- Three Sol xhigh authoring lanes now own disjoint note sets (01/15/26, 40/46/50, 62/72). They must full-replace from the extracts, meet segment-specific prose-depth targets without padding, use no invented math/data, and leave README until all 8 notes pass the new coverage/duplication gate.
- User reviewed the source-deep drafts and rejected their essay-like delivery. The same eight notes are being rebuilt again with compact, keyword-led text and varied official components selected by information type; source coverage stays intact while long narrative paragraphs are removed.
# 2026-08-29 project-template and Slidev reset

- Paused all note-writing lanes.
- Confirmed the project root has no `.ok/templates/`.
- Confirmed `ComputerScience/.ok/templates/` has same-name local templates that override root templates and contain superseded image/PDF-link rules.
- User changed the render gate: all rewritten documents receive `slides: true`; render validation uses Slidev CLI, not browser-based QA.
- Official OpenKnowledge template/palette guidance and the local markdown-slides skill were re-read before authoring the project templates.

## 2026-08-29 project-template reset completed; Slidev fidelity blocked

- Created the project-root `.ok/templates/lecture-note.md`, `course-index.md`, and `practice-note.md` as the only three default templates. Removed the obsolete same-name `ComputerScience/.ok/templates/` overrides, so deep course folders inherit the root contract.
- Kept the templates as short skeletons. They require `slides: true`, logical `---` slide boundaries, source-derived section order, source-backed visual slots, and no embedded palette example data.
- Recalled OpenKnowledge palette v1. The authoritative embed starters remain `chart`, `stat-cards`, `custom-svg`, and `interactive-control`; source-shape gates and starter-preservation rules now live in `AGENTS.md` and `ComputerScience/.ok/frontmatter.yml` rather than being copied into every note.
- Added durable gates for visual roles, the 10-meaningful-component review floor, source-sparsity exceptions, sibling-note duplication, placeholder/sample-data rejection, and truthful validation labels.
- Instantiated all three templates in a temporary folder and verified OpenKnowledge lint/audit: 4 documents scanned, `markdownlint`, `frontmatter`, `okf`, and `links` all ran with zero findings. The temporary folder was deleted afterward.
- Verified raw Slidev 52.19.1 build and export for all three template probes: 3/3 builds and 3/3 PDF exports exited zero; exported decks were 5, 5, and 6 pages. `playwright-chromium` 1.61.0 is installed in Slidev's dependency tree.
- Validator originality audit found no tracked drift in `.ok/config.yml`, `scripts/check_mermaid.mjs`, `scripts/graph_check.py`, or any tracked `scripts/` file. `.ok/okf/**` is ignored and has no Git authority, so it was preserved rather than guessed or overwritten.
- The global OpenKnowledge setting file has `slides.enabled: true`, but the running MCP process still reports `false`; treat the current service state as stale until OpenKnowledge is restarted or otherwise reloads user settings.
- Critical blocker: OpenKnowledge 0.64.2 has no official OK-to-Slidev adapter. Raw Slidev renders Mermaid, KaTeX, and the five base GFM alerts, but it does not execute `html preview` and has no `Tabs/Tab` component. Build/export success therefore proves compilation only, not OpenKnowledge visual fidelity. No course rewrite may receive a Slidev visual-compatibility PASS until this contract conflict is resolved.

## 2026-08-29 approved local Slidev compatibility layer

- Implemented the separately approved local addon under `packages/slidev-addon-openknowledge/` and pinned the project toolchain in `package.json`/`package-lock.json`.
- Added source-contract and negative-control fixtures under `tests/slidev-compat/`; fixtures materialize only in temporary directories, not in the vault.
- `npm run test:slidev-compat` passes: source contract, raw Slidev negative control, addon build, CLI export/static assertions. The required final label remains `VISUAL/INTERACTION COMPATIBILITY UNVERIFIED`.
- Proved root-addon inheritance from a temporary deep course path with Slidev build and a four-page CLI export; removed the probe document, its generated index entry, and both transient build directories afterward.
- Repaired durable authority drift in `AGENTS.md` and `ComputerScience/.ok/frontmatter.yml`: lecture/index sources are PDF+extract only, prior lane bodies/gates are rejected, `start --open` and the write-capable ledger script were removed from the default rewrite gate, and validation uses the pinned local addon/toolchain.
- OpenKnowledge audit for `AGENTS.md` and the cleaned course index ran markdownlint/frontmatter/okf/links with zero findings. Root template files remain hidden-path artifacts that OpenKnowledge refuses to audit directly, so no direct template-audit PASS is claimed.
- First resumed proving course: `ComputerScience/03_ai-ml-data/generative-ai-fine-tuning`; a Sol xhigh lane is rebuilding the source map before any note write.
- Source map completed without reading rejected note prose: the two 82-page PDFs have different hashes but identical 23,091-character extracted page streams, so one is canonical and the ` 2.pdf` file is duplicate-content evidence only.
- Eight non-overlapping note spans are fixed at 01=1–14, 15=15–25, 26=26–39, 40=40–45, 46=46–49, 50=50–59, 60=60–71, and 72=72–82. The stale `62` note will be moved to `60` because the source-backed manufacturing section begins at original slide 60 and filenames must follow the original slide number.
- Every segment has zero extracted source equations, so Math is prohibited throughout this course. Source-grounded palette gates are limited to the documented comparisons/ratios; sparse or approximate quantities do not open a starter gate.
- Lane A (01/15/26) completed full replacement with Sol xhigh. Root independently corrected the 01 consortium-count conflict so the document now exposes, rather than reconciles, the source title/body disagreement; all three documents remain OpenKnowledge lint/audit clean.
- Lanes B (40/46/50) and C (60/72) are active under Sol xhigh with disjoint write ownership. Root owns README, source-fidelity review, sibling-duplication review, and the final course gate.
- The local Slidev compatibility layer was strengthened after independent review: nested callouts transform independently; official bounded icons flow through Callout/Accordion/Toggle; preview title, active theme tokens, and CSP diagnostics are represented; CLI assertions now include closed-callout content, the second Tab panel, and authored/output preview sentinels.
- The strengthened suite passes all four declared surfaces: source contract, raw-Slidev negative control, addon build, and CLI export/static assertions. Viewport fit, visual fidelity, and real interaction remain explicitly unverified; no browser, DOM, screenshot, or rsvg validation was performed.
- `generative-ai-fine-tuning` is the first accepted proving course under the reset contract. Eight notes plus README were rebuilt from one canonical 82-slide extracted stream; the second PDF is duplicate-content only. Independent Sol review initially rejected the README Mermaid and note 50, and root corrected both before acceptance.
- Final notes-only inventory: 8 documents, 30 callouts, 9 HTML previews, 11 details, 11 Tabs, 14 Mermaid blocks, and 0 source-slide embeds. Including README: 9 documents, 33 callouts, 11 HTML previews, 12 details, 11 Tabs, 15 Mermaid blocks, and 0 source-slide embeds. Every lecture note has at least 12 meaningful component surfaces.
- Course-wide OpenKnowledge lint/audit scanned 11 documents with markdownlint/frontmatter/okf/links and found 0 errors/0 warnings. Mermaid parser checked 15 blocks with 0 failures. README and all 8 notes built and exported through the pinned Slidev CLI; exported page counts were README 6 and notes 9/8/8/7/6/9/6/8.
- The compatibility regression suite again reported SOURCE CONTRACT PASS, RAW SLIDEV NEGATIVE CONTROL PASS, SLIDEV BUILD PASS, and SLIDEV EXPORT/STATIC PASS. Visual/interaction compatibility remains explicitly unverified.
