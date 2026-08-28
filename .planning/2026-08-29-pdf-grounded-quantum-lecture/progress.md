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

## Errors

| Error | Attempt | Resolution |
| --- | --- | --- |
| JavaScript template literal ended at Markdown code fences before the OpenKnowledge write call | 1 | Assemble fence markers with a character variable so source text is transmitted unchanged. |
| OpenKnowledge exec rejected `nl`, which is outside its read-command allowlist | 1 | Use the structured lint result for line diagnostics instead. |
| OpenKnowledge exec rejected a grep pattern containing code-fence backticks | 1 | Count component markers with individual safe patterns instead. |
| Requested physics-slide render p011 does not exist | 1 | Enumerated the exact asset family; embed only available pages and avoid a textual page citation. |
