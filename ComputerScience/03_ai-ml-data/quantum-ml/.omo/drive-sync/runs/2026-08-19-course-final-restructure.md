# Quantum ML Course Final Restructure — 2026-08-19

## Result

- Status: **PASS**
- Approved design: `DS@v1`
- Base remote HEAD: `a90aa57af8aa5de4e85b6d916cfadd6334990177`
- Migration commit: `7939a462c4941f692f9fb3505ad8e90b5bca20ae`
- Migration push: fast-forward `a90aa57af8aa5de4e85b6d916cfadd6334990177..7939a462c4941f692f9fb3505ad8e90b5bca20ae`
- Migration remote readback: `7939a462c4941f692f9fb3505ad8e90b5bca20ae` — PASS
- Final receipt commit hash: recorded in the final agent handoff because a commit cannot embed its own stable hash without changing its content
- Blockers: none

## Approved artifacts

- Design SHA-256: `560bb7c70f581611dc82451117562c971183e5f7041a15440fbc8c9fc864536c`
- Migration ledger: `.omo/drive-sync/course-final-migration-2026-08-19.json`
- Migration-ledger SHA-256: `8b435d9cf09fd09d5bdd563ebc868e9f9c193bd4f739372d14c545eca124f35c`
- Derived registry: `.omo/drive-sync/quantum-ml-derived-artifacts.json`
- Derived-registry SHA-256: `cfb0b7ee85a12035af1cd5d906e7386799614c3571f3e3bbb7b7275320ffd4d0`

## Migration accounting

- Repository course files before migration: 440
- Repository course files after migration commit: 444
- Expected files after this receipt commit: 445
- Public assets: 184 = 174 notebooks + 10 PDFs
- Raw Drive assets: 179 = 168 Python + 10 PDFs + 1 CSV
- Public stage counts: `50/48/30/22/21/11/2`
- Migration changed-path set: 375 exact approved paths
- Missing approved paths: 0
- Unexpected paths: 0
- Protected-path changes: 0
- Manifest records: 179 unique Drive IDs/raw/artifact paths
- Public manifest artifacts: 178
- Raw-only Netflix CSV: 1
- Derived/curated registry entries: 12

## Validation

### Static and structural

- Tool tests: `20` PASS
- Notebook JSON: `174/174` PASS
- Executable Python cells: `236/236` AST PASS
- Notebook mutation classes: move-only `54`; single path rewrite `110`; duplicate badge collapse `9`; duplicate badge plus `%pip` comment `1`
- Colab URL rewrites: `133`
- Duplicate badges collapsed: `10`
- Executable code changes: exactly one complete-line `%pip` comment; recovery string preserved
- Course index targets: `184/184` resolve exactly once
- Graph/interface links: 8 unique targets PASS
- External live consumers: 4 unchanged
- Contract routing: 7 roots and 16 destination subfolders PASS
- `git diff --check`: PASS for every staged path except the exact preserved remote Colab log, whose output whitespace is retained verbatim and validated through markers/exit status

### Raw and PDF protection

- Raw tree OID before/after: `dbf68906e31e03b1891007b162c7e9290a3a818a`
- Raw paths/bytes changed: 0
- PDF Git LFS OID/payload pairs preserved: `10/10`
- Existing Colab receipts changed: 0
- Existing dated run receipts changed: 0
- ULW historical research changed: 0
- External historical evidence changed: 0

### Remote Colab

- Runner: `.omo/drive-sync/colab-runs/2026-08-19/course_final_derived_notebooks_runner.py`
- Log: `.omo/drive-sync/colab-runs/2026-08-19/course_final_derived_notebooks.log`
- Per-notebook success markers: `12/12`
- Final marker: `COURSE_FINAL_DERIVED_NOTEBOOKS_OK=12/12`
- Colab CLI exit marker: `COLAB_CLI_EXIT_STATUS=0`
- Dependency installation: remote Colab VM only

## Migration remote readback

Fresh clone: `/tmp/quantum-ml-readback.v5cAXt/repo`

- HEAD: `7939a462c4941f692f9fb3505ad8e90b5bca20ae`
- Parent: `a90aa57af8aa5de4e85b6d916cfadd6334990177`
- Course files: 444
- Public files: 184
- Raw files: 179
- Migration paths: 375
- Manifest records: 179
- Derived entries: 12
- Index targets: 184
- Colab success markers: 12
- Protected raw/run/ULW surfaces: unchanged

## Publication boundary

The migration and this receipt are ordinary fast-forward commits. No force push, history rewrite, PDF regeneration, raw-file mutation, external-note edit, or broad staging was performed. Primary-vault reconciliation occurs only after the final receipt commit is pushed and read back from remote.
