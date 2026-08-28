# Edu Reference Layer Stage 1–5 Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Every task and verification step uses checkbox (`- [ ]`) tracking. Do not use the retired `.omo` workflow.

**Goal:** Deliver a full-vault, local-first evidence substrate whose immutable source bindings, canonical evidence IDs, citations, preflight verdicts, and Stage-5 acceptance evidence are identical across Codex, Claude Code, and Gemini CLI.

**Architecture:** Build a tracked control plane over the existing safe walker/inventory/registry foundation; materialize immutable external extraction and canonical closures; build replaceable lexical/graph/vector adapters from one committed snapshot; expose exactly four read-only local STDIO MCP tools; then evaluate the golden corpus and deterministic full-vault batches before a truthful release decision. JSONL remains authority, Parquet and indexes remain digest-bound derivatives, and only a verified committed head selects a snapshot.

**Tech Stack:** Python 3.12 under `uv`, Java 21 LTS for OpenDataLoader PDF 2.5.0, JSON Schema 2020-12, canonical JSON/JSONL, PyArrow 25.0.0, SQLite FTS5, NumPy exact cosine, optional LanceDB 0.34.0, MCP Python SDK 1.28.1 (`<2`), pytest, and local Codex/Claude/Gemini CLIs.

## Global Constraints

- Authority: [approved design](<../specs/2026-07-22-edu-reference-layer-stage1-5-design.md>), this master plan, and the five linked stage plans.
- Source vault is read-only. No source overwrite, deletion, rename, content normalization, or metadata repair is authorized.
- `.omo/**` is retired and must not be read, executed, created, modified, staged, or used as acceptance evidence.
- Registry, raw runs, canonical JSONL/Parquet, models, databases, logs, and reports live outside the vault and Git under a configured `0700` root; files containing source-derived data use `0600`.
- Commit only explicit task paths. Never use `git add .`, and preserve all unrelated dirty worktree changes.
- No remote inference, remote hybrid extraction, public HTTP MCP, telemetry egress, or client write tool.
- A scoped test PASS proves only its named scope. It does not imply stage acceptance, client activation, operational readiness, or final release acceptance.
- Stop a publish/activation step on source drift, head-CAS mismatch, schema or closure mismatch, unresolved PDF quarantine, citation mismatch, benchmark regression, resource-budget failure, or unrelated-path mutation. Diagnose and repair within the same stage; do not mark the active goal blocked.
- All golden-corpus and full-vault operations require a frozen exact selection/batch manifest and one verified `snapshot_commit_id`.

## Plan Set and Execution Order

1. [Stage 2 Task 1 — isolated runtime foundation](<./2026-07-22-edu-reference-layer-stage2-extraction-canonicalization.md#task-1-lock-and-attest-the-isolated-extraction-runtime>)
2. [Stage 1 — Inventory and registry](<./2026-07-22-edu-reference-layer-stage1-inventory-registry.md>)
3. [Stage 2 Tasks 2–9 — Extraction and canonicalization](<./2026-07-22-edu-reference-layer-stage2-extraction-canonicalization.md>)
4. [Stage 3 — Search and graph adapters](<./2026-07-22-edu-reference-layer-stage3-search-graph-adapters.md>)
5. [Stage 4 — Read-only MCP and clients](<./2026-07-22-edu-reference-layer-stage4-mcp-clients.md>)
6. [Stage 5 — Evaluation and operations](<./2026-07-22-edu-reference-layer-stage5-evaluation-operations.md>)

The first pass executes Stage 1→5 against the digest-frozen representative corpus. The second pass reuses the same locked pipeline in deterministic batches of at most 100 assets or 512 MiB, with an oversized single asset in its own batch, until the frozen full-vault population closes.

---

### Task 0: Freeze the pre-implementation baseline

**Files:**
- Read: `docs/superpowers/specs/2026-07-22-edu-reference-layer-stage1-5-design.md`
- Read: `scripts/pkm/reference_*.py`
- Read: `scripts/pkm/tests/test_reference_*.py`
- Modify: `.planning/2026-07-22-gjc-reference-layer-codex-execution/progress.md`

- [ ] Record `git status --short`, `git diff --name-only`, current HEAD, Python/uv/Java/SQLite/client versions, free disk space, and the exact external-root filesystem identity in `progress.md`.
- [ ] Assert `git diff --name-only -- .omo` is empty without opening `.omo` content.
- [ ] Run the OMO-independent baseline:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_asset_walker.py \
  scripts/pkm/tests/test_reference_inventory_contracts.py \
  scripts/pkm/tests/test_reference_inventory_runtime.py \
  scripts/pkm/tests/test_reference_registry_contracts.py \
  scripts/pkm/tests/test_reference_registry_runtime.py \
  scripts/pkm/tests/test_reference_release_contracts.py \
  scripts/pkm/tests/test_reference_canonical_contracts.py \
  scripts/pkm/tests/test_reference_evaluation_contracts.py \
  scripts/pkm/tests/test_reference_security_contracts.py \
  scripts/pkm/tests/test_reference_dependency_contracts.py \
  --deselect scripts/pkm/tests/test_reference_dependency_contracts.py::test_task7_manifest_paths \
  --deselect scripts/pkm/tests/test_reference_dependency_contracts.py::test_task7_required_artifact_present
```

Expected: all collected non-retired tests pass; exactly the two selected test functions are deselected, with the parametrized artifact cases reported under the second function. Record the exact count rather than hard-coding it as future acceptance.

- [ ] Verify no cache artifact was created:

```bash
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
```

Expected: no output.

- [ ] If baseline failure is unrelated existing drift, record exact evidence and isolate it; if it touches a planned path, repair it as the first TDD slice before continuing.

### Task 1: Execute the single runtime-lock owner task

**Files:**
- Execute: Stage-2 plan Task 1 only

- [ ] Execute [Stage-2 Task 1](<./2026-07-22-edu-reference-layer-stage2-extraction-canonicalization.md#task-1-lock-and-attest-the-isolated-extraction-runtime>) exactly once before Stage-1 live census or any media extraction. That task is the sole file/commit owner for the runtime project, manifest code, schema, tests, Java receipt, and `uv.lock`.
- [ ] Record its verified runtime-manifest digest in the Stage-1 selection/publication evidence and reuse the same digest through Stage 5. Any later dependency/model change creates a new lock and restarts the golden Stage 1→5 sequence.

### Task 2: Execute golden-corpus Stage 1→5

**Files:** Follow all five stage plans.

- [ ] Stage 1 publishes an `activation=false`, `stage_verified=1` committed snapshot containing the exact golden-corpus inventory, source registry, and selection manifest closure.
- [ ] Stage 2 publishes immutable extraction bundles and the four canonical authorities: `source-observations.jsonl`, `elements.jsonl`, `relations.jsonl`, and `lineage.jsonl`; every accepted PDF has `document.md`, `document.json`, `assets/` when present, and `manifest.json`.
- [ ] Stage 3 builds and verifies SQLite lexical/graph indexes, exact vector oracle, selected model vectors, and hybrid retrieval while retaining canonical IDs and citations.
- [ ] Stage 4 proves raw strict protocol/security/tool bounds, isolated three-client template/discovery parity, and deterministic direct MCP response parity with `model_e2e=not_run` and no persistent activation.
- [ ] Stage 5 closes quality, citation, retrieval, performance, backup/restore, recovery, rollback, and isolated actual Codex/Claude/Gemini model-E2E evidence for the golden corpus. It does not perform persistent activation or emit a post-persistent-activation acceptance report during the golden slice.
- [ ] Publish only a golden `golden_accepted=true`, `release_accepted=false`, `activation=false` head if every golden Stage-5 gate passes and no selected asset remains quarantined. The slice never authorizes release or client activation.

### Task 3: Expand deterministically to the full vault

**Files:**
- Create externally: `$EDU_REFERENCE_ROOT/control/full-vault-batches.json`
- Create externally per batch: `$EDU_REFERENCE_ROOT/runs/batches/<batch_id>/manifest.json`
- Modify: `.planning/2026-07-22-gjc-reference-layer-codex-execution/progress.md`

- [ ] Freeze the full-vault denominator from the verified Stage-1 inventory and sort allowed assets by canonical path ordering.
- [ ] Generate non-overlapping batches of no more than 100 assets or 512 MiB; an asset larger than 512 MiB becomes a one-item oversized batch.
- [ ] For each batch, run Stage 1B first: revalidate the frozen inventory hash/identity for every member, reconcile logical-source/location/observation IDs under the v1.1 registry writer lock, publish typed selection/source-registry closures, and verify the committed Stage-1B snapshot from a fresh process.
- [ ] Run Stage 2–5 only from that exact Stage-1B snapshot and the same locked code/runtime/model/profile; do not substitute an unbenchmarked parser, model, adapter, or client profile.
- [ ] On asset-local extraction failure, quarantine that asset and continue diagnostic work, but do not mark the batch or final release accepted.
- [ ] After each batch, reconcile processed, quarantined, excluded terminal, and remaining counts against the frozen denominator and publish only `activation=false` stage evidence.
- [ ] Predict total wall time from observed throughput after the golden run and early batches; require projected completion within 24 hours and at least 20% free-space margin before continuing bulk work.

### Task 4: Run independent full-vault acceptance and handoff

**Files:**
- Create externally: `$EDU_REFERENCE_ROOT/reports/final-acceptance.json`
- Modify: `.planning/2026-07-22-gjc-reference-layer-codex-execution/{task_plan.md,findings.md,progress.md,.attestation}`

- [ ] Re-run every stage's contract, runtime, negative, determinism, and performance suite against the exact final snapshot.
- [ ] Independently recompute closure digests, canonical ID sets, JSONL/Parquet parity, adapter snapshot bindings, citation resolutions, and three-client output parity without trusting self-reported run events.
- [ ] Perform restore to a fresh external root, point temporary clients at the restored root, and repeat the four-tool smoke/E2E suite.
- [ ] Perform rollback from current accepted head to the previous accepted head and back, proving no source or canonical object mutation.
- [ ] Generate acceptance reports with separate fields for `contract_tests`, `implementation_complete`, `golden_accepted`, `full_vault_complete`, `operational_ready`, `client_activation_permitted`, `activation_verified`, and `release_accepted`.
- [ ] Require population conservation 100%, accepted provenance 100%, full-population provenance at least 95%, every allowed PDF accepted through OpenDataLoader Markdown+JSON, no unresolved PDF quarantine, important-claim citation precision/coverage 100%, all retrieval/SLO gates, and three-client parity 100%.
- [ ] If and only if the pre-activation report and independent audit pass, publish its accepted inactive head and render commands bound to that exact digest. Run disposable activation/parity, then guarded persistent local activation; remote/public activation remains prohibited.
- [ ] After post-persistent-activation four-tool smoke and parity pass, independently audit and publish the terminal accepted active head containing the live-smoke/config evidence closure. On failure, remove only exact created client entries and forward-roll back.
- [ ] Re-attest the task plan and update the active goal only after no required work remains.

## Commit Discipline

Each task uses: failing test → minimal implementation → targeted tests → stage suite → explicit-path `git add` → `git diff --cached --check` → explicit commit. Suggested commit sequence:

1. `build: lock reference layer runtime`
2. Stage-1 commits listed in the Stage-1 plan
3. Stage-2 commits listed in the Stage-2 plan
4. Stage-3 commits listed in the Stage-3 plan
5. Stage-4 commits listed in the Stage-4 plan
6. Stage-5 commits listed in the Stage-5 plan
7. `docs: record reference layer live acceptance`

Never squash evidence-bearing stage commits before final acceptance; the commit chain is part of the audit trail, though it is not by itself runtime acceptance.
