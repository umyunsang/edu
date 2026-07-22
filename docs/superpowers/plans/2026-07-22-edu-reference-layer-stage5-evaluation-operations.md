# Edu Reference Layer Stage 5 Evaluation and Operations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Do not claim acceptance from test counts alone; close every named report and drill.

**Goal:** Prove golden and full-vault correctness, retrieval/citation quality, three-client parity, performance, privacy, retention, recovery, and rollback; then issue one honest live acceptance decision and only conditionally permit local client activation.

**Architecture:** Production evaluators consume immutable Stage 1–4 evidence and recompute metrics rather than trusting run summaries. Operations use sanitized local telemetry, mark-and-stage retention, digest-verified backup/restore, forward-only rollback, and deterministic batch expansion. A final acceptance report is itself an immutable closure; only an authorized publisher may advance a `release_accepted=true` head bound to that report.

**Tech Stack:** Python 3.12 frozen runtime, Fraction/Decimal, PyArrow/SQLite/NumPy, pytest, local CLI probes, POSIX durable file operations, canonical JSON.

## Stage Entry Gate

- Golden Stage 1–4 snapshots and reports verify from fresh processes; every Stage-4 client probe used temporary/nonpersistent configuration.
- Frozen evaluation query/claim/citation goldens are privacy-safe and digest-bound to the exact snapshot. Labels may be corrected only through an explicit reviewed change with prior evidence preserved.
- Full-vault acceptance cannot begin until golden Stage 5 passes. Persistent client activation remains false throughout evaluation and restore/rollback drills.

---

### Task 1: Implement production evaluation and acceptance schemas

**Files:**
- Create: `docs/reference-layer/production-evaluation-spec.md`
- Create: `docs/reference-layer/live-acceptance-spec.md`
- Create: `schemas/reference-layer/v1/stage5-quality-report.schema.json`
- Create: `schemas/reference-layer/v1/stage5-external-report.schema.json`
- Create: `schemas/reference-layer/v1/stage5-performance-report.schema.json`
- Create: `schemas/reference-layer/v1/stage5-client-e2e-report.schema.json`
- Create: `schemas/reference-layer/v1/stage5-operations-report.schema.json`
- Create: `schemas/reference-layer/v1/stage5-telemetry-report.schema.json`
- Create: `schemas/reference-layer/v1/stage5-retention-plan.schema.json`
- Create: `schemas/reference-layer/v1/stage5-backup-manifest.schema.json`
- Create: `schemas/reference-layer/v1/stage5-drill-report.schema.json`
- Create: `schemas/reference-layer/v1/stage5-batch-report.schema.json`
- Create: `schemas/reference-layer/v1/release-acceptance.schema.json`
- Create: `scripts/pkm/reference_evaluation.py`
- Create: `scripts/pkm/reference_acceptance.py`
- Create: `scripts/pkm/tests/test_reference_stage5_contracts.py`
- Create: `scripts/pkm/tests/test_reference_evaluation_runtime.py`
- Create: `scripts/pkm/tests/test_reference_acceptance.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage5/schema-positive.json`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage5/schema-negative.json`

- [ ] Write failing tests for strict schemas, incomplete denominator, stale snapshot/report, illegal eligibility/outcome combination, float token, missing metric, self-reported-only evidence, acceptance true with any failed gate, and activation true without accepted release closure.
- [ ] Include named contract tests `test_successor_schemas_are_strict_closed`, `test_stage0_evaluation_and_release_fixtures_remain_byte_identical`, `test_release_cannot_accept_missing_failed_or_na_required_gate`, `test_every_report_is_bound_to_one_committed_snapshot`, `test_trec_unavailable_forbids_official_score_claim`, and `test_ragperf_is_external_research_and_not_mlcommons`.
- [ ] Assert `evaluation-receipt.schema.json`, `release-manifest.schema.json`, and their Stage-0 fixture bytes remain unchanged. The old release schema is frozen to Stage-1 scope and must not be reused or widened for Stage 5.
- [ ] Move/reimplement the test-only evaluation classifier as production code without weakening the frozen Stage-0 vectors. Reuse exact `Fraction` arithmetic and independent `Decimal(...).quantize(ROUND_HALF_UP)` display oracle.
- [ ] Implement:

```python
@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    eligibility: Literal["eligible", "N/A", "blocked"]
    outcome: Literal["pass", "fail", "not_applicable"]
    evidence_digests: tuple[str, ...]
    measured: Mapping[str, object]

def evaluate_snapshot(snapshot: SnapshotView, evidence: EvidenceSet) -> EvaluationReport: ...
def decide_acceptance(report_set: Sequence[bytes]) -> LiveAcceptanceReport: ...
```

- [ ] Acceptance reports have separate booleans/statuses for `contract_tests`, `implementation_complete`, `golden_accepted`, `full_vault_complete`, `operational_ready`, `client_activation_permitted`, `activation_verified`, and `release_accepted`; no field is inferred from a single aggregate PASS.
- [ ] Final report also fixes `trec_rag_2026_official_claim=false`, `ragperf_mlcommons_official=false`, and requires empty `open_failures`, `quarantined_allowed_assets`, and `unresolved_pdfs` before acceptance.
- [ ] Require every evidence digest to resolve through the pinned snapshot or named external report closure. Missing/indirect evidence is `not_proven`, not pass.
- [ ] Run `uv run --frozen --project scripts/pkm/reference-runtime python -m pytest -q -p no:cacheprovider scripts/pkm/tests/test_reference_stage5_contracts.py scripts/pkm/tests/test_reference_evaluation_runtime.py scripts/pkm/tests/test_reference_acceptance.py`. Expected: zero skip/deselect, PASS with frozen scores unchanged.
- [ ] Commit explicit paths with `feat: add production reference acceptance evaluator`.

### Task 2: Close golden inventory, canonical, PDF, retrieval, and citation quality

**Files:**
- Create: `scripts/pkm/reference_quality.py`
- Create: `scripts/pkm/tests/test_reference_quality.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage5/quality-cases.json`
- External create: `$EDU_REFERENCE_ROOT/reports/stage5/<snapshot_commit_id>/quality-report.json`

- [ ] Write failing tests for each threshold boundary, missing PDF JSON/Markdown, page gap, provenance gap, JSONL/Parquet ID mismatch, non-deterministic rebuild, invalid citation support, and incomplete claim denominator.
- [ ] Recompute and gate: population conservation 100%, accepted source hash binding 100%, required canonical fields 100%, accepted provenance 100%, full-population provenance ≥95%, JSONL/Parquet row+ID parity 100%, repeated canonical digest parity 100%.
- [ ] Recompute OpenDataLoader goldens: born-digital NID≥0.95/TEDS≥0.90/MHS≥0.90; selected complex/scanned ≥0.90/0.85/0.85; page coverage 100%; no required Korean OCR/formula/table omission; every allowed PDF has accepted `document.md` + `document.json` closure.
- [ ] Recompute retrieval/citation: nDCG@10≥0.80, MRR@10≥0.80, Recall@20≥0.90, important-claim citation precision=100%, change/decision citation coverage=100%, exploration evidence quality≥85%, hybrid relative improvement ≥5% on nDCG or Recall with no other core metric regression >2%, ANN Recall@20≥0.98.
- [ ] Treat Stage-3 report values as claims to verify, not inputs to trust. Recompute from raw qrels, ranked runs, citation closures, and latency/resource samples bound to the same snapshot.
- [ ] Validate citations by resolving element content digest/locator/provenance in the pinned snapshot; mere citation-ID presence is insufficient.
- [ ] Run targeted tests and the golden quality evaluator twice. Expected: PASS and byte-identical report digest.
- [ ] Commit tracked code/test/fixture with `feat: close golden reference quality gates`.

### Task 3: Add honest external benchmark evidence

**Files:**
- Create: `docs/reference-layer/external-benchmark-policy.md`
- Create: `scripts/pkm/reference_external_benchmark.py`
- Create: `scripts/pkm/tests/test_reference_external_benchmark.py`
- External create: `$EDU_REFERENCE_ROOT/benchmarks/<benchmark_id>/manifest.json`

- [ ] Write failing tests for unpinned dataset/tool, missing license/provenance, benchmark-to-golden leakage, unavailable judgments claimed as final, RAGPerf mislabeled MLCommons, and metric-name/scale mismatch.
- [ ] Implement `acquire-manifest`, `prepare`, `run`, and `verify` commands. Network acquisition is a separate named action; runs use immutable local staged data and record every artifact digest/license.
- [ ] Treat TREC RAG 2026 topics/dev tools as development/submission-readiness evidence only while official results/judgments remain TBD. `official_final=false` is mandatory until exact official judgments are acquired and verified.
- [ ] Treat RAGPerf as a pinned external research methodology/framework, not an MLCommons official benchmark. Record the paper/code revision and which latency/throughput/quality measures are reused.
- [ ] Never mix external scores into edu golden thresholds. Report each evidence lane separately with scope and date.
- [ ] Run tests. Expected: PASS and honest `unavailable` classification for absent TREC final judgments.
- [ ] Commit explicit tracked paths with `docs: define honest external benchmark evidence`.

### Task 4: Measure local performance, capacity, and sanitized telemetry

**Files:**
- Create: `scripts/pkm/reference_benchmark.py`
- Create: `scripts/pkm/reference_telemetry.py`
- Create: `scripts/pkm/tests/test_reference_benchmark.py`
- Create: `scripts/pkm/tests/test_reference_telemetry.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage5/workloads.json`
- External create: `$EDU_REFERENCE_ROOT/reports/stage5/<snapshot_commit_id>/performance-report.json`

- [ ] Write failing tests for warmup omission, unstable hardware profile, percentile error, binary-float report values, query/log leakage, listener/egress attempt, error-rate denominator drift, and threshold boundaries.
- [ ] Pin hardware/OS/power profile, concurrency, query-set digest, warmup count, measured iteration count, and timing source. Record latency as integer microseconds and RSS/bytes as integers.
- [ ] Gate warm p95 lexical/graph≤250ms, vector≤750ms, hybrid≤1500ms; read/citation≤300ms; preflight≤2000ms; MCP cold start≤8000ms; hybrid p99≤3000ms; search errors<1%; RSS≤4GiB; hybrid throughput≥5qps; hybrid p95≤2× lexical p95.
- [ ] Telemetry records only allowlisted sanitized fields from Stage 4, writes local `0600` JSONL, and rotates by byte/time without losing closure accounting.
- [ ] Predict full-vault processing from measured extraction/index throughput. Require predicted total≤24h and free-space margin≥20%; otherwise optimize or resize the external root and rerun.
- [ ] Run tests and benchmark twice. Expected: tests PASS and report closes; any SLO failure keeps acceptance false and enters root-cause tuning with a new manifest.
- [ ] Commit explicit paths with `feat: benchmark local reference operations`.

### Task 5: Implement quarantine, retention, and recoverable garbage collection

**Files:**
- Create: `docs/reference-layer/operations-policy.md`
- Create: `scripts/pkm/reference_retention.py`
- Create: `scripts/pkm/reference_gc.py`
- Create: `scripts/pkm/reference_quarantine.py`
- Create: `scripts/pkm/tests/test_reference_retention.py`
- Create: `scripts/pkm/tests/test_reference_gc.py`
- Create: `scripts/pkm/tests/test_reference_quarantine.py`

- [ ] Write failing tests for deleting current/previous-two accepted snapshots, citation closure younger than 365 days, selected raw bundle, 30-day quarantine/superseded run, 14-day log, active lease, unverified backup, broad/unresolved path, symlink, and interrupted GC.
- [ ] Encode defaults: accepted current+previous two and at least 90 days; active citation closures at least 365 days; selected raw bundles retained with their snapshot; superseded attempts/quarantine 30 days; logs 14 days.
- [ ] If fewer than three accepted releases exist, retain all of them. Retain current and previous accepted indexes and current+previous-two verified backups for at least 90 days.
- [ ] Implement mark → dry-run report → verify backup+restore drill → same-filesystem move of exact digest-addressed objects to `$EDU_REFERENCE_ROOT/gc/staged/<gc_run_id>/` → seven-day recoverable grace → exact purge. Never recursively target a variable root, glob, vault, home, or workspace root.
- [ ] `reference_retention.py plan|verify` computes reachability and retention. `reference_gc.py dry-run|stage|restore-staged|purge` requires a canonical plan listing every exact path/digest/size plus expected plan/head/restore-report digests and revalidates identity before each move/delete.
- [ ] Quarantine records stable error code, source observation, failed run digest, retry eligibility, and expiry; no raw source content in the record.
- [ ] Run targeted tests including crash at each stage. Expected: PASS and full restoration from staged GC.
- [ ] Commit explicit paths with `feat: add recoverable reference retention`.

### Task 6: Implement backup, restore, crash recovery, and forward rollback

**Files:**
- Create: `scripts/pkm/reference_backup.py`
- Create: `scripts/pkm/reference_recovery.py`
- Create: `scripts/pkm/reference_rollback.py`
- Create: `scripts/pkm/tests/test_reference_backup.py`
- Create: `scripts/pkm/tests/test_reference_recovery.py`
- Create: `scripts/pkm/tests/test_reference_rollback.py`
- External create: `$EDU_REFERENCE_ROOT/reports/stage5/<snapshot_commit_id>/operations-report.json`

- [ ] Write failing tests for partial backup, wrong filesystem identity, digest drift, missing closure, restored permissions, stale temp, interrupted registry publish, restored-client mismatch, manual historical-head edit, and rollback to unverified snapshot.
- [ ] Implement `backup plan|create|verify`, `restore --fresh-root`, `recover`, and `rollback-forward --target-commit-id`. Backup manifest includes every reachable closure, relative object path, digest, size, mode, source snapshot, runtime, and created-at.
- [ ] Restore only to a fresh exact external root; independently hash every object, rebuild verified head selection, and set modes `0700/0600` before use.
- [ ] Recovery scans only documented temp/candidate/quarantine paths, verifies reachability, and never deletes an object referenced by any retained snapshot.
- [ ] Rollback publishes a new CAS commit that selects a previously verified accepted snapshot. It never rewrites or deletes head/commit history.
- [ ] Against the restored root, rerun snapshot verification, adapter integrity, four-tool STDIO smoke, and three-client parity. Expected: identical stable IDs, citations, and verdicts.
- [ ] Run targeted tests and live drills. Expected: PASS and closed operations report.
- [ ] Commit explicit paths with `feat: add reference backup restore and rollback`.

### Task 7: Close identical three-client end-to-end evaluation

**Files:**
- Create: `scripts/pkm/reference_client_e2e.py`
- Create: `scripts/pkm/tests/test_reference_client_e2e.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage5/client-e2e-golden.json`
- External create: `$EDU_REFERENCE_ROOT/reports/stage5/<snapshot_commit_id>/client-e2e-report.json`

- [ ] Write failing tests for different snapshot/tool schema, reordered IDs, content/citation digest drift, preflight mismatch, client omission, config persistence, and prose-only false parity.
- [ ] Run identical change, decision, exploration, and summary tasks through each client in temporary/nonpersistent profiles against both primary and restored roots.
- [ ] Require exact tool inventory/schema hashes, snapshot commit, ordered element/citation IDs, read content digests, preflight action/reasons, stable error codes, and primary/restored parity. Do not compare free-form prose as identity evidence.
- [ ] Confirm real client config before/after digests, no network listener/egress, and clean server shutdown.
- [ ] Run targeted tests and E2E. Expected: all three clients 100% parity and report closure.
- [ ] Commit tracked code/test/fixture with `test: close three-client reference e2e`.

### Task 8: Orchestrate deterministic full-vault expansion

**Files:**
- Create: `scripts/pkm/reference_batches.py`
- Create: `scripts/pkm/tests/test_reference_batches.py`
- External create: `$EDU_REFERENCE_ROOT/control/full-vault-batches.json`
- External create per batch: `$EDU_REFERENCE_ROOT/runs/batches/<batch_id>/manifest.json`

- [ ] Write failing tests for frozen-denominator drift, nondeterministic order, overlap/gap, >100 items, >512MiB batch, oversized item mixed with others, resume double-count, missing/unverified Stage-1B snapshot, source-ID/hash bypass, quarantined-PDF false completion, and wrong pipeline/runtime manifest.
- [ ] Sort accepted allowed assets by canonical path ordering and partition to at most 100 assets or 512MiB; an oversized asset is a one-item batch. `batch_id` is the canonical batch-payload digest.
- [ ] Reuse `reference_asset_walker.path_order_key()` exactly so segment ordering remains `(NFC, raw filesystem bytes)` and repeated plan generation is byte-identical.
- [ ] Implement `plan|verify|run|resume|reconcile`. Before extraction, `run|resume` invokes the Stage-1 `batch-reconcile` contract, verifies its committed Stage-1B snapshot in a fresh process, and passes only that commit to Stage 2. Every batch pins the same accepted golden pipeline/runtime/model/config family and creates a new immutable snapshot chain.
- [ ] After each batch reconcile processed, excluded terminals, quarantined, and remaining against the frozen full-vault denominator. No overlap or gap is permitted.
- [ ] Asset-local failures quarantine and trigger bounded repair/retry; unresolved PDFs and any unproven allowed asset keep `full_vault_complete=false`.
- [ ] Re-run Stage 2–5 gates cumulatively after each batch and full performance/citation/client/restored-root suites at terminal population.
- [ ] Run tests. Expected: PASS.
- [ ] Commit explicit paths with `feat: orchestrate deterministic full-vault expansion`.

### Task 9: Issue and publish the final live acceptance decision

**Files:**
- Create: `scripts/pkm/reference_acceptance_audit.py`
- Create: `scripts/pkm/tests/test_reference_acceptance_audit.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage5/acceptance-evidence.json`
- External create: `$EDU_REFERENCE_ROOT/reports/final-acceptance.json`
- Modify: `.planning/2026-07-22-gjc-reference-layer-codex-execution/{task_plan.md,findings.md,progress.md,.attestation}`

- [ ] From a fresh verifier process, recompute every required gate and evidence digest for the terminal full-vault snapshot. Do not consume the candidate acceptance boolean as evidence.
- [ ] Implement `reference_acceptance_audit.py` without importing the producer verifier. It independently parses raw head/commit/closures/reports/ID sets/metric samples and recomputes the final decision.
- [ ] Require all Stage 1–5 tests, golden/full-vault closure, all allowed PDF dual bundles, zero unresolved PDF quarantine, quality/retrieval/citation thresholds, SLOs, retention/backup/restore/recovery/rollback, and three-client primary/restored parity.
- [ ] Emit a canonical final report with exact failures and `release_accepted=false` if any evidence is missing, stale, indirect, or below threshold. Continue root-cause repair loops until every required gate is proven.
- [ ] When and only when the producer verifier and independent audit agree on all pre-activation gates, close a pre-activation report with `release_accepted=true`, `client_activation_permitted=true`, `activation_verified=false` and publish its accepted inactive snapshot; independently verify head/commit/closures before any client mutation.
- [ ] Render activation commands bound to that exact report and snapshot. First execute disposable three-client activation and live four-tool parity. Only after it passes, execute persistent local activation through the guarded CLI that re-verifies both digests and keeps the exact four-tool allowlist; remote/public activation remains prohibited.
- [ ] Rerun a post-persistent-activation smoke and prove identical stable IDs/citations/preflight across all clients. On failure, remove/disable only the exact created entries and forward-roll back to the last accepted inactive head.
- [ ] On success, produce and independently audit the terminal `final-acceptance.json` with `activation_verified=true`, add the live-smoke/config-before-after evidence closures, and publish the current accepted active head. This second publication is the completion authority.
- [ ] Update file-backed plan evidence and goal only after a requirement-by-requirement audit shows no missing work.
- [ ] Run the complete Stage-5 contract, quality, external, client, performance, telemetry, retention/GC, backup/restore, recovery, rollback, batch, producer-acceptance, and independent-audit test selection with zero skip/deselect; then repeat the live commands against the exact final snapshot.

## Exit Gate

- Final acceptance is true only with exact full-vault, operational, restored-root, and three-client evidence; otherwise the report stays false and the active goal continues.
- GC operations remain recoverable during grace and report exact removed targets/recovery status.
- External benchmark claims retain scope: TREC final remains unavailable until official judgments exist; RAGPerf remains research methodology, not MLCommons.
- Permanent activation, if performed, is local STDIO, exact four tools, pinned release/snapshot, and reversible without source mutation.

## Exact Per-task Test and Commit Commands

Every pytest selection must report zero skip/deselect, and every cache scan must print nothing.

### Task 1 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_stage5_contracts.py \
  scripts/pkm/tests/test_reference_evaluation_runtime.py \
  scripts/pkm/tests/test_reference_acceptance.py \
  scripts/pkm/tests/test_reference_evaluation_contracts.py \
  scripts/pkm/tests/test_reference_release_contracts.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- docs/reference-layer/production-evaluation-spec.md docs/reference-layer/live-acceptance-spec.md \
  schemas/reference-layer/v1/stage5-quality-report.schema.json \
  schemas/reference-layer/v1/stage5-external-report.schema.json \
  schemas/reference-layer/v1/stage5-performance-report.schema.json \
  schemas/reference-layer/v1/stage5-client-e2e-report.schema.json \
  schemas/reference-layer/v1/stage5-operations-report.schema.json \
  schemas/reference-layer/v1/stage5-telemetry-report.schema.json \
  schemas/reference-layer/v1/stage5-retention-plan.schema.json \
  schemas/reference-layer/v1/stage5-backup-manifest.schema.json \
  schemas/reference-layer/v1/stage5-drill-report.schema.json \
  schemas/reference-layer/v1/stage5-batch-report.schema.json \
  schemas/reference-layer/v1/release-acceptance.schema.json \
  scripts/pkm/reference_evaluation.py scripts/pkm/reference_acceptance.py \
  scripts/pkm/tests/test_reference_stage5_contracts.py \
  scripts/pkm/tests/test_reference_evaluation_runtime.py scripts/pkm/tests/test_reference_acceptance.py \
  scripts/pkm/tests/fixtures/reference_layer/stage5/schema-positive.json \
  scripts/pkm/tests/fixtures/reference_layer/stage5/schema-negative.json
git diff --cached --check
git commit -m "docs: define stage 5 acceptance contracts"
```

### Task 2 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider scripts/pkm/tests/test_reference_quality.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_quality.py scripts/pkm/tests/test_reference_quality.py \
  scripts/pkm/tests/fixtures/reference_layer/stage5/quality-cases.json
git diff --cached --check
git commit -m "feat: close golden reference quality gates"
```

### Task 3 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider scripts/pkm/tests/test_reference_external_benchmark.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- docs/reference-layer/external-benchmark-policy.md \
  scripts/pkm/reference_external_benchmark.py scripts/pkm/tests/test_reference_external_benchmark.py
git diff --cached --check
git commit -m "docs: define honest external benchmark evidence"
```

### Task 4 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_benchmark.py scripts/pkm/tests/test_reference_telemetry.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_benchmark.py scripts/pkm/reference_telemetry.py \
  scripts/pkm/tests/test_reference_benchmark.py scripts/pkm/tests/test_reference_telemetry.py \
  scripts/pkm/tests/fixtures/reference_layer/stage5/workloads.json
git diff --cached --check
git commit -m "feat: benchmark local reference operations"
```

### Task 5 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_retention.py scripts/pkm/tests/test_reference_gc.py \
  scripts/pkm/tests/test_reference_quarantine.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- docs/reference-layer/operations-policy.md scripts/pkm/reference_retention.py \
  scripts/pkm/reference_gc.py scripts/pkm/reference_quarantine.py \
  scripts/pkm/tests/test_reference_retention.py scripts/pkm/tests/test_reference_gc.py \
  scripts/pkm/tests/test_reference_quarantine.py
git diff --cached --check
git commit -m "feat: add recoverable reference retention"
```

### Task 6 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_backup.py scripts/pkm/tests/test_reference_recovery.py \
  scripts/pkm/tests/test_reference_rollback.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_backup.py scripts/pkm/reference_recovery.py \
  scripts/pkm/reference_rollback.py scripts/pkm/tests/test_reference_backup.py \
  scripts/pkm/tests/test_reference_recovery.py scripts/pkm/tests/test_reference_rollback.py
git diff --cached --check
git commit -m "feat: add reference backup restore and rollback"
```

### Task 7 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider scripts/pkm/tests/test_reference_client_e2e.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_client_e2e.py scripts/pkm/tests/test_reference_client_e2e.py \
  scripts/pkm/tests/fixtures/reference_layer/stage5/client-e2e-golden.json
git diff --cached --check
git commit -m "test: close three-client reference e2e"
```

### Task 8 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider scripts/pkm/tests/test_reference_batches.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_batches.py scripts/pkm/tests/test_reference_batches.py
git diff --cached --check
git commit -m "feat: orchestrate deterministic full-vault expansion"
```

### Task 9 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_acceptance.py scripts/pkm/tests/test_reference_acceptance_audit.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_acceptance.py scripts/pkm/reference_acceptance_audit.py \
  scripts/pkm/tests/test_reference_acceptance.py scripts/pkm/tests/test_reference_acceptance_audit.py \
  scripts/pkm/tests/fixtures/reference_layer/stage5/acceptance-evidence.json
git diff --cached --check
git commit -m "feat: close reference layer live acceptance"
```

### Full Stage-5 verification command

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_stage5_contracts.py \
  scripts/pkm/tests/test_reference_quality.py scripts/pkm/tests/test_reference_external_benchmark.py \
  scripts/pkm/tests/test_reference_client_e2e.py scripts/pkm/tests/test_reference_benchmark.py \
  scripts/pkm/tests/test_reference_telemetry.py scripts/pkm/tests/test_reference_retention.py \
  scripts/pkm/tests/test_reference_gc.py scripts/pkm/tests/test_reference_quarantine.py \
  scripts/pkm/tests/test_reference_backup.py scripts/pkm/tests/test_reference_recovery.py \
  scripts/pkm/tests/test_reference_rollback.py scripts/pkm/tests/test_reference_batches.py \
  scripts/pkm/tests/test_reference_acceptance.py scripts/pkm/tests/test_reference_acceptance_audit.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
```

Expected: full Stage-5 selection PASS with zero skip/deselect and empty cache scan. The same exact code/runtime then runs against the terminal snapshot and live clients; external reports, registry objects, models, and indexes are never staged.
