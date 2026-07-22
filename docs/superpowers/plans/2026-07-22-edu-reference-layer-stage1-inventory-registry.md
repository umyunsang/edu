# Edu Reference Layer Stage 1 Inventory and Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Track every step with `- [ ]` and keep the active `.planning` files current.

**Goal:** Produce a safe full-vault census, frozen golden selection, persistent logical-source/location/observation authority, and crash-safe external committed snapshot whose closure is independently verifiable.

**Architecture:** Extend the existing no-follow walker and two-pass inventory rather than replacing them. Refactor registry publication so bootstrap, ID allocation, closure creation, commit creation, head CAS, and final verification all occur under one writer lock on one filesystem. Publish the inventory, source registry, selection manifest, and typed snapshot manifest as immutable closures; only the verified head is authoritative and all Stage-1 heads remain `activation=false`.

**Tech Stack:** Python 3.12 standard library, canonical JSON, JSON Schema 2020-12, POSIX `flock`, `openat`/`O_NOFOLLOW`/`O_EXCL`, fsync, pytest.

## Stage Entry Gate

- Approved design authority: base commit `19e221d6159e83f44732347afe10e22e1b0724d4` plus the local-hybrid boundary clarification at `9d665019aa4569e6104269f0d7d4b1d2df8fc1f6`.
- External root is absolute, outside and not an ancestor of the vault, symlink-free, owned by the current user, mode `0700`, and has at least 20% projected free-space margin.
- `$EDU_REFERENCE_ROOT/control/golden-corpus-selection.json` is exact-path, SHA-bound, regular, nlink 1, mode `0600`, and validated against the tracked role policy.
- Standing approval covers census, hashing, external registry publication, and local selection-manifest creation. It does not permit source writes or retired control paths.

---

### Task 1: Amend the inventory and selection contracts

**Files:**
- Modify: `scripts/pkm/reference_layer_config.yaml`
- Modify: `scripts/pkm/reference_layer.local.example.yaml`
- Modify: `docs/reference-layer/inventory-policy.md`
- Create: `docs/reference-layer/source-registry-spec.md`
- Create: `docs/reference-layer/snapshot-manifest-spec.md`
- Create: `scripts/pkm/reference_golden_corpus_policy.yaml`
- Create: `schemas/reference-layer/v1/golden-corpus-selection.schema.json`
- Create: `schemas/reference-layer/v1/source-registry.schema.json`
- Create: `schemas/reference-layer/v1/snapshot-manifest.schema.json`
- Create: `schemas/reference-layer/v1/registry-commit-v1.1.schema.json`
- Create: `schemas/reference-layer/v1/registry-head-v1.1.schema.json`
- Create: `schemas/reference-layer/v1/registry-snapshot-v1.1.schema.json`
- Modify: `scripts/pkm/tests/test_reference_inventory_contracts.py`
- Create: `scripts/pkm/tests/test_reference_stage1_contracts.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage1-contract-positive.json`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage1-contract-negative.json`

- [ ] Add failing tests named `test_stage1_schemas_are_closed_and_strict`, `test_stage0_registry_schemas_and_genesis_vectors_remain_byte_identical`, `test_selection_requires_exact_unique_paths_and_sha256`, `test_role_policy_covers_every_required_media_family`, `test_registry_snapshot_entries_are_typed`, and `test_retired_omo_path_is_not_allowlisted`.
- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_inventory_contracts.py \
  scripts/pkm/tests/test_reference_stage1_contracts.py
```

Expected: new tests fail because `.ipynb`/code suffixes, typed entries, and new schemas do not exist.

- [ ] Extend `suffix_policy.allowed_suffixes` with `.ipynb`, `.py`, `.js`, `.ts`, `.java`, `.c`, `.h`, `.cpp`, `.hpp`, `.cu`, `.sql`, `.sh`, `.toml`, `.jsonl`, and `.parquet`; retain suffix-only, no-sniffing semantics.
- [ ] Remove `.omo` from `contract_artifact_allowlist`; keep `.omo` as a denied boundary and `CONTROL_PREFIXES` exclusion.
- [ ] Change `stage1.approval` to `standing_goal_approval_2026_07_22` and document the exact non-expansion boundary.
- [ ] Define the tracked role policy without personal paths. Required roles are Markdown/wikilinks, born-digital PDF, complex/scanned PDF, notebook, source code, structured data, experiment output, and one quality-fallback case.
- [ ] Define external selection records as exactly `{relative_path, source_sha256, size, roles}` plus top-level `{schema_version, policy_sha256, inventory_id, items, selection_id}`; `selection_id = sha256(canonical_json(payload_without_selection_id))`.
- [ ] Define source-registry records with exact fields `{logical_source_id, location_id, observation_id, source_path, source_sha256, size, observed_at, state}` and `state ∈ {active,moved,copied,quarantined}`. UUID fields use the existing registry UUID contract; paths are relative, NFC-valid, and contain no control prefix.
- [ ] Leave the existing v1.0 registry schemas and genesis fixture bytes unchanged. Define v1.1 commit/head/snapshot successors; typed snapshot entries are `{artifact_type, digest, media_type, size}` sorted by `(artifact_type,digest)`, where `artifact_type` is a closed enum covering inventory, selection, source registry, snapshot manifest, extraction, canonical, adapter, evaluation, and release reports.
- [ ] Re-run the tests. Expected: PASS.
- [ ] Commit explicit paths:

```bash
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_layer_config.yaml scripts/pkm/reference_layer.local.example.yaml \
  scripts/pkm/reference_golden_corpus_policy.yaml docs/reference-layer/inventory-policy.md \
  docs/reference-layer/source-registry-spec.md docs/reference-layer/snapshot-manifest-spec.md \
  schemas/reference-layer/v1/golden-corpus-selection.schema.json \
  schemas/reference-layer/v1/source-registry.schema.json \
  schemas/reference-layer/v1/snapshot-manifest.schema.json \
  schemas/reference-layer/v1/registry-commit-v1.1.schema.json \
  schemas/reference-layer/v1/registry-head-v1.1.schema.json \
  schemas/reference-layer/v1/registry-snapshot-v1.1.schema.json \
  scripts/pkm/tests/test_reference_inventory_contracts.py \
  scripts/pkm/tests/test_reference_stage1_contracts.py \
  scripts/pkm/tests/fixtures/reference_layer/stage1-contract-positive.json \
  scripts/pkm/tests/fixtures/reference_layer/stage1-contract-negative.json
git diff --cached --check
git commit -m "docs: close stage 1 inventory contracts"
```

### Task 2: Make registry publication durably transactional

**Files:**
- Modify: `scripts/pkm/reference_release.py`
- Modify: `scripts/pkm/reference_registry.py`
- Modify: `scripts/pkm/tests/test_reference_registry_runtime.py`
- Create: `scripts/pkm/tests/test_reference_release_runtime.py`

- [ ] Add failing tests for: bootstrap lock precedes directory/marker creation; registry/closures/commits/heads/lock share `st_dev`; genesis head uses final-path no-replace; existing identical durable object is fsynced before reuse; current head cross-fields validate before generation increment; final success rereads and verifies head, commit, and every referenced closure; injected crashes at each fsync/CAS point preserve the previous authority; concurrent publishers yield one CAS winner.
- [ ] Introduce these interfaces:

```python
@dataclass(frozen=True, slots=True)
class DurableCreateResult:
    created: bool
    file_fsynced: bool
    directory_fsynced: bool
    reread_verified: bool

def durable_create(path: Path, raw: bytes, *, mode: int = 0o600) -> DurableCreateResult: ...
def durable_create_final(path: Path, raw: bytes, *, mode: int = 0o600) -> bytes: ...

class Registry:
    def __init__(self, root: Path, *, profile: Literal["1.0", "1.1"] = "1.0") -> None: ...
    def transact(
        self,
        *,
        build_entries: Callable[[tuple[dict[str, object], ...]], list[dict[str, object]]],
        committed_prefix: bytes,
        expected_head_digest: str | None,
    ) -> PublishResult: ...
```

- [ ] Move bootstrap inside `with self._lock()` and split it into `_bootstrap_locked()`; verify mode/type/nlink/owner and `st_dev` for every registry directory and lock descriptor.
- [ ] Preserve default profile `1.0` and its exact genesis payload/envelope/head bytes. The Stage-1 orchestrator explicitly selects profile `1.1`; both profiles share the corrected durability primitives, and cross-profile parents are rejected unless an explicit verified migration entry binds them.
- [ ] Make `publish(entries=...)` delegate to `transact(build_entries=lambda _: entries, ...)` so existing callers remain compatible.
- [ ] Before any new closure, parse and fully verify the current head, commit envelope, snapshot closure, and committed-prefix closure.
- [ ] For genesis, create `heads/active.json` directly with `O_EXCL`; for later generations keep same-directory temp + stale reread + atomic replace.
- [ ] Before `success`, call an internal `_verify_head_bytes(head_raw)` that rereads the final head, named commit, typed snapshot, committed prefix, and all snapshot entries by digest.
- [ ] Instrument tests at syscall boundaries with monkeypatches; do not accept self-reported trace strings as proof.
- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_release_runtime.py \
  scripts/pkm/tests/test_reference_registry_contracts.py \
  scripts/pkm/tests/test_reference_registry_runtime.py
```

Expected: PASS including every crash point and concurrent CAS case.

- [ ] Commit exactly:

```bash
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_release.py scripts/pkm/reference_registry.py \
  scripts/pkm/tests/test_reference_registry_runtime.py \
  scripts/pkm/tests/test_reference_release_runtime.py
git diff --cached --check
git commit -m "fix: make reference registry publication transactional"
```

Expected: cache scan empty, staged diff contains only the four paths, commit succeeds.

### Task 3: Implement persistent source identity reconciliation

**Files:**
- Create: `scripts/pkm/reference_source_registry.py`
- Create: `scripts/pkm/tests/test_reference_source_registry.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/source-registry-positive.json`
- Create: `scripts/pkm/tests/fixtures/reference_layer/source-registry-negative.json`

- [ ] Add failing tests named `test_first_observation_allocates_three_ids_under_writer_lock`, `test_same_path_same_digest_reuses_logical_and_location_ids`, `test_same_logical_source_new_bytes_allocates_observation_only`, `test_rename_copy_and_ambiguity_require_explicit_resolution`, `test_failed_cas_exposes_no_allocated_id`, and `test_repeated_reconciliation_is_byte_deterministic`.
- [ ] Implement:

```python
@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    records: tuple[dict[str, object], ...]
    source_registry_digest: str
    unresolved: tuple[dict[str, object], ...]
    publication: PublishResult

def reconcile_sources(
    registry: Registry,
    inventory: InventoryResult,
    selection: dict[str, object],
    *,
    expected_head_digest: str | None,
    observed_at: str,
    id_factory: Callable[[], UUID] = uuid4,
) -> ReconciliationResult: ...
```

- [ ] Treat exact prior `(source_path,source_sha256)` as the only automatic reuse rule. A new digest at the same path reuses logical/location IDs and gets a new observation ID. Rename/copy candidates are emitted as unresolved and never heuristically merged.
- [ ] Allocate IDs only inside `Registry.transact`'s writer-locked builder. A failed/stale transaction must not publish or return usable IDs.
- [ ] Serialize records in `logical_source_id,location_id,observation_id` order and validate every record before closure publication.
- [ ] Add CLI `reference_source_registry.py reconcile --registry-root ... --inventory ... --selection ... --expected-head-digest ... --observed-at ...` and `verify --registry-root ... --commit-id ...`; errors return exit 65 with sanitized JSON.
- [ ] Run targeted tests and the Stage-1 contract suite. Expected: PASS.
- [ ] Execute and commit exactly:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_source_registry.py \
  scripts/pkm/tests/test_reference_stage1_contracts.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_source_registry.py \
  scripts/pkm/tests/test_reference_source_registry.py \
  scripts/pkm/tests/fixtures/reference_layer/source-registry-positive.json \
  scripts/pkm/tests/fixtures/reference_layer/source-registry-negative.json
git diff --cached --check
git commit -m "feat: add persistent source identity registry"
```

Expected: pytest PASS with zero skip/deselect, cache scan empty, staged diff contains only the four paths, commit succeeds.

### Task 4: Build and verify typed snapshot manifests

**Files:**
- Create: `scripts/pkm/reference_snapshot.py`
- Create: `scripts/pkm/tests/test_reference_snapshot.py`
- Create: `scripts/pkm/tests/test_reference_stage1_batch_reconcile.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/snapshot-manifest-positive.json`
- Create: `scripts/pkm/tests/fixtures/reference_layer/snapshot-manifest-negative.json`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage1-batch-positive.json`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage1-batch-negative.json`
- Modify: `scripts/pkm/reference_contract.py`

- [ ] Add failing tests for missing/extra closure, wrong digest/size/media type, unsorted/duplicate entry, activation true before Stage 5, source head mismatch, and independent closure recomputation.
- [ ] Add failing Stage-1B tests for an asset absent from the frozen inventory, batch overlap/duplicate, pre-open or post-read SHA/size drift, direct Stage-2 inventory-row bypass, source-ID reconciliation outside the writer lock, stale expected head, missing typed selection/source-registry closure, and a candidate that cannot be verified from a fresh process.
- [ ] Implement:

```python
@dataclass(frozen=True, slots=True)
class SnapshotClosure:
    snapshot_commit_id: str
    manifest_digest: str
    head_digest: str
    stage_verified: int
    activation: bool

def build_snapshot_manifest(*, stage: int, activation: bool, entries: Sequence[ArtifactEntry], parents: Sequence[str]) -> bytes: ...
def verify_snapshot(registry_root: Path, *, expected_commit_id: str | None = None) -> SnapshotClosure: ...

@dataclass(frozen=True, slots=True)
class SnapshotView:
    commit_id: str
    snapshot_manifest_sha256: str
    def require_object(self, role: str) -> ClosureObject: ...

def open_committed_snapshot(registry_root: Path, commit_id: str) -> SnapshotView: ...
```

- [ ] Require `activation=false` for Stage 1–4. Require `release_accepted` evaluation closure before Stage-5 activation can be true.
- [ ] Change `orchestrate_inventory()` to publish typed inventory/selection/source-registry/snapshot-manifest entries rather than `entries=[]`; add explicit `selection_path` and `observed_at` arguments.
- [ ] Implement `reference_contract.py batch-reconcile --inventory ... --batch-manifest ... --registry-root ... --expected-head-digest ... --observed-at ...`. It reuses only the frozen full-vault inventory authority, reopens every batch member through no-follow descriptors, recomputes size/SHA before publication, calls the Task-3 source reconciler inside the v1.1 registry transaction, publishes typed batch-selection/source-registry/snapshot closures, and returns only after a fresh verifier process accepts the exact new commit.
- [ ] Require Stage 2 to consume the returned Stage-1B `snapshot_commit_id`; neither an inventory row, batch manifest, uncommitted candidate, nor current-head discovery is a legal substitute.
- [ ] Verify every entry by stable reread, byte count, digest, media type, and schema before returning success.
- [ ] `open_committed_snapshot` accepts only a commit reachable from a verified registry chain, pins exact bytes, and exposes closure roles without filesystem discovery. `require_object` rechecks role, canonical bytes, digest, row count, and sorted-ID-set digest.
- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_snapshot.py \
  scripts/pkm/tests/test_reference_stage1_batch_reconcile.py \
  scripts/pkm/tests/test_reference_source_registry.py \
  scripts/pkm/tests/test_reference_inventory_runtime.py \
  scripts/pkm/tests/test_reference_registry_runtime.py
```

Expected: PASS.

- [ ] Commit exactly:

```bash
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_snapshot.py \
  scripts/pkm/tests/test_reference_snapshot.py \
  scripts/pkm/tests/test_reference_stage1_batch_reconcile.py \
  scripts/pkm/tests/fixtures/reference_layer/snapshot-manifest-positive.json \
  scripts/pkm/tests/fixtures/reference_layer/snapshot-manifest-negative.json \
  scripts/pkm/tests/fixtures/reference_layer/stage1-batch-positive.json \
  scripts/pkm/tests/fixtures/reference_layer/stage1-batch-negative.json \
  scripts/pkm/reference_contract.py
git diff --cached --check
git commit -m "feat: publish typed reference snapshots"
```

Expected: cache scan empty, staged diff contains only the eight paths listed above, commit succeeds.

### Task 5: Freeze and publish the golden Stage-1 snapshot

**Files:**
- External read: `$EDU_REFERENCE_ROOT/control/golden-corpus-selection.json`
- External write: `$EDU_REFERENCE_ROOT/runs/inventory/<inventory_id>/`
- External write: `$EDU_REFERENCE_ROOT/registry/`
- Modify: `.planning/2026-07-22-gjc-reference-layer-codex-execution/progress.md`

- [ ] Validate the external root and selection file before vault traversal; record mode, owner, `st_dev`, free bytes, selection ID, and policy digest.
- [ ] Run the inventory CLI with explicit local config and expected head digest. Expected output has `outcome=pass`, source pre/post digests equal, observed=terminal conservation, and no source write.
- [ ] Recompute every selected file SHA from no-follow descriptors and prove it matches both inventory and selection manifest.
- [ ] Reconcile source IDs, publish typed closures, and verify the selected committed head from a fresh process.
- [ ] Run all Stage-1 tests plus the baseline suite. Expected: PASS; no `.omo` access and no cache artifacts.
- [ ] Independently inspect registry layout and recompute head/commit/payload/closure digests. Require mode `0700` directories, `0600` files, same filesystem, and no temporary residue.
- [ ] Record Stage-1 report fields separately: `contracts_passed`, `runtime_passed`, `population_frozen`, `source_registry_closed`, `stage_verified=1`, `activation=false`.
- [ ] Invoke the Task-4 `reference_contract.py batch-reconcile --inventory ... --batch-manifest ... --registry-root ... --expected-head-digest ... --observed-at ...` CLI on a golden-sized rehearsal batch and independently verify that only its returned Stage-1B commit can enter Stage 2.
- [ ] Run the exact Stage-1 closeout command:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_asset_walker.py \
  scripts/pkm/tests/test_reference_inventory_contracts.py \
  scripts/pkm/tests/test_reference_inventory_runtime.py \
  scripts/pkm/tests/test_reference_release_runtime.py \
  scripts/pkm/tests/test_reference_registry_contracts.py \
  scripts/pkm/tests/test_reference_registry_runtime.py \
  scripts/pkm/tests/test_reference_stage1_contracts.py \
  scripts/pkm/tests/test_reference_stage1_batch_reconcile.py \
  scripts/pkm/tests/test_reference_source_registry.py \
  scripts/pkm/tests/test_reference_snapshot.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
```

Expected: zero skip/deselect, PASS, empty cache scan. Task 5 writes only external closures/reports and updates `.planning`; it creates no Git commit.

## Stage Exit Gate

- Population terminal conservation, selected source hash binding, source-registry schema validity, closure completeness, and repeated verification are 100%.
- Crash and concurrency tests prove the previous head survives every pre-CAS failure and one winner owns each CAS.
- Golden selection covers every required role and contains no control path.
- The committed head is verified from a fresh process; no unpublished candidate is treated as authority.
- Stage 2 may start from this exact `snapshot_commit_id`; client activation remains false.
- Full-vault expansion must repeat Stage 1B for every non-golden batch; no asset may enter Stage 2 from an inventory row alone.
