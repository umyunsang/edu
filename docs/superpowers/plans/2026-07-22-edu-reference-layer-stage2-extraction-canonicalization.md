# Edu Reference Layer Stage 2 Extraction and Canonicalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Follow TDD, preserve immutable sources, and track every step with checkboxes.

**Goal:** Convert the frozen Stage-1 corpus into immutable raw extraction bundles and deterministic canonical source-observation, element, authored-relation, and experiment-lineage authorities with Parquet parity.

**Architecture:** A no-follow source reader opens only Stage-1 selected observations. Media-specific extractors write fresh external run candidates; validators close them before publication. Every accepted PDF is processed by OpenDataLoader PDF 2.5.0 into both Markdown and structural JSON. Canonicalizers allocate no source IDs, produce strict sorted JSONL, derive path-independent evidence IDs, resolve authored links deterministically, and write digest-bound Parquet projections. Failures quarantine individual assets unless they invalidate snapshot identity or closure integrity.

**Tech Stack:** Python 3.12/uv, Java 21 LTS, OpenDataLoader PDF 2.5.0, PyArrow 25.0.0, Python `ast`/`tokenize`/`json`, canonical JSON/JSONL, JSON Schema 2020-12, pytest.

## Stage Entry Gate

- Runtime-lock Task 1 is the bootstrap exception and executes before Stage 1. Tasks 2–9 require all remaining entry gates below.
- Stage-1 snapshot, inventory, selection, and source registry verify from a fresh process and are pinned by exact `snapshot_commit_id`.
- External run root and model/runtime roots are symlink-free, same-owner, mode `0700`; source-derived files are `0600`.
- Every source open is selected by `(observation_id, source_path, source_sha256, size)` and revalidated through an `openat`/no-follow descriptor before and after read.
- Dependency acquisition may use network only in its named acquisition task. Extraction itself runs with egress denied. Remote hybrid services are prohibited.

---

### Task 1: Lock and attest the isolated extraction runtime

**Files:**
- Create: `scripts/pkm/reference-runtime/pyproject.toml`
- Create: `scripts/pkm/reference-runtime/uv.lock`
- Create: `scripts/pkm/reference-runtime/README.md`
- Create: `schemas/reference-layer/v1/runtime-lock.schema.json`
- Create: `scripts/pkm/reference_runtime.py`
- Create: `scripts/pkm/tests/test_reference_runtime.py`

- [ ] Write failing tests for wrong Python minor, floating/unlocked dependency, missing artifact hash, Java absence/version below 11, changed Java binary, egress-enabled execution profile, and runtime-manifest digest mismatch.
- [ ] Set project Python to `>=3.12,<3.13`. Pin direct packages `opendataloader-pdf==2.5.0`, `pyarrow==25.0.0`, `mcp[cli]==1.28.1`, and `lancedb==0.34.0`; add exact direct test/schema/model dependencies and let `uv.lock` freeze every transitive artifact.
- [ ] Pin the OpenDataLoader 2.5.0 wheel SHA-256 `0f415c75bafe824393276ac7814d53858190690e2ab1fce69944ff42f63535b0`, sdist SHA-256 `6e77f876f90ee45c67168c8d79f39bd9ff8c5c85a885ae879a2e6abc0aaba878`, and upstream commit `2bd7466d4742491b05920483bdf2ea7395444a16`; verify the installed distribution and bundled executable JAR separately.
- [ ] Use Java 21 LTS. Record vendor, full version output, resolved executable SHA-256, architecture, install receipt, and license in the runtime manifest.
- [ ] Implement:

```python
@dataclass(frozen=True, slots=True)
class RuntimeManifest:
    manifest_id: str
    python: Mapping[str, object]
    uv: Mapping[str, object]
    java: Mapping[str, object]
    packages: tuple[Mapping[str, object], ...]
    platform: Mapping[str, object]
    network_policy: str

def collect_runtime(project_root: Path, *, java_executable: Path) -> RuntimeManifest: ...
def verify_runtime(raw: bytes, *, require_offline: bool) -> RuntimeManifest: ...
```

- [ ] `manifest_id` is the canonical payload digest; package entries include name, version, source URL/index identity, and artifact SHA-256 from the lock/installed distribution.
- [ ] After creating `pyproject.toml`, run `uv lock --project scripts/pkm/reference-runtime --python 3.12` once to create the reviewed lockfile. Then run `uv sync --frozen --project scripts/pkm/reference-runtime` and the runtime tests. Expected: PASS and import smoke for OpenDataLoader/PyArrow/MCP/LanceDB.
- [ ] Run an OpenDataLoader synthetic one-page PDF smoke in a temporary external directory with `format="markdown,json"`; expected both output types and zero source mutation. This is dependency evidence, not PDF quality acceptance.
- [ ] Commit exact runtime files/code/test/schema with `build: lock reference layer runtime`.

### Task 2: Define Stage-2 canonical and run contracts

**Files:**
- Modify: `docs/reference-layer/canonical-spec.md`
- Create: `docs/reference-layer/extraction-run-spec.md`
- Create: `docs/reference-layer/authored-relation-spec.md`
- Create: `docs/reference-layer/experiment-lineage-spec.md`
- Create: `schemas/reference-layer/v1/canonical-record-v1.1.schema.json`
- Create: `schemas/reference-layer/v1/parquet-projection-v1.1.schema.json`
- Create: `schemas/reference-layer/v1/extraction-run-manifest.schema.json`
- Create: `schemas/reference-layer/v1/pdf-run-manifest.schema.json`
- Create: `schemas/reference-layer/v1/source-observation.schema.json`
- Create: `schemas/reference-layer/v1/authored-relation.schema.json`
- Create: `schemas/reference-layer/v1/lineage-edge.schema.json`
- Create: `schemas/reference-layer/v1/quarantine-record.schema.json`
- Create: `schemas/reference-layer/v1/canonical-stream-manifest.schema.json`
- Create: `scripts/pkm/tests/test_reference_stage2_contracts.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage2/{contracts-positive.json,contracts-negative.json}`

- [ ] Write failing tests for the two new element kinds, invalid locator-kind pairs, relation/lineage field leakage, missing PDF dual output, unsafe raw path, unpinned runtime/options, unordered outputs, Parquet writer drift, and byte/hash drift in the frozen v1.0 canonical/Parquet schemas and fixtures.
- [ ] Leave `canonical-record.schema.json`, `parquet-projection.schema.json`, and their Stage-0 vectors byte-identical. Create v1.1 successors whose kinds are exactly `markdown_section`, `pdf_block`, `notebook_cell`, `source_code_unit`, and `experiment_result` with reviewed locator-major v1 patterns. Add no absolute path, URI, backslash, or `..` allowance.
- [ ] Define `source-observations.jsonl`, `elements.jsonl`, `relations.jsonl`, and `lineage.jsonl` as four separate strict authorities. Each record is canonical JSON on one LF-terminated line, unique by its domain ID, and globally sorted by that ID.
- [ ] Define domain-separated IDs:

```text
element_id      = SHA256("edu-reference-element-id-v1\0"      || canonical(identity_material))
relation_id     = SHA256("edu-reference-relation-id-v1\0"     || canonical(relation_material))
lineage_edge_id = SHA256("edu-reference-lineage-edge-id-v1\0" || canonical(lineage_material))
```

- [ ] Keep parser version/path/observation time outside element identity. Include logical source, kind, locator major, locator, and content digest in identity material.
- [ ] Relation predicates are the approved explicit Obsidian fields plus `wikilink` and `embed`; resolution status is `resolved|unresolved|ambiguous|denied`, with nullable target and exact source citation.
- [ ] Lineage predicates are exactly `uses|generated_by|derived_from|executed_with|measures`; lineage never substitutes for authored graph semantics.
- [ ] The v1.1 Parquet successor uses exact PyArrow version/options; do not replace the frozen v1.0 `synthetic-reference-writer` vector.
- [ ] Run contract tests. Expected: PASS.
- [ ] Commit explicit paths with `docs: close stage 2 canonical contracts`.

### Task 3: Implement selected-observation reading and immutable run closure

**Files:**
- Create: `scripts/pkm/reference_source_reader.py`
- Create: `scripts/pkm/reference_extract.py`
- Create: `scripts/pkm/tests/test_reference_source_reader.py`
- Create: `scripts/pkm/tests/test_reference_extract.py`

- [ ] Write failing tests for path traversal, symlink/hardlink substitution, inode/size/mtime drift, digest mismatch, unselected observation, temp residue, output collision, manifest mismatch, crash-before-close, and quarantine isolation.
- [ ] Implement:

```python
@dataclass(frozen=True, slots=True)
class SelectedObservation:
    logical_source_id: str
    location_id: str
    observation_id: str
    relative_path: str
    source_sha256: str
    size: int

@contextmanager
def open_selected(root_fd: int, selected: SelectedObservation) -> Iterator[BinaryIO]: ...

class Extractor(Protocol):
    media_family: str
    def extract(self, source: SelectedObservation, source_fd: int, candidate_dir: Path) -> RawRun: ...

def close_run(candidate_dir: Path, manifest: Mapping[str, object], final_dir: Path) -> str: ...

def stage_selected_source(
    *, vault_root: Path, inventory: InventoryResult,
    source_record: Mapping[str, object], destination: Path,
) -> StagedSource: ...
```

- [ ] `open_selected` walks parents using dir FDs and `O_NOFOLLOW`, verifies regular+nlink1 and exact Stage-1 identity, hashes through the descriptor, then rechecks `fstat` before return.
- [ ] Copy selected bytes through that verified descriptor into a fresh external `0600` staged file and digest-verify the copy. Media tools receive only staged external paths, never the live vault path. Recheck the live source identity/digest after extraction.
- [ ] Each run uses `$EDU_REFERENCE_ROOT/runs/extraction/<observation_id>/<run_id>/`; `run_id` is the canonical manifest-payload digest. Candidate directories are fresh, never overwrite, and become immutable only after all output digests validate.
- [ ] `reference_extract.py run|verify|batch` routes by declared suffix only and returns sanitized JSON. Batch input is an exact manifest, not a glob.
- [ ] Run targeted tests. Expected: PASS.
- [ ] Commit explicit paths with `feat: add immutable extraction run orchestrator`.

### Task 4: Implement Markdown extraction and authored-source capture

**Files:**
- Create: `scripts/pkm/reference_extract_markdown.py`
- Create: `scripts/pkm/tests/test_reference_extract_markdown.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage2/markdown/{representative.md,expected-elements.jsonl,expected-links.jsonl}`

- [ ] Write failing tests for YAML frontmatter boundaries, repeated headings, Korean headings, code fences, inline links, wikilinks, embeds, aliases, fragments, explicit graph fields, and deterministic locators.
- [ ] Emit sections using locator `md:v1:heading=<escaped-heading>:occurrence=<n>:anchor=<content-anchor>`; preamble is an explicit stable pseudo-heading. Preserve source bytes/text semantics; do not normalize the vault file.
- [ ] Capture body wikilinks/embeds and approved frontmatter/inline fields as unresolved authored-relation candidates with exact source span and element citation.
- [ ] Run targeted tests. Expected: PASS and repeated extraction byte identity.
- [ ] Commit explicit paths with `feat: extract markdown evidence and authored links`.

### Task 5: Materialize every PDF through OpenDataLoader Markdown+JSON

**Files:**
- Create: `scripts/pkm/reference_extract_pdf.py`
- Create: `scripts/pkm/reference_odl_sidecar.py`
- Create: `scripts/pkm/reference_pdf_quality.py`
- Create: `scripts/pkm/tests/test_reference_extract_pdf.py`
- Create: `scripts/pkm/tests/test_reference_odl_sidecar.py`
- Create: `scripts/pkm/tests/test_reference_pdf_quality.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage2/pdf/{born-digital.pdf,complex.pdf,scanned.pdf,golden.json}`

- [ ] Write failing tests with a fake converter for exact `format="markdown,json"`, output normalization, missing Markdown, missing JSON, unsafe asset path, source mutation, runtime/options drift, local→hybrid routing, `0.0.0.0`/remote URL rejection, sidecar cleanup after success/error/cancel, extraction-time download/egress, and immutable retry/supersession.
- [ ] Call the pinned Python API once per bounded batch:

```python
opendataloader_pdf.convert(
    input_path=[str(path) for path in verified_inputs],
    output_dir=str(candidate_output),
    format="markdown,json",
    image_output="external",
    image_format="png",
    image_dir=str(candidate_output / "assets"),
    reading_order="xycut",
    table_method="default",
    threads="1",
    content_safety_off=None,
    hybrid=None,
    hybrid_fallback=False,
    quiet=True,
)
```

- [ ] Normalize each successful isolated output to this immutable bundle:

```text
<run_id>/
  document.md
  document.json
  assets/
  manifest.json
```

`assets/` is required when extracted media exist. Both `document.md` and `document.json` are always required; neither may substitute for the other.

- [ ] Manifest binds source SHA/observation, OpenDataLoader 2.5.0 distribution/JAR digest, Java/runtime ID, all flags, local/hybrid route, OCR languages, output names/digests/bytes, pages/elements/assets, NID/TEDS/MHS when golden truth exists, structural omissions, selected/superseded status, and egress=`denied`.
- [ ] First run deterministic local mode. Route only structural-low-quality, complex, or scanned cases to a local `docling-fast` hybrid server whose model artifacts were acquired and sealed separately; OCR uses `ko,en`.
- [ ] `reference_odl_sidecar.py` pre-binds an OS-assigned `127.0.0.1:0` socket, starts the pinned app only for one batch, rejects `0.0.0.0`, remote URL, fixed public port, proxy/API-key environment, model download, and outbound connections, then terminates and proves the listener is gone on success, error, timeout, or cancellation. `hybrid_fallback=False` prevents implicit selection of the fast run.
- [ ] Do not combine struct-tree precedence with hybrid. Formula/picture enrichment uses full hybrid only when the golden role requires it and the local model manifest authorizes it.
- [ ] Golden gates: born-digital NID≥0.95, TEDS≥0.90, MHS≥0.90; selected complex/scanned NID≥0.90, TEDS≥0.85, MHS≥0.85; page coverage 100% and no Korean OCR/formula/table golden omissions.
- [ ] Non-golden PDFs use structural omission/page/empty-text/table/image heuristics for routing, but are not assigned synthetic official metric scores.
- [ ] A failed run remains immutable and superseded; a corrected attempt gets a new run ID. Unresolved PDF quarantine blocks Stage-2/full completion.
- [ ] Run tests, then run the three golden PDFs in the frozen offline runtime. Expected: dual bundles close, routes match golden, metrics pass.
- [ ] Commit code/tests/fixtures with `feat: materialize pdf markdown and provenance json`.

### Task 6: Extract notebook, code, data, and experiment evidence

**Files:**
- Create: `scripts/pkm/reference_extract_notebook.py`
- Create: `scripts/pkm/reference_extract_code.py`
- Create: `scripts/pkm/reference_extract_data.py`
- Create: `scripts/pkm/reference_extract_experiment.py`
- Create: `scripts/pkm/tests/test_reference_extract_notebook.py`
- Create: `scripts/pkm/tests/test_reference_extract_code.py`
- Create: `scripts/pkm/tests/test_reference_extract_data.py`
- Create: `scripts/pkm/tests/test_reference_extract_experiment.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage2/{notebook,code,data}/`

- [ ] Notebook tests cover stable cell IDs, missing cell ID, source/output separation, execution count, attachments, error output, secrets redaction from logs, and no execution. Derive a UUIDv5 cell locator from logical source ID plus existing cell ID; when absent, use content-anchor plus occurrence without writing back.
- [ ] Code tests cover Python AST symbols, comments/docstrings, syntax-error fallback, line spans, Java/JS/SQL/CUDA representative units, and deterministic symbol/occurrence locators. Extraction parses only; it never executes code.
- [ ] Data tests cover JSON/JSONL/CSV/YAML/TOML/Parquet summaries, schema/column/dtype/row-count evidence, large-file bounded reads, nonfinite numbers, and no raw row leakage into logs. Experiment tests require the Stage-1 role/config mapping and reject filename-only inference.
- [ ] Emit `source_code_unit` and `experiment_result` raw records plus lineage candidates. Record datasets, parameters, environment references, outputs, and metrics only when source evidence explicitly supports them; missing links remain unresolved.
- [ ] Run the three targeted suites. Expected: PASS and deterministic byte output.
- [ ] Commit explicit paths with `feat: extract notebook code and experiment evidence`.

### Task 7: Canonicalize elements, authored relations, and lineage

**Files:**
- Create: `scripts/pkm/reference_canonical.py`
- Create: `scripts/pkm/reference_elements.py`
- Create: `scripts/pkm/reference_relations.py`
- Create: `scripts/pkm/reference_lineage.py`
- Create: `scripts/pkm/tests/test_reference_canonical_runtime.py`
- Create: `scripts/pkm/tests/test_reference_relations.py`
- Create: `scripts/pkm/tests/test_reference_lineage.py`

- [ ] Write failing tests for ID vectors, duplicate IDs, parser/path independence, locator-major change, resolution order, ambiguous stem/alias, denied target, relation/lineage conflation, missing citation, and repeat-build digest parity.
- [ ] Centralize strict duplicate-key/no-float canonical parsing and LF JSONL writing in `reference_canonical.py`; have existing registry code reuse it without changing frozen semantic vectors.
- [ ] Resolve authored targets in exact order: source-relative exact path, vault-root exact path, unique stem, unique alias. Any non-unique candidate is `ambiguous`; control-boundary target is `denied`; no match is `unresolved`.
- [ ] Canonicalize PDF JSON page/order/bbox/type records into `pdf_block` locators and cross-check every block's page against the bundle manifest and Markdown page coverage.
- [ ] Implement CLIs `reference_elements.py build|verify`, `reference_relations.py build|verify`, and `reference_lineage.py build|verify` pinned to one Stage-1 snapshot and selected run IDs.
- [ ] Run targeted plus existing canonical-contract tests. Expected: PASS.
- [ ] Commit explicit paths with `feat: build canonical evidence authorities`.

### Task 8: Write deterministic Parquet projections

**Files:**
- Create: `scripts/pkm/reference_parquet.py`
- Create: `scripts/pkm/tests/test_reference_parquet.py`
- Create: `scripts/pkm/tests/test_reference_stage2_integration.py`

- [ ] Write failing tests for field/row ordering, duplicate ID, schema drift, writer version/options drift, row/ID mismatch, corrupt footer, and two-build byte/digest parity.
- [ ] Use PyArrow 25.0.0 with explicit field schema, ID-ascending rows, row-group size 65536, compression none, dictionary encoding false, statistics false, and a fresh output path. Record Arrow schema serialization digest and writer/runtime ID.
- [ ] Implement `write_projection(authority_jsonl, output_path, *, id_pointer, schema)` and `verify_projection(jsonl, parquet, manifest)`; verify row count, sorted ID-set digest, required-field shape, and source/projection digests.
- [ ] Do not use floating aggregation for identity or equality gates.
- [ ] Run targeted tests. Expected: PASS and exact repeated-build digest parity on the frozen runtime/hardware profile.
- [ ] Commit explicit paths with `feat: add deterministic parquet projections`.

### Task 9: Close and publish the golden Stage-2 snapshot

**Files:**
- External create: `$EDU_REFERENCE_ROOT/canonical/<stage2_snapshot_id>/`
- External create: `$EDU_REFERENCE_ROOT/reports/stage2/<stage2_snapshot_id>/stage2-report.json`
- Modify: `.planning/2026-07-22-gjc-reference-layer-codex-execution/progress.md`

- [ ] Run extraction for every frozen golden item; validate and close each raw run before canonicalization.
- [ ] Require every selected PDF to have one accepted OpenDataLoader dual bundle; unresolved PDF quarantine fails the stage.
- [ ] Build all four canonical JSONL authorities and Parquet projections twice in fresh directories; require canonical digest and JSONL/Parquet row+ID-set parity 100%.
- [ ] Require accepted element provenance 100%, required fields 100%, authored-link resolution explicitly classified 100%, and lineage edges source-supported.
- [ ] Run the complete Stage-2 suite under `uv run --frozen`; expected zero skip/deselect and PASS.
- [ ] The integration suite must prove selected-population accepted/quarantined conservation 100%, source pre/post digest identity, no Stage-1 head advancement on any failed/quarantined run, and fresh-process verification.
- [ ] Publish only typed extraction/canonical/projection/report closures in a Stage-2 `activation=false` snapshot and verify it from a fresh process.

## Rollback and Exit Gate

- Raw and canonical candidates are immutable and unreferenced until verified; failures never replace the authoritative head.
- Asset-local failures may quarantine and continue diagnosis, but no unresolved golden/PDF quarantine can pass the stage.
- Source drift, snapshot mismatch, dual-output loss, canonical duplicate, or closure mismatch is run-fatal.
- Stage-3 entry requires exact four-authority closure, JSONL/Parquet parity, accepted PDF dual bundles, runtime lock, and Stage-2 report digest. Client activation remains false.

## Exact Per-task Test and Commit Commands

Run each block only after its failing test has been observed and the minimal implementation is ready. Every cache scan must print nothing; every pytest selection must report zero skip/deselect.

### Task 1 commands

```bash
uv lock --project scripts/pkm/reference-runtime --python 3.12
uv sync --frozen --project scripts/pkm/reference-runtime
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider scripts/pkm/tests/test_reference_runtime.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference-runtime/pyproject.toml \
  scripts/pkm/reference-runtime/uv.lock scripts/pkm/reference-runtime/README.md \
  schemas/reference-layer/v1/runtime-lock.schema.json \
  scripts/pkm/reference_runtime.py scripts/pkm/tests/test_reference_runtime.py
git diff --cached --check
git commit -m "build: lock reference layer runtime"
```

### Task 2 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_stage2_contracts.py \
  scripts/pkm/tests/test_reference_canonical_contracts.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- docs/reference-layer/canonical-spec.md docs/reference-layer/extraction-run-spec.md \
  docs/reference-layer/authored-relation-spec.md docs/reference-layer/experiment-lineage-spec.md \
  schemas/reference-layer/v1/canonical-record-v1.1.schema.json \
  schemas/reference-layer/v1/parquet-projection-v1.1.schema.json \
  schemas/reference-layer/v1/extraction-run-manifest.schema.json \
  schemas/reference-layer/v1/pdf-run-manifest.schema.json \
  schemas/reference-layer/v1/source-observation.schema.json \
  schemas/reference-layer/v1/authored-relation.schema.json \
  schemas/reference-layer/v1/lineage-edge.schema.json \
  schemas/reference-layer/v1/quarantine-record.schema.json \
  schemas/reference-layer/v1/canonical-stream-manifest.schema.json \
  scripts/pkm/tests/test_reference_stage2_contracts.py \
  scripts/pkm/tests/fixtures/reference_layer/stage2/contracts-positive.json \
  scripts/pkm/tests/fixtures/reference_layer/stage2/contracts-negative.json
git diff --cached --check
git commit -m "docs: close stage 2 canonical contracts"
```

### Task 3 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_source_reader.py scripts/pkm/tests/test_reference_extract.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_source_reader.py scripts/pkm/reference_extract.py \
  scripts/pkm/tests/test_reference_source_reader.py scripts/pkm/tests/test_reference_extract.py
git diff --cached --check
git commit -m "feat: add immutable extraction run orchestrator"
```

### Task 4 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider scripts/pkm/tests/test_reference_extract_markdown.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_extract_markdown.py \
  scripts/pkm/tests/test_reference_extract_markdown.py \
  scripts/pkm/tests/fixtures/reference_layer/stage2/markdown
git diff --cached --check
git commit -m "feat: extract markdown evidence and authored links"
```

### Task 5 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_extract_pdf.py \
  scripts/pkm/tests/test_reference_odl_sidecar.py \
  scripts/pkm/tests/test_reference_pdf_quality.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_extract_pdf.py scripts/pkm/reference_odl_sidecar.py \
  scripts/pkm/reference_pdf_quality.py scripts/pkm/tests/test_reference_extract_pdf.py \
  scripts/pkm/tests/test_reference_odl_sidecar.py scripts/pkm/tests/test_reference_pdf_quality.py \
  scripts/pkm/tests/fixtures/reference_layer/stage2/pdf
git diff --cached --check
git commit -m "feat: materialize pdf markdown and provenance json"
```

### Task 6 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_extract_notebook.py \
  scripts/pkm/tests/test_reference_extract_code.py \
  scripts/pkm/tests/test_reference_extract_data.py \
  scripts/pkm/tests/test_reference_extract_experiment.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_extract_notebook.py scripts/pkm/reference_extract_code.py \
  scripts/pkm/reference_extract_data.py scripts/pkm/reference_extract_experiment.py \
  scripts/pkm/tests/test_reference_extract_notebook.py scripts/pkm/tests/test_reference_extract_code.py \
  scripts/pkm/tests/test_reference_extract_data.py scripts/pkm/tests/test_reference_extract_experiment.py \
  scripts/pkm/tests/fixtures/reference_layer/stage2/notebook \
  scripts/pkm/tests/fixtures/reference_layer/stage2/code \
  scripts/pkm/tests/fixtures/reference_layer/stage2/data
git diff --cached --check
git commit -m "feat: extract notebook code and experiment evidence"
```

### Task 7 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_canonical_runtime.py \
  scripts/pkm/tests/test_reference_relations.py scripts/pkm/tests/test_reference_lineage.py \
  scripts/pkm/tests/test_reference_canonical_contracts.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_canonical.py scripts/pkm/reference_elements.py \
  scripts/pkm/reference_relations.py scripts/pkm/reference_lineage.py \
  scripts/pkm/tests/test_reference_canonical_runtime.py \
  scripts/pkm/tests/test_reference_relations.py scripts/pkm/tests/test_reference_lineage.py
git diff --cached --check
git commit -m "feat: build canonical evidence authorities"
```

### Task 8 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_parquet.py \
  scripts/pkm/tests/test_reference_stage2_integration.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_parquet.py scripts/pkm/tests/test_reference_parquet.py \
  scripts/pkm/tests/test_reference_stage2_integration.py
git diff --cached --check
git commit -m "feat: add deterministic parquet projections"
```

### Task 9 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_runtime.py scripts/pkm/tests/test_reference_stage2_contracts.py \
  scripts/pkm/tests/test_reference_source_reader.py scripts/pkm/tests/test_reference_extract.py \
  scripts/pkm/tests/test_reference_extract_markdown.py scripts/pkm/tests/test_reference_extract_pdf.py \
  scripts/pkm/tests/test_reference_odl_sidecar.py scripts/pkm/tests/test_reference_pdf_quality.py \
  scripts/pkm/tests/test_reference_extract_notebook.py scripts/pkm/tests/test_reference_extract_code.py \
  scripts/pkm/tests/test_reference_extract_data.py scripts/pkm/tests/test_reference_extract_experiment.py \
  scripts/pkm/tests/test_reference_canonical_runtime.py scripts/pkm/tests/test_reference_relations.py \
  scripts/pkm/tests/test_reference_lineage.py scripts/pkm/tests/test_reference_parquet.py \
  scripts/pkm/tests/test_reference_stage2_integration.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
```

Expected: full Stage-2 selection PASS, zero skip/deselect, empty cache scan; Task 9 creates external closures/reports only and makes no Git commit.
