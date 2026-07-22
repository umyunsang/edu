# Edu Reference Layer Stage 3 Search and Graph Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Track every step with checkboxes and do not mutate the source vault.

**Goal:** Build snapshot-pinned, backend-neutral lexical, graph, vector, and hybrid retrieval adapters that preserve canonical IDs/citations and pass strong quality and local-performance gates.

**Architecture:** Open one committed Stage-2 snapshot through a verified closure API; build deterministic SQLite FTS5 and adjacency references, a NumPy exact-cosine oracle, and a gated LanceDB scale candidate. Select BGE-M3 or multilingual-E5 from a frozen local bake-off. Fuse lexical/vector ranks with exact RRF while keeping graph results separate. Adapters never allocate canonical IDs or interpret new source relations.

**Tech Stack:** Python 3.12 frozen runtime, SQLite 3 with FTS5, NumPy float32/float64, Hugging Face local model artifacts, LanceDB 0.34.0, canonical JSON, pytest.

## Stage Entry Gate

- Stage-2 committed snapshot exposes verified roles `elements_jsonl`, `relations_jsonl`, `lineage_jsonl`, and `snapshot_manifest` through `open_committed_snapshot(registry_root, commit_id)`.
- `SnapshotView.require_object(role)` verifies canonical bytes, digest, role, row count, and sorted-ID-set digest; missing or wrong closures fail with `SNAPSHOT_CLOSURE_INCOMPLETE` or `CLOSURE_DIGEST_MISMATCH`.
- Frozen runtime and model acquisition are local; Stage-3 execution performs no network access.

---

### Task 1: Define adapter, citation, and benchmark contracts

**Files:**
- Create: `docs/reference-layer/adapter-contract.md`
- Create: `docs/reference-layer/retrieval-benchmark.md`
- Create: `scripts/pkm/reference_adapter_config.yaml`
- Create: `scripts/pkm/reference_adapters.py`
- Create: `schemas/reference-layer/v1/adapter-index-manifest.schema.json`
- Create: `schemas/reference-layer/v1/embedding-model-manifest.schema.json`
- Create: `schemas/reference-layer/v1/retrieval-query-set.schema.json`
- Create: `schemas/reference-layer/v1/retrieval-result.schema.json`
- Create: `schemas/reference-layer/v1/retrieval-report.schema.json`
- Create: `scripts/pkm/tests/test_reference_adapter_contracts.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage3/{elements.jsonl,relations.jsonl,lineage.jsonl,schema-positive.json,schema-negative.json}`

- [ ] Write failing tests for strict closed schemas, duplicate/NFC-colliding IDs, fixed citation-ID vectors, backend-native row-ID rejection, and mixed snapshot pins.
- [ ] Implement exact DTOs and protocols:

```python
@dataclass(frozen=True, slots=True)
class SnapshotPin:
    commit_id: str
    manifest_sha256: str

@dataclass(frozen=True, slots=True)
class CitationRef:
    citation_id: str
    resource_uri: str
    logical_source_id: str
    observation_id: str
    element_id: str
    canonical_locator: str
    content_sha256: str
    snapshot: SnapshotPin

@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    limit: int
    snapshot: SnapshotPin

@dataclass(frozen=True, slots=True)
class RankedHit:
    element_id: str
    rank: int
    citation: CitationRef
    channel_ranks: tuple[tuple[str, int], ...]

class RankedAdapter(Protocol):
    snapshot: SnapshotPin
    manifest_sha256: str
    def search(self, request: SearchRequest) -> tuple[RankedHit, ...]: ...
```

- [ ] Define `GraphRequest`, `GraphHit`, and `GraphAdapter.expand()` with hops 1/2, direction `out|in|both`, predicate/status filters, a bounded limit, and the same snapshot pin.
- [ ] Compute `citation_id` from domain `edu-reference-citation-id-v1\0` plus canonical JSON of logical source, observation, element, locator, and content digest. Exclude the snapshot from citation identity but include the pin in every response. URI is exactly `edu-ref://resource/<logical_source_id>`.
- [ ] Define stable errors: `SNAPSHOT_NOT_COMMITTED`, `SNAPSHOT_MISMATCH`, `SNAPSHOT_CLOSURE_INCOMPLETE`, `CLOSURE_DIGEST_MISMATCH`, `ADAPTER_INPUT_MISMATCH`, `FTS5_UNAVAILABLE`, `INDEX_INTEGRITY_FAILED`, `QUERY_EMPTY`, `VECTOR_NONFINITE`, `VECTOR_ZERO_NORM`, `MODEL_MANIFEST_MISMATCH`, `MODEL_GATE_FAILED`, `LANCEDB_UNAVAILABLE`, `ANN_RECALL_GATE_FAILED`, `HYBRID_CHANNEL_UNAVAILABLE`, `CITATION_MISSING`, `BENCHMARK_GATE_FAILED`.
- [ ] Run `uv run --frozen --project scripts/pkm/reference-runtime python -m pytest -q scripts/pkm/tests/test_reference_adapter_contracts.py`. Expected: zero skip/deselect, PASS.
- [ ] Commit explicit paths with `feat: define snapshot-pinned adapter contracts`.

### Task 2: Build deterministic SQLite FTS5 lexical retrieval

**Files:**
- Create: `scripts/pkm/reference_lexical.py`
- Create: `scripts/pkm/tests/test_reference_lexical.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage3/lexical-cases.json`

- [ ] Write failing tests for deterministic double build, Korean/English literal queries, punctuation/quote injection, empty query, BM25 tie, bounds, corrupt DB, wrong snapshot, and CLI JSON.
- [ ] Create strict `element_rows(rowid INTEGER PRIMARY KEY, element_id TEXT UNIQUE, citation_json BLOB)` and contentless `elements_fts` using `unicode61 remove_diacritics 2` with prefix `2 3 4`.
- [ ] Assign rowids from `element_id` ascending. NFC-normalize queries, extract Unicode word tokens, double-quote each literal, and join with `AND`; raw FTS operators are never passed through.
- [ ] Order by `bm25(elements_fts) ASC, element_id ASC`; expose ranks and citations, never raw scores or SQLite rowids.
- [ ] Implement:

```python
class Fts5Adapter:
    @classmethod
    def build(cls, snapshot: SnapshotView, output_dir: Path, config: Mapping[str, object]) -> IndexManifest: ...
    @classmethod
    def open(cls, snapshot: SnapshotView, index_dir: Path) -> Fts5Adapter: ...
    def search(self, request: SearchRequest) -> tuple[RankedHit, ...]: ...
```

- [ ] CLI is `reference_lexical.py build|verify|search --profile ... --snapshot-commit-id ... --index-dir ...`.
- [ ] Manifest records SQLite version/compile options, tokenizer/prefix, input/row-map/DB digests, and `PRAGMA integrity_check`; build in a fresh candidate directory with page size 4096 and sorted inserts.
- [ ] Run targeted tests. Expected: PASS, deterministic manifest and results.
- [ ] Commit `reference_lexical.py`, test, fixture, and config change with `feat: add deterministic sqlite fts5 adapter`.

### Task 3: Build canonical SQLite graph adjacency

**Files:**
- Create: `scripts/pkm/reference_graph.py`
- Create: `scripts/pkm/tests/test_reference_graph.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage3/graph-cases.json`

- [ ] Write failing tests for out/in/both direction, predicate filters, two-hop cycles, unresolved diagnostics, inferred-edge absence, citation preservation, mixed snapshot, and deterministic build.
- [ ] Build `nodes` and `edges` only from canonical `relations.jsonl` and `lineage.jsonl`, with indexes on source/predicate/target and target/predicate/source.
- [ ] Traverse only `resolved` edges. Return `unresolved|ambiguous|denied` only when explicitly filtered, never as traversal links.
- [ ] Deduplicate cycles by canonical `edge_id`; order by hop, predicate, edge ID, then nullable target.
- [ ] Implement `GraphSqliteAdapter.build/open/expand` and CLI `reference_graph.py build|verify|expand`.
- [ ] Run targeted tests. Expected: PASS.
- [ ] Commit explicit paths with `feat: add canonical sqlite graph adapter`.

### Task 4: Implement the NumPy exact-cosine oracle

**Files:**
- Create: `scripts/pkm/reference_vector.py`
- Create: `scripts/pkm/tests/test_reference_vector.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage3/vector-cases.json`

- [ ] Write failing tests for exact ties, NaN/Inf, zero norm, duplicate ID, dimension mismatch, repeated build, corrupt NPY, wrong snapshot, and CLI.
- [ ] Store `element_ids.jsonl`, little-endian float32 `vectors.npy`, and `manifest.json`; use float64 for norm/dot accumulation, normalize to float32, and rank by score descending then element ID ascending.
- [ ] Implement `normalize_embeddings`, `ExactVectorIndex.build/open/search`, and CLI `build-exact|verify-exact|search-exact`.
- [ ] Do not expose raw similarity scores in the common API.
- [ ] Run targeted tests. Expected: PASS.
- [ ] Commit explicit paths with `feat: add exact numpy cosine oracle`.

### Task 5: Select a local embedding model by frozen bake-off

**Files:**
- Create: `scripts/pkm/reference_embedding.py`
- Create: `scripts/pkm/tests/test_reference_embedding_bakeoff.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage3/model-bakeoff-cases.json`
- External create: `$EDU_REFERENCE_ROOT/models/<model_id>/<revision>/manifest.json`

- [ ] Write failing tests using fake local model manifests for pooling/prefix choice, `local_files_only=True`, digest mismatch, both-fail, quality winner, resource tie-break, and deterministic final selection.
- [ ] Profile BGE-M3 as CLS pooling, unchanged query/passage text, native max length 8192; profile multilingual-E5-base as attention-mask mean pooling, `query: ` / `passage: ` prefixes, max length 512. Both emit L2-normalized float32 on the CPU acceptance profile.
- [ ] Require exact model revision, every file digest, license, tokenizer/config digest, dimension, runtime, and hardware profile. Model acquisition is a separately recorded network action; bake-off is offline.
- [ ] Implement `TransformersEmbedder.from_manifest(..., local_files_only=True)`, `embed_queries`, `embed_passages`, and `select_model`.
- [ ] Require each candidate independently to meet nDCG@10 ≥0.80, MRR@10 ≥0.80, Recall@20 ≥0.90. Select by nDCG, MRR, Recall descending; exact quality tie uses peak RSS then p95 latency ascending, then model ID ascending. If both fail, emit `MODEL_GATE_FAILED` and repair data/chunking rather than add an unplanned model.
- [ ] Run targeted tests and frozen bake-off. Expected: tests PASS and one closed selection manifest.
- [ ] Commit tracked code/tests/fixture only with `feat: add local embedding model bakeoff`; model bytes stay external.

### Task 6: Gate the LanceDB scale candidate against the oracle

**Files:**
- Create: `scripts/pkm/reference_lancedb.py`
- Create: `scripts/pkm/tests/test_reference_lancedb.py`

- [ ] Write failing tests without `importorskip` for flat parity, deterministic manifest, wrong snapshot, corrupt table, IVF-PQ config constraints, and Recall@20 against exact.
- [ ] Implement explicit `flat` and `ivf_pq` modes only. Pin `metric=cosine`, `num_partitions=16`, `num_sub_vectors=16`, `nprobes=8`, `refine_factor=10`; reject dimension divisibility and minimum-corpus violations.
- [ ] Implement `LanceVectorAdapter.build/open/search`, `compare_recall(candidate, exact_oracle, queries, k=20)`, and CLI `build|verify|search|compare-exact`.
- [ ] Require ANN Recall@20 ≥0.98. No automatic backend or mode fallback.
- [ ] Run targeted tests. Expected: PASS.
- [ ] Commit explicit paths with `feat: add gated lancedb vector candidate`.

### Task 7: Implement deterministic snapshot-pinned hybrid search

**Files:**
- Create: `scripts/pkm/reference_hybrid.py`
- Create: `scripts/pkm/reference_search.py`
- Create: `scripts/pkm/tests/test_reference_hybrid.py`

- [ ] Write failing tests for hand-computed RRF, ties, missing channel, mixed snapshots, head movement after open, citation corruption, graph separation, and deterministic JSON.
- [ ] Candidate depth is `min(200, max(20, 4*limit))`; RRF uses `k=60`, equal weights, exact `Fraction` ordering, and element ID tie-break. Graph is a separate result channel and is never fused.
- [ ] Construction rejects adapters with different snapshot pins. Golden manifest explicitly selects `exact_numpy`; full-scale manifest selects only a verified `lancedb_ivf_pq` candidate.
- [ ] Implement CLI `reference_search.py search --profile ... --snapshot-commit-id ... --hybrid-manifest ... --query ... --limit ...` returning fused hits, separate graph hits, channel ranks, citations, and pin.
- [ ] Run targeted tests. Expected: PASS.
- [ ] Commit explicit paths with `feat: add deterministic snapshot-pinned hybrid search`.

### Task 8: Close Stage-3 retrieval quality and performance

**Files:**
- Create: `scripts/pkm/reference_retrieval_benchmark.py`
- Create: `scripts/pkm/tests/test_reference_retrieval_benchmark.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage3/retrieval-golden.json`
- External create: `$EDU_REFERENCE_ROOT/reports/stage3/<snapshot_commit_id>/retrieval-report.json`

- [ ] Implement graded nDCG@10 (`2^rel-1`), MRR@10 (rel≥1), Recall@20, important-claim citation precision/coverage, and exploration evidence quality. Emit six-place decimal strings, not binary floats.
- [ ] Bootstrap intervals use exactly 10,000 PCG64 samples seeded from the first 16 hex characters of the query-set SHA-256. Performance values use integer microseconds/bytes.
- [ ] Implement CLI `run|compare|gate`; `gate` requires the design thresholds, hybrid relative gain ≥5% on nDCG or Recall, no other core regression >2%, ANN Recall@20 ≥0.98, p95/p99/RSS/error/throughput SLOs, and hybrid p95 ≤2× lexical p95.
- [ ] Run the complete Stage-3 suite in the frozen runtime. Expected: zero skip/deselect, PASS.
- [ ] Run the benchmark gate against the exact golden snapshot. Expected: exit 0 and a closed digest-bound report; otherwise repair and repeat without relabeling merely to pass.
- [ ] Publish only verified adapter/model/report manifests in a Stage-3 `activation=false` snapshot.
- [ ] Commit code/test/fixture with `feat: close stage 3 retrieval acceptance`.

## Rollback and Exit Gate

- Adapter failures leave unreferenced candidate directories; Stage-2 JSONL authority remains valid.
- Mixed snapshots, stale/uncommitted inputs, digest mismatch, or missing citation are run-fatal.
- Tuning creates a new manifest and full golden rerun; no in-place parameter mutation.
- Forward rollback publishes a new head selecting a prior verified snapshot; history files are never edited.
- Exit evidence: runtime/model locks, hardware profile, adapter manifests, integrity results, exact-vs-ANN report, query-set digest, retrieval and performance reports, complete tests, and Stage-3 closure digest.
- MCP/client activation remains false; Stage 4 consumes this exact snapshot and adapter manifest set.

## Exact Per-task Test and Commit Commands

Every pytest selection below must report zero skip/deselect, and each cache scan must print nothing.

### Task 1 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider scripts/pkm/tests/test_reference_adapter_contracts.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- docs/reference-layer/adapter-contract.md docs/reference-layer/retrieval-benchmark.md \
  scripts/pkm/reference_adapter_config.yaml scripts/pkm/reference_adapters.py \
  schemas/reference-layer/v1/adapter-index-manifest.schema.json \
  schemas/reference-layer/v1/embedding-model-manifest.schema.json \
  schemas/reference-layer/v1/retrieval-query-set.schema.json \
  schemas/reference-layer/v1/retrieval-result.schema.json \
  schemas/reference-layer/v1/retrieval-report.schema.json \
  scripts/pkm/tests/test_reference_adapter_contracts.py \
  scripts/pkm/tests/fixtures/reference_layer/stage3/elements.jsonl \
  scripts/pkm/tests/fixtures/reference_layer/stage3/relations.jsonl \
  scripts/pkm/tests/fixtures/reference_layer/stage3/lineage.jsonl \
  scripts/pkm/tests/fixtures/reference_layer/stage3/schema-positive.json \
  scripts/pkm/tests/fixtures/reference_layer/stage3/schema-negative.json
git diff --cached --check
git commit -m "feat: define snapshot-pinned adapter contracts"
```

### Task 2 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider scripts/pkm/tests/test_reference_lexical.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_lexical.py scripts/pkm/tests/test_reference_lexical.py \
  scripts/pkm/tests/fixtures/reference_layer/stage3/lexical-cases.json \
  scripts/pkm/reference_adapter_config.yaml
git diff --cached --check
git commit -m "feat: add deterministic sqlite fts5 adapter"
```

### Task 3 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider scripts/pkm/tests/test_reference_graph.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_graph.py scripts/pkm/tests/test_reference_graph.py \
  scripts/pkm/tests/fixtures/reference_layer/stage3/graph-cases.json
git diff --cached --check
git commit -m "feat: add canonical sqlite graph adapter"
```

### Task 4 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider scripts/pkm/tests/test_reference_vector.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_vector.py scripts/pkm/tests/test_reference_vector.py \
  scripts/pkm/tests/fixtures/reference_layer/stage3/vector-cases.json
git diff --cached --check
git commit -m "feat: add exact numpy cosine oracle"
```

### Task 5 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider scripts/pkm/tests/test_reference_embedding_bakeoff.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_embedding.py scripts/pkm/tests/test_reference_embedding_bakeoff.py \
  scripts/pkm/tests/fixtures/reference_layer/stage3/model-bakeoff-cases.json
git diff --cached --check
git commit -m "feat: add local embedding model bakeoff"
```

### Task 6 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider scripts/pkm/tests/test_reference_lancedb.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_lancedb.py scripts/pkm/tests/test_reference_lancedb.py
git diff --cached --check
git commit -m "feat: add gated lancedb vector candidate"
```

### Task 7 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider scripts/pkm/tests/test_reference_hybrid.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_hybrid.py scripts/pkm/reference_search.py \
  scripts/pkm/tests/test_reference_hybrid.py
git diff --cached --check
git commit -m "feat: add deterministic snapshot-pinned hybrid search"
```

### Task 8 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_adapter_contracts.py scripts/pkm/tests/test_reference_lexical.py \
  scripts/pkm/tests/test_reference_graph.py scripts/pkm/tests/test_reference_vector.py \
  scripts/pkm/tests/test_reference_embedding_bakeoff.py scripts/pkm/tests/test_reference_lancedb.py \
  scripts/pkm/tests/test_reference_hybrid.py scripts/pkm/tests/test_reference_retrieval_benchmark.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_retrieval_benchmark.py \
  scripts/pkm/tests/test_reference_retrieval_benchmark.py \
  scripts/pkm/tests/fixtures/reference_layer/stage3/retrieval-golden.json
git diff --cached --check
git commit -m "feat: close stage 3 retrieval acceptance"
```

Expected for every block: targeted/full selection PASS, empty cache scan, cached diff limited to the listed paths, and the named commit succeeds. External model/index/report bytes are never staged.
