# Edu Reference Layer Stage 4 Read-only MCP and Clients Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Keep persistent client activation false until the Stage-5 release report closes.

**Goal:** Expose the same pinned evidence and risk verdict through exactly four bounded read-only local STDIO MCP tools and prove parity from Codex, Claude Code, and Gemini CLI.

**Architecture:** A single MCP server opens one verified Stage-3 snapshot and adapter set at startup. A raw-byte strict STDIO wrapper feeds the low-level MCP `Server`, preserving lexical validation before SDK model decoding. Search, read, citation, and preflight share a strict service layer; every request and response binds the same snapshot commit. Opaque `edu-ref` URIs prevent filesystem disclosure. Client profiles are thin generated local configurations; Stage 4 proves isolated discovery/config parity and direct protocol parity, while actual model E2E and persistent activation remain Stage 5 work.

**Tech Stack:** Python 3.12 frozen runtime, MCP Python SDK 1.28.1 stable v1 (`<2`), protocol baseline 2025-11-25, local STDIO, canonical JSON, pytest, installed Codex/Claude/Gemini CLIs.

## Stage Entry Gate

- Exact Stage-3 `snapshot_commit_id`, snapshot-manifest digest, adapter manifests, model manifest, and retrieval report verify from a fresh process.
- MCP process receives only the external profile path and snapshot commit ID; it has no vault write authority, network endpoint, secret, or approval-minting capability.
- Existing ADR-003 and security fixture remain historical static contract evidence. Runtime tests must call production validators and actual STDIO protocol surfaces.

---

### Task 1: Define production security, tool, and client-profile contracts

**Files:**
- Create: `docs/reference-layer/mcp-runtime-profile.md`
- Create: `docs/reference-layer/client-parity-spec.md`
- Create: `scripts/pkm/reference_mcp_profile.yaml`
- Create: `schemas/reference-layer/v1/mcp-runtime-profile.schema.json`
- Create: `schemas/reference-layer/v1/mcp-tool-manifest.schema.json`
- Create: `schemas/reference-layer/v1/mcp-client-probe-report.schema.json`
- Create: `schemas/reference-layer/v1/mcp-client-parity-report.schema.json`
- Create: `scripts/pkm/reference_security.py`
- Create: `scripts/pkm/tests/test_reference_security_runtime.py`

- [ ] Write failing tests that replay all security negative candidates through production code, reject floats/integral-float aliases, traversal/absolute paths, write tools, prompt/resources advertisement, raw content logs, environment/secret leakage, network transport, trust bypass, approval minting, and oversized input/output.
- [ ] Implement exact closed limits:

| Tool | Input bytes | Max items | Output bytes | Timeout |
|---|---:|---:|---:|---:|
| `search` | 4,096 | 50 | 1,048,576 | 30,000 ms |
| `read` | 2,048 | 20 | 1,048,576 | 30,000 ms |
| `citation` | 8,192 | 100 | 262,144 | 30,000 ms |
| `preflight` | 8,192 | 1 | 65,536 | 30,000 ms |

- [ ] Implement `validate_tool_request(tool_name, payload, snapshot_pin)` and `bounded_response(tool_name, payload)` using strict canonical byte counts before work and after serialization. Unknown fields and non-integer limits fail closed.
- [ ] Implement `loads_strict_json(raw)` with fixed order `strict UTF-8 → BOM reject → duplicate-key reject → fraction/exponent integer-alias reject → schema → semantic`. Do not use the SDK's replacement-decoding STDIO helper.
- [ ] Permit only sanitized event name, stable error code, duration bucket/integer microseconds, counts, snapshot commit ID, and citation IDs in logs. Forbid queries, source text, locators, paths, model prompts, and citation bodies.
- [ ] Run security contract + runtime tests under frozen uv. Expected: all 43 negative contract cases and runtime cases reject with exact codes; positive profile passes.
- [ ] Commit explicit paths with `feat: enforce reference mcp runtime security`.

### Task 2: Implement pinned citation and read resolution

**Files:**
- Create: `scripts/pkm/reference_citation.py`
- Create: `scripts/pkm/reference_resource_uri.py`
- Create: `scripts/pkm/reference_read_service.py`
- Create: `scripts/pkm/tests/test_reference_citation.py`
- Create: `scripts/pkm/tests/test_reference_read_service.py`

- [ ] Write failing tests for fixed citation vectors, wrong snapshot, content digest drift, unknown/duplicate citation, malformed opaque URI, traversal encoding, wrong logical source, output truncation at element boundaries, and head movement after service open.
- [ ] Implement:

```python
@dataclass(frozen=True, slots=True)
class ServiceContext:
    snapshot: SnapshotView
    lexical: RankedAdapter
    vector: RankedAdapter
    graph: GraphAdapter

class CitationResolver:
    def resolve(self, citation_ids: Sequence[str], pin: SnapshotPin) -> tuple[CitationRef, ...]: ...

class ReadService:
    def read(self, resource_uri: str, element_ids: Sequence[str], pin: SnapshotPin) -> Mapping[str, object]: ...
```

- [ ] Accept only `edu-ref://resource/<logical_source_id>` with no query, fragment, userinfo, percent-encoded separator, or extra segment. Resolve through canonical element/citation closures, never a supplied path.
- [ ] Return canonical text, media type, locator, provenance IDs, citation, and snapshot pin. Do not return absolute source/registry paths or backend row IDs.
- [ ] Pin `ServiceContext` once at process startup; later head advancement cannot change the open service. A request for another commit returns `SNAPSHOT_MISMATCH`.
- [ ] Run targeted tests. Expected: PASS.
- [ ] Commit explicit paths with `feat: add pinned citation and read services`.

### Task 3: Implement risk-sensitive preflight

**Files:**
- Create: `scripts/pkm/reference_preflight.py`
- Create: `scripts/pkm/tests/test_reference_preflight.py`
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage4/preflight-cases.json`

- [ ] Write failing tests for every task kind, evidence quality boundaries 84.99/85.00, missing important-claim citation, wrong snapshot, unsupported citation, duplicated claim unit, approval-mint attempt, and deterministic reasons/order.
- [ ] Implement:

```python
@dataclass(frozen=True, slots=True)
class PreflightRequest:
    task_kind: Literal["change", "decision", "exploration", "summary"]
    claim_units: tuple[ClaimUnit, ...]
    citation_ids: tuple[str, ...]
    snapshot: SnapshotPin

@dataclass(frozen=True, slots=True)
class PreflightVerdict:
    action: Literal["allow", "warn", "block", "request-approval"]
    evidence_quality: str
    reasons: tuple[str, ...]
    limitations: tuple[str, ...]
    snapshot: SnapshotPin
```

- [ ] `change`: block on any evidence/citation gap. `decision`: request approval on gap and never mint approval. `exploration|summary`: warn with explicit limitations only when evidence quality ≥85%; below that return insufficient evidence.
- [ ] Important change/decision claim citation precision and coverage must be 100%. Validate cited content against the pinned canonical snapshot.
- [ ] Use exact rational/Decimal calculation and six-place decimal strings; no binary float policy decisions.
- [ ] Run targeted tests. Expected: PASS.
- [ ] Commit explicit paths with `feat: add risk-sensitive evidence preflight`.

### Task 4: Implement the local STDIO MCP server

**Files:**
- Create: `scripts/pkm/reference_mcp.py`
- Create: `scripts/pkm/reference_mcp_stdio.py`
- Create: `scripts/pkm/tests/test_reference_mcp_protocol.py`
- Create: `scripts/pkm/tests/test_reference_mcp_stdio.py`

- [ ] Write failing protocol tests for strict malformed UTF-8/BOM/duplicate key/fraction/exponent ordering, initialize/version negotiation, capability negotiation, exactly four tools, schema bounds, request timeout/cancellation, malformed JSON-RPC, concurrent calls, deterministic responses, clean EOF shutdown, SIGTERM escalation, and no stdout noise outside protocol frames.
- [ ] Set `MCP_PROTOCOL_VERSION="2025-11-25"`, `MCP_SDK_VERSION="1.28.1"`, and `TOOL_ORDER=("search","read","citation","preflight")`. Reject every other negotiated version with `PROTOCOL_VERSION_MISMATCH`; do not fall back to an older SDK-supported version.
- [ ] Build one low-level `Server` named `edu-reference-layer` behind `reference_mcp_stdio.py`; do not use `FastMCP` or the SDK default `stdio_server()`. Register exactly:

```python
async def search(query: str, limit: int, snapshot_commit_id: str) -> dict[str, object]: ...
async def read(resource_uri: str, element_ids: list[str], snapshot_commit_id: str) -> dict[str, object]: ...
async def citation(citation_ids: list[str], snapshot_commit_id: str) -> dict[str, object]: ...
async def preflight(task_kind: str, claim_units: list[dict[str, object]], citation_ids: list[str], snapshot_commit_id: str) -> dict[str, object]: ...
```

- [ ] Advertise tools capability only: no write tool, prompt, resource capability, sampling, elicitation, remote transport, or approval capability. Mark every tool read-only, non-destructive, idempotent, and closed-world through annotations.
- [ ] Disable SDK raw-message DEBUG logs. Only the production sanitized telemetry allowlist may emit events.
- [ ] Startup verifies runtime profile, committed snapshot, adapter/canonical closure digests, and Stage-3 report before accepting initialization. Readiness failure exits nonzero with one sanitized stderr JSON record.
- [ ] Search wraps the Stage-3 hybrid retriever; graph results remain separately labeled. All tools enforce production bounds and the pinned commit.
- [ ] Run actual subprocess raw-STDIO and stable SDK client tests. Expected: protocol 2025-11-25 negotiated, exact four tools, strict malformed-byte ordering, all tests PASS, clean input-close shutdown.
- [ ] Commit explicit paths with `feat: expose read-only reference mcp`.

### Task 5: Generate thin client profiles without persistent activation

**Files:**
- Create: `scripts/pkm/reference_client_config.py`
- Create: `scripts/pkm/reference_client_probe.py`
- Create: `config/reference-layer/clients/codex.config.toml.template`
- Create: `config/reference-layer/clients/claude.mcp.json.template`
- Create: `config/reference-layer/clients/gemini.settings.json.template`
- Create: `scripts/pkm/tests/test_reference_client_profiles.py`
- Create: `scripts/pkm/tests/test_reference_client_probe.py`

- [ ] Write failing tests for absolute/private path committed in template, different executable/profile/snapshot across clients, tool allowlist mismatch, Codex HTTP mode, Claude project approval mutation, Gemini `trust=true`, extra environment variables, and persistent real-config write.
- [ ] Templates allow exactly `@@PYTHON_EXECUTABLE@@`, `@@SERVER_SCRIPT@@`, `@@PROFILE_PATH@@`, `@@SNAPSHOT_COMMIT_ID@@`, and `@@SERVER_CWD@@`; any other or unexpanded token fails. The generator resolves them into an external `0700` probe directory with `0600` files and validates exact digests.
- [ ] Render reviewable commands from current CLI surfaces:

```text
codex mcp add edu-reference-layer --env EDU_REFERENCE_PROFILE=@@PROFILE_PATH@@ -- @@PYTHON_EXECUTABLE@@ @@SERVER_SCRIPT@@ --snapshot-commit-id @@SNAPSHOT_COMMIT_ID@@
claude mcp add --scope local --transport stdio edu-reference-layer -e EDU_REFERENCE_PROFILE=@@PROFILE_PATH@@ -- @@PYTHON_EXECUTABLE@@ @@SERVER_SCRIPT@@ --snapshot-commit-id @@SNAPSHOT_COMMIT_ID@@
gemini mcp add --scope project --transport stdio --timeout 30000 --include-tools search,read,citation,preflight --description "Read-only edu evidence" edu-reference-layer @@PYTHON_EXECUTABLE@@ @@SERVER_SCRIPT@@ --snapshot-commit-id @@SNAPSHOT_COMMIT_ID@@
```

The Gemini command omits `--trust` so it remains false. These are rendered activation commands; Stage 4 does not execute them against the real project/user config.

- [ ] Codex template sets exact four enabled tools and `required=true`; Claude uses an isolated temporary `.mcp.json` or `--mcp-config --strict-mcp-config`; Gemini sets exact server allowlist/includeTools and `trust=false`. All three resolve to the same Python, server script, profile digest, working directory, and snapshot.
- [ ] `reference_client_probe.py` uses fresh temporary client-supported config locations or command-line overlays, rejects any attempt to repurpose `HOME`, `CODEX_HOME`, or real client configuration, and compares the real config existence/digest before and after. Drift fails immediately and is not auto-restored.
- [ ] Stage-4 probe performs template parsing, isolated MCP discovery, tool-manifest comparison, and a direct MCP harness parity workload. It does not require the Codex/Claude/Gemini language models to decide to call tools; actual model E2E is Stage 5.
- [ ] Run profile/probe unit tests. Expected: PASS and no modification to real Codex/Claude/Gemini configuration.
- [ ] Commit explicit tracked paths with `feat: add thin reference client profiles`.

### Task 6: Close isolated client discovery and direct Stage-4 parity

**Files:**
- Create: `scripts/pkm/tests/fixtures/reference_layer/stage4/client-parity-cases.json`
- External create: `$EDU_REFERENCE_ROOT/reports/stage4/<snapshot_commit_id>/client-probe-report.json`
- Modify: `.planning/2026-07-22-gjc-reference-layer-codex-execution/progress.md`

- [ ] Snapshot the live client versions and supported MCP flags; compare to the runtime profile. Version drift triggers fresh compatibility probes, not silent acceptance.
- [ ] Prove all three clients discover the same server executable/profile/snapshot and exact tool manifest in isolated configuration without persisting activation.
- [ ] Run search→read→citation→preflight, timeout, malformed payload, oversized request, wrong snapshot, server-not-ready, and shutdown through the direct MCP harness. Require deterministic response/stable-ID/citation/preflight/error digests.
- [ ] Independently confirm no real client config changed and no network listener or outbound connection existed.
- [ ] Run the full Stage-4 test suite. Expected: zero skip/deselect and PASS.
- [ ] Publish the client profiles/probe/parity reports and MCP runtime manifest in a Stage-4 snapshot with exact fields `stage_verified=4`, `activation=false`, and `model_e2e=not_run`.
- [ ] Commit the privacy-safe parity fixture with `test: verify stage 4 mcp and client parity`; external reports remain outside Git.

## Rollback and Exit Gate

- Server/probe failures terminate temporary processes/configurations; the previous verified Stage-3 head remains valid.
- A snapshot mismatch, citation mismatch, tool-surface expansion, config persistence, trust bypass, or egress observation is run-fatal.
- Stage-4 completion proves temporary E2E parity, not permanent activation. Activation stays false until the Stage-5 release report and restored-root drill both pass.
- Exit evidence: exact server/runtime/profile digests, protocol transcript digests, security replay, client versions, tool-schema hashes, parity report, config before/after hashes, no-listener/egress observation, and Stage-4 closure.

## Exact Per-task Test and Commit Commands

Every pytest selection must report zero skip/deselect, and every cache scan must print nothing.

### Task 1 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_security_contracts.py \
  scripts/pkm/tests/test_reference_security_runtime.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- docs/reference-layer/mcp-runtime-profile.md docs/reference-layer/client-parity-spec.md \
  scripts/pkm/reference_mcp_profile.yaml schemas/reference-layer/v1/mcp-runtime-profile.schema.json \
  schemas/reference-layer/v1/mcp-tool-manifest.schema.json \
  schemas/reference-layer/v1/mcp-client-probe-report.schema.json \
  schemas/reference-layer/v1/mcp-client-parity-report.schema.json \
  scripts/pkm/reference_security.py scripts/pkm/tests/test_reference_security_runtime.py
git diff --cached --check
git commit -m "docs: define stage 4 MCP runtime contracts"
```

### Task 2 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_citation.py scripts/pkm/tests/test_reference_read_service.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_resource_uri.py scripts/pkm/reference_citation.py \
  scripts/pkm/reference_read_service.py scripts/pkm/tests/test_reference_citation.py \
  scripts/pkm/tests/test_reference_read_service.py
git diff --cached --check
git commit -m "feat: add stable reference URI and citation service"
```

### Task 3 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider scripts/pkm/tests/test_reference_preflight.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_preflight.py scripts/pkm/tests/test_reference_preflight.py \
  scripts/pkm/tests/fixtures/reference_layer/stage4/preflight-cases.json
git diff --cached --check
git commit -m "feat: add read-only reference preflight"
```

### Task 4 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_mcp_protocol.py scripts/pkm/tests/test_reference_mcp_stdio.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_mcp_stdio.py scripts/pkm/reference_mcp.py \
  scripts/pkm/tests/test_reference_mcp_protocol.py scripts/pkm/tests/test_reference_mcp_stdio.py
git diff --cached --check
git commit -m "feat: add strict read-only reference MCP"
```

### Task 5 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_client_profiles.py scripts/pkm/tests/test_reference_client_probe.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/reference_client_config.py scripts/pkm/reference_client_probe.py \
  config/reference-layer/clients/codex.config.toml.template \
  config/reference-layer/clients/claude.mcp.json.template \
  config/reference-layer/clients/gemini.settings.json.template \
  scripts/pkm/tests/test_reference_client_profiles.py scripts/pkm/tests/test_reference_client_probe.py
git diff --cached --check
git commit -m "feat: add isolated reference client probes"
```

### Task 6 commands

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --project scripts/pkm/reference-runtime \
  python -m pytest -q -p no:cacheprovider \
  scripts/pkm/tests/test_reference_security_contracts.py \
  scripts/pkm/tests/test_reference_security_runtime.py \
  scripts/pkm/tests/test_reference_citation.py scripts/pkm/tests/test_reference_read_service.py \
  scripts/pkm/tests/test_reference_preflight.py scripts/pkm/tests/test_reference_mcp_protocol.py \
  scripts/pkm/tests/test_reference_mcp_stdio.py scripts/pkm/tests/test_reference_client_profiles.py \
  scripts/pkm/tests/test_reference_client_probe.py
find scripts/pkm -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \)
git add -- scripts/pkm/tests/fixtures/reference_layer/stage4/client-parity-cases.json
git diff --cached --check
git commit -m "test: verify stage 4 MCP and client parity"
```

Expected for every block: PASS, empty cache scan, cached diff limited to the listed paths, and the named commit succeeds. External probe/parity reports are not staged.
