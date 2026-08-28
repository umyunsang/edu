---
aliases:
  - edu reference layer stage 1-5 design
created: '2026-07-22'
status: approved
tags:
  - specification
  - reference-layer
  - approved
title: Edu LLM-readable reference layer Stage 1-5 설계
type: specification
updated: '2026-07-22'
---

# Edu LLM-readable reference layer Stage 1–5 설계

## 1. 상태와 권위

이 문서는 GJC 딥인터뷰 원본과 2026-07-22 현재 `edu` 저장소의 live authority를 바탕으로 승인된 Stage 1–5 제품 설계를 고정한다.

- 권위 있는 딥인터뷰 원본: `.gjc/_session-019f6576-8efa-7000-a627-19ae29be4687/specs/deep-interview-edu-llm-reference-layer.md`
- 원본 SHA-256: `51e5726b6ef76774130123587277386fb499b0881fed8bd52ebac65512491277`
- 승인된 rollout: Approach A — digest-frozen representative corpus의 Stage 1→5 vertical slice 후 deterministic full-vault expansion
- 사용자 승인: 2026-07-22, 목표 진행에 필요한 모든 역질의 승인과 가장 정합한 방법·강한 성능 benchmark 사용
- workflow: `.planning/2026-07-22-gjc-reference-layer-codex-execution/`

이 문서는 제품 설계를 승인하지만 그 자체로 구현 완료, runtime activation, live acceptance 또는 release approval을 주장하지 않는다. 실제 상태는 live files, digests, registry head, test output, benchmark report와 restore/rollback 증거로만 판정한다.

`.omo/**`는 폐기된 workflow의 역사 자료다. 이 설계는 `.omo` actor, amendment, receipt, ledger, launcher, cleanup 또는 capability machinery를 복원하지 않는다.

## 2. 제품 포지셔닝

### 2.1 Category

Edu reference layer는 Obsidian 검색 기능이나 특정 RAG backend가 아니다. 개인 지식 자산을 stable, traceable, citation-bearing evidence로 변환하여 Codex, Claude Code, Gemini CLI가 동일한 근거와 preflight 판정을 사용하게 만드는 local-first evidence substrate이자 reference control plane이다.

### 2.2 핵심 문제

현재 개인 지식 자산은 Markdown, PDF, notebook, source code, data, experiment output에 분산되어 있다. 클라이언트별 검색·chunking·PDF 변환이 다르면 다음 문제가 생긴다.

- 동일한 근거가 서로 다른 ID와 chunk로 보인다.
- PDF 표·수식·페이지·좌표와 code/experiment lineage가 사라진다.
- 검색 backend 교체가 evidence identity를 바꾼다.
- 중요한 claim의 provenance와 citation completeness를 증명할 수 없다.
- client마다 change/decision/exploration preflight가 달라진다.

### 2.3 제품 약속

- 원본 vault와 PDF를 변경하지 않는다.
- canonical evidence identity는 backend와 파일 경로에 종속되지 않는다.
- 모든 accepted evidence는 source digest, observation, locator, citation으로 역추적된다.
- JSONL이 authority이고 Parquet·SQLite·LanceDB 등은 재생성 가능한 derivative다.
- 세 client가 동일 snapshot에서 동일 stable ID, citation, preflight verdict를 받는다.
- 위험도가 높은 change/decision은 근거 부족 시 차단하거나 승인을 요청하고, exploration은 임계치를 만족할 때만 limitation과 함께 진행한다.

## 3. Scope와 non-goals

### 3.1 Stage 1–5 scope

1. 전체 vault의 policy-safe inventory, hash population, source registry와 durable external publication
2. Markdown/PDF/notebook/source/data/experiment extraction, canonical JSONL, deterministic Parquet, authored relation과 lineage
3. replaceable lexical/graph/vector adapters와 deterministic hybrid retrieval
4. 하나의 local read-only STDIO MCP와 Codex/Claude Code/Gemini thin client configuration
5. edu golden, PDF structure, retrieval/citation, external benchmark, three-client E2E, performance, telemetry, retention, backup/restore, crash recovery와 rollback

### 3.2 명시적 non-goals

- vault source, 원본 PDF 또는 기존 note의 자동 수정·삭제·덮어쓰기
- derived Markdown/JSON/index의 Git commit 또는 public publication
- 특정 graph/vector database를 canonical source of truth로 사용
- MCP write tool, approval minting 또는 autonomous release
- remote inference, remote hybrid, externally reachable or persistent HTTP listener, multi-user service 또는 ambient network egress
- global GJC source/release 변경
- `.omo` workflow 재생성

Remote inference, externally reachable/persistent HTTP 또는 source mutation이 필요해지면 이 설계의 승인 범위를 벗어나므로 별도 threat model과 명시 승인이 필요하다. OpenDataLoader local-hybrid/OCR가 요구하는 batch-scoped sidecar만 예외다. 이 sidecar는 `127.0.0.1`의 OS-assigned port에만 bind하고, model/runtime은 사전 획득·digest 검증하며, extraction 중 egress와 download를 차단하고, 성공·실패·취소 후 즉시 종료해야 한다. `0.0.0.0`, remote URL, persistent port와 implicit fast fallback은 허용하지 않는다.

## 4. Rollout 결정

### 4.1 Approach A

첫 acceptance unit은 representative corpus가 Stage 1부터 Stage 5까지 동일 pipeline을 통과하는 vertical slice다. 이 slice는 contract·identity·lineage·retrieval·citation·client parity의 오류를 작은 blast radius에서 찾는다.

Slice PASS는 terminal completion이 아니다. 이후 동일 versioned pipeline으로 전체 vault를 deterministic batch 처리하고 final full-vault acceptance를 수행해야 한다.

### 4.2 Corpus policy와 exact selection 분리

- Git에는 개인정보가 없는 `corpus-policy`와 role template만 둔다.
- 실제 relative path, source digest, source ID와 selection evidence는 외부 registry의 `0600` exact selection manifest에 둔다.
- whole-vault metadata census 후 candidate를 선택하고, bounded allowlisted read로 role과 digest를 확인한다.
- exact selection manifest는 extraction이나 registry publication 전에 frozen closure가 된다.

대표 corpus는 최소한 다음 역할을 포함한다.

- Korean Obsidian Markdown: YAML/frontmatter, explicit relation fields, wikilink, alias, embed, callout, LaTeX, Mermaid, code block
- born-digital PDF: table 또는 formula 포함
- complex/scanned PDF: local hybrid/OCR fallback을 실제로 실행
- executed notebook: Markdown cell, code cell, input/output/metric
- standalone source code, config/data, experiment result
- resolved, unresolved, ambiguous graph target
- vault 밖 test-owned negative containment/quality fixtures

### 4.3 Full-vault deterministic batches

- Stage 1A: whole-vault metadata population snapshot과 selected-corpus hashes publication
- Vertical slice Stage 1–5 PASS 후 Stage 1B: 나머지 allowed assets를 NFC comparison key와 raw-name bytes 순으로 처리
- 기본 batch 상한: regular asset 100개 또는 declared input 512 MiB 중 먼저 도달하는 값
- 상한보다 큰 단일 asset은 one-item batch
- batch policy와 partition digest를 snapshot manifest에 고정
- failed/quarantined/placeholder asset은 frozen denominator에서 제거하지 않음
- dataless placeholder는 명시적 hydration과 post-hydration observation 없이 accepted가 될 수 없음

## 5. 시스템 아키텍처

### 5.1 Tracked control plane

다음만 Git에 둔다.

- policies, JSON schemas, fixed vectors, corpus role template
- dependency/model acquisition definitions와 expected artifact digests
- benchmark definitions, query sets, scoring code와 report schemas
- client configuration templates without secrets, approval state or absolute personal paths

### 5.2 Read-only source plane

- vault는 immutable input이다.
- no-follow resolver는 approved inventory/selection manifest에 있는 asset만 연다.
- lexical containment, segment-by-segment `lstat`, NFC collision, symlink, TOCTOU와 source-change-before/after-hash를 검사한다.
- source open은 bounded streaming이고 content sniffing으로 policy를 넓히지 않는다.

### 5.3 Ingest and registry plane

- external registry/cache는 vault와 iCloud root 밖의 한 local filesystem에 둔다.
- directory `0700`, regular file `0600`, symlink segment 없음, free-space와 backup root 검증을 요구한다.
- writer lock은 모든 head read, source-ID allocation, bootstrap, retry, rollback보다 먼저 획득한다.
- `committed-head.json`의 verified exact bytes/digest만 authority다.
- closure object와 commit envelope는 create-new, file fsync, parent-directory fsync, reread와 digest check를 head CAS 전에 완료한다.
- candidate나 orphan object는 head가 가리키기 전에는 authority, source-ID allocation, active run 또는 citation source가 아니다.

### 5.4 Canonical evidence plane

별도 canonical JSONL authority를 둔다.

1. `source-observations.jsonl`
2. `elements.jsonl`
3. `relations.jsonl`
4. `lineage.jsonl`

각 stream은 canonical JSON, NFC, signed-int64-only, deterministic key/row order, duplicate rejection, final LF와 digest 규칙을 갖는다. Parquet은 각 stream의 typed derivative이며 authority가 아니다.

### 5.5 Adapter plane

- lexical: SQLite FTS5 reference adapter
- graph: SQLite read-only adjacency projection
- vector correctness oracle: NumPy exact normalized cosine
- vector scale candidate: LanceDB OSS embedded local adapter
- hybrid: rank-position 기반 reciprocal-rank fusion

모든 adapter는 하나의 committed snapshot만 소비한다. Backend-native ID는 evidence identity로 노출하지 않는다.

### 5.6 Read service plane

하나의 local STDIO MCP가 다음 네 tool만 exact order로 제공한다.

1. `search`
2. `read`
3. `citation`
4. `preflight`

각 search는 `snapshot_commit_id`를 반환한다. 후속 read/citation/preflight는 같은 snapshot을 요구한다. 한 task가 head generation을 섞지 못한다.

### 5.7 Evaluation and operations plane

Evaluation, telemetry, backup/restore, retention, recovery와 rollback은 immutable evidence를 소비하지만 스스로 head를 publish하거나 approval을 mint하지 않는다. Authorized publisher만 closed report를 검증한 뒤 head를 advance한다.

## 6. Identity와 registry contracts

### 6.1 Identity layers

- `logical_source_id`: path-independent logical source UUID
- `location_id`: observed physical location UUID
- `observation_id`: one acquisition/observation UUID
- `element_id`: domain-separated SHA-256
- `relation_id`: domain-separated SHA-256
- `lineage_edge_id`: domain-separated SHA-256

Source/location/observation UUID는 writer lock 아래 candidate-scoped로 할당하며 head CAS 후에만 authoritative하다. Failed candidate에서 ID를 재사용하지 않는다.

### 6.2 Element kinds

기존 v1 ID를 유지하면서 canonical element v1.1에 다음 kind를 지원한다.

- `markdown_section`
- `pdf_block`
- `notebook_cell`
- `source_code_unit`
- `experiment_result`

Locator는 filesystem path를 ID 재료에 포함하지 않는다. Code locator는 language, symbol/cell anchor와 occurrence를, experiment result locator는 run/output/metric anchor를 사용한다. 호환되지 않는 locator 규칙 변경은 major 증가와 reviewed migration map을 요구한다.

### 6.3 Source registry

Closed source registry records는 logical source, location, observation과 content digest를 명시한다.

- 같은 path의 새 content observation은 같은 logical source를 유지한다.
- 독립 복사는 새 logical source다.
- rename/move는 unique evidence 또는 approved migration으로만 연결한다.
- duplicate/rename ambiguity는 `identity_unresolved`로 quarantine한다.
- uncommitted candidate, orphan commit, derived index는 source ID allocator가 읽지 않는다.

### 6.4 Snapshot/closure manifest

Snapshot manifest는 다음 object role과 digest를 닫힌 배열로 열거한다.

- inventory and population
- raw extraction run
- source, element, relation, lineage JSONL
- typed Parquet projections
- lexical, graph, vector index manifests
- evaluation and parity reports
- backup and citation-retention evidence

Manifest는 `capability_stage`, `activation_state`, corpus/population digest, parent head, dependency/model/profile digest를 고정한다. Registry commit은 이 manifest digest를 closure로 참조한다.

## 7. PDF materialization contract

### 7.1 Mandatory outputs

모든 accepted PDF는 `opendataloader-project/opendataloader-pdf`를 통해 다음 immutable external bundle을 만든다.

```text
pdf-run/<source-observation-id>/<run-id>/
  document.md
  document.json
  assets/
  manifest.json
```

- `document.md`: human/LLM-readable Markdown와 RAG chunk source
- `document.json`: page, bounding box, semantic type, reading order와 element provenance source
- `assets/`: extracted image/chart/table media가 있을 때 필수
- `manifest.json`: source/runtime/options/output/quality/selection closure

OpenDataLoader의 combined output은 `format="markdown,json"`으로 실행한다. 같은 batch 안의 PDF는 한 conversion invocation으로 묶어 JVM startup overhead를 줄인다.

Generated output은 원본 PDF 옆, vault 또는 Git에 쓰지 않는다. Original PDF는 immutable source authority다.

### 7.2 Run manifest

Manifest는 최소한 다음을 결속한다.

- source PDF SHA-256, logical source/location/observation ID
- OpenDataLoader repository revision, package/artifact digest
- Java vendor/version/runtime digest와 isolated Python environment digest
- local/hybrid/OCR/formula/picture/language options
- `document.md`, `document.json`, assets의 digest와 byte count
- page/element/table/formula/image counts
- NID/TEDS/MHS와 edu golden/omission result
- fast/local run, local-hybrid run, selected run과 superseded run 관계

### 7.3 Routing

1. 모든 PDF를 deterministic local mode로 먼저 처리한다.
2. 구조 metric 또는 edu omission heuristic이 threshold 미만이면 approved local hybrid/OCR로 재처리한다.
3. Korean/non-English scan은 pinned `ko,en` OCR profile을 사용한다.
4. Formula/picture enrichment는 golden role이나 detected structure가 요구할 때만 pinned option으로 켠다.
5. Fast와 hybrid run은 서로 다른 immutable bundle이다. In-place overwrite는 금지한다.
6. Remote API나 hosted inference로 fallback하지 않는다.

Markdown 생성만 성공하거나 JSON traceability만 성공하면 accepted가 아니다. 두 representation과 closure가 모두 유효해야 한다.

## 8. Markdown, notebook, code, data, experiment extraction

### 8.1 Markdown

- YAML/frontmatter와 body를 분리 보존한다.
- heading occurrence와 stable content anchor로 section locator를 생성한다.
- wikilink, embed, callout, code fence, LaTeX와 Mermaid를 구조적으로 보존한다.
- source byte digest와 generated normalized view digest를 구분한다.

### 8.2 Notebook

- cell ID가 있으면 보존하고 없으면 path-independent content anchor를 생성한다.
- Markdown/code/raw cell, execution count, input, output, error, display data와 metric을 분리한다.
- notebook cell과 experiment result 관계는 lineage에 기록한다.

### 8.3 Source code

- language-aware symbol/function/class 단위가 가능하면 사용하고 실패 시 deterministic bounded line block으로 fallback한다.
- file path는 provenance에만 있고 element ID 재료에는 넣지 않는다.
- parser raw ID와 version은 provenance이지 canonical identity가 아니다.

### 8.4 Data and experiment results

- CSV/JSON/YAML 등 structured data는 schema, column/type, row count와 digest를 기록한다.
- 대용량 row data는 bounded preview와 digest를 canonical evidence로 사용하고 원본을 복제하지 않는다.
- experiment environment, input dataset, code/notebook, parameters, outputs와 metrics는 lineage로 연결한다.

## 9. Authored relations와 lineage

### 9.1 Authored relations

`relations.jsonl`은 다음 source-authored 관계를 canonicalize한다.

- body wikilink and embed
- `graph`, `domain`, `stage`, `module`, `bridge`, `schema`
- `source_model`, `relation_type`, `tech_stack`, `research`, `ecosystem`, `competency`, `related`

Relation ID material은 source element, predicate, NFC-authored target과 occurrence다. `resolved_target_logical_source_id`는 snapshot-relative resolution이므로 ID 재료에서 제외한다.

Target resolution order는 다음과 같다.

1. source-relative exact note path
2. vault-root exact note path
3. unique note stem
4. unique declared alias

Missing, multiple match, denied boundary 또는 path escape는 guessed edge가 아니라 `unresolved`, `ambiguous`, `denied` 상태다.

### 9.2 Experiment lineage

`lineage.jsonl`의 initial relation vocabulary는 다음과 같다.

- `uses`
- `generated_by`
- `derived_from`
- `executed_with`
- `measures`

Edges는 source/element/observation/run/environment digest와 citation을 가진다. Graph backend ID나 inference-only relationship은 canonical lineage가 아니다.

## 10. Deterministic projections and adapters

### 10.1 Parquet

Source/element/relation/lineage별 typed manifest를 둔다. 각 manifest는 다음을 고정한다.

- canonical source digest
- projection digest
- row count and sorted ID-set digest
- schema fingerprint
- writer name/version/artifact digest
- field/row order, compression, dictionary, statistics와 timestamp policy

Identity/equality gate에 nondeterministic float aggregation을 사용하지 않는다.

### 10.2 Lexical adapter

- SQLite FTS5 contentless/external-content projection
- pinned SQLite version and compile options
- Unicode61 tokenizer와 exact prefix/config options
- BM25 ordering, escaped query grammar와 `element_id` ascending tie-break
- integrity check, row-map digest와 canonical input digest

### 10.3 Graph adapter

- SQLite node/edge adjacency tables
- `relations.jsonl`과 `lineage.jsonl`만 canonical edge source로 사용
- one/two-hop traversal, predicate and resolution-status filters
- every edge result includes canonical edge ID와 source citation

### 10.4 Vector adapters and model selection

- NumPy exact normalized cosine는 golden correctness oracle다.
- LanceDB OSS는 full-scale local candidate이며 canonical authority가 아니다.
- Initial model candidates: `BAAI/bge-m3`, `intfloat/multilingual-e5-base`
- Model download는 separate checksum/license/egress-gated acquisition이다.
- Exact revision, weights, tokenizer, config, pooling, normalization, max length, dimension, dtype, runtime와 hardware profile을 manifest에 고정한다.

Golden bake-off에서 quality threshold를 통과한 모델 중 quality가 높은 모델을 선택한다. 동률은 peak memory, latency 순으로 결정한다. 두 후보 모두 threshold에 실패하면 Stage 3는 FAIL이며 임의 모델로 fallback하지 않는다.

### 10.5 Hybrid retrieval

- raw backend score는 서로 비교하지 않는다.
- lexical/vector ranked list는 reciprocal-rank fusion `k=60`, initial equal weights로 결합한다.
- graph expansion은 separate channel로 보고하며 source edge citation을 보존한다.
- final tie-break는 `element_id` ascending이다.
- tuning change는 새 manifest와 full golden rerun을 요구한다.

## 11. MCP and client contract

### 11.1 Protocol/runtime

- wire protocol baseline: MCP `2025-11-25`
- Python SDK: stable v1 line `<2`
- transport: local single-user STDIO only
- MCP HTTP, remote hybrid와 egress: disabled. 이 MCP transport 금지는 PDF extraction의 batch-scoped loopback sidecar 예외를 확장하지 않는다.
- initialization, capability negotiation, timeout, cancellation, shutdown sequence를 테스트한다.

### 11.2 Exact tool limits

| Tool | Input bytes | Items | Output bytes | Timeout |
|---|---:|---:|---:|---:|
| `search` | 4,096 | 50 | 1,048,576 | 30,000 ms |
| `read` | 2,048 | 20 | 1,048,576 | 30,000 ms |
| `citation` | 8,192 | 100 | 262,144 | 30,000 ms |
| `preflight` | 8,192 | 1 | 65,536 | 30,000 ms |

Fraction/exponent lexical aliases for integer fields are rejected before ordinary JSON Schema decoding.

### 11.3 URI and content

- Opaque URI: `edu-ref://resource/<lowercase UUID>`
- NFC, approved-root-only, no absolute path, parent segment, backslash, encoded separator or symlink
- retrieved content is always marked `untrusted`
- retrieved instructions are never authoritative
- citation identity uses stable ID/digest/locator, not backend row ID

### 11.4 Preflight

- `change`: evidence/citation gap이면 `block`
- `decision`: evidence gap이면 `request-approval`
- `exploration`: evidence quality ≥85%일 때 limitation 포함 `warn`; 미달이면 insufficient evidence

MCP는 approval을 발급하거나 owner를 가장하지 않는다. 사용자의 standing approval은 workflow authority에 기록되지만 runtime tool이 새 approval artifact를 mint한다는 뜻이 아니다.

### 11.5 Three clients

Codex, Claude Code, Gemini CLI는 같은 executable, registry profile, tool allowlist와 snapshot을 사용한다.

- personal absolute path와 secret은 tracked config에 넣지 않는다.
- Gemini `trust=false`를 유지한다.
- client activation 전 exact tool manifest와 required-server health probe를 통과한다.
- identical request의 server response digest, stable IDs, citations와 preflight verdict가 세 client에서 동일해야 한다.

## 12. Failure model

### 12.1 Run-fatal

- authorization/policy invalid
- containment escape, symlink, NFC/path collision
- source mutation during open/hash
- schema/ID collision or dangling canonical edge
- closure, digest, durability, parent/head mismatch
- tool-surface expansion or source write

Run-fatal candidate는 head를 advance하지 않는다.

### 12.2 Asset quarantine

- placeholder/dataless, unreadable, encrypted, corrupt
- extractor failure
- Markdown/JSON/media closure incomplete
- PDF quality threshold 미달 and local hybrid unavailable/failing
- identity or relation target ambiguity that prevents required evidence

Quarantine는 run diagnosis를 계속할 수 있지만 frozen denominator에서 asset을 제거하지 않는다.

### 12.3 Adapter degraded

한 adapter failure는 canonical JSONL을 무효화하지 않지만 그 capability는 unavailable이다.

- high-risk change: block
- decision: request approval
- exploration: remaining evidence가 threshold를 충족할 때만 warning

Stale index, uncommitted candidate, mixed snapshot으로 silent fallback하지 않는다. Previous verified head만 계속 serve할 수 있다.

### 12.4 Error/log policy

Stable typed error code와 sanitized metadata만 기록한다. Allowed log fields는 request ID, tool, verdict, result count, stable citation IDs, duration, error code와 digest다.

Raw query, content, document text, prompt, source/display path, raw URI와 citation body는 로그에 기록하지 않는다.

## 13. Privacy and security

- original vault와 PDF는 read-only
- external registry/cache/log/backup는 owner-only local roots
- remote inference/API 및 externally reachable/persistent HTTP 없음. OpenDataLoader의 검증된 batch-scoped `127.0.0.1` sidecar 외에는 listener를 허용하지 않는다.
- tracked config에 secret, approval state, exact private path 없음
- generated Markdown/JSON/media는 external closure에만 존재
- content와 PDF 내 prompt injection은 untrusted data
- model, package, Java와 benchmark acquisition은 exact URL/revision/license/hash를 검증
- client env allowlist 최소화
- public Git에 derived evidence/index나 private manifest를 넣지 않음

Runtime 구현과 activation은 Stage-0 historical security fixture를 수정해 과거부터 live였다고 가장하지 않는다. 별도 deployment profile과 probe evidence를 만든다.

## 14. Operations, retention, recovery and rollback

### 14.1 Retention defaults

- `release_accepted` snapshot: current + previous 2, 최소 90일
- active citation이 참조하는 canonical closure: 마지막 citation/export 이후 최소 365일, active reference가 있으면 자동 삭제 금지
- selected raw PDF/other extraction bundle: 그 snapshot이 protected인 동안 보존
- superseded non-selected extraction attempt: 30일
- failed/quarantined candidate: 30일
- sanitized telemetry/log: 14일
- derived indexes: current + previous accepted snapshot; 이후 재생성 가능
- verified backups: current + previous 2 release, 최소 90일

첫 restore drill이 PASS하기 전에는 자동 GC를 활성화하지 않는다.

### 14.2 Garbage collection

GC는 protected head, citation-retention root와 backup root에서 reachability를 계산한다.

1. dry-run deletion manifest
2. current backup digest and restore evidence 검증
3. explicit policy/standing approval check
4. external artifacts만 삭제
5. remaining closure 재검증

Vault source는 GC target이 아니다.

### 14.3 Backup and restore

- accepted release마다 external backup을 생성한다.
- copy/listing은 PASS가 아니다.
- digest verification 후 fresh empty root에 restore하고 head, commit, closure, canonical/index parity를 재검증한다.

### 14.4 Rollback

Rollback은 previously verified snapshot을 선택하는 새 forward CAS publication이다. History, old commit 또는 head file을 수동 수정하지 않는다.

Post-activation live smoke failure 시 이전 verified release로 forward rollback하고 client health probe를 재실행한다.

## 15. Benchmark and acceptance design

### 15.1 Benchmark 원칙

- frozen population/query/model/runtime/hardware profile
- exact denominator와 terminal conservation
- deterministic tie-break and report schema
- baseline과 candidate를 같은 snapshot/workload에서 비교
- average뿐 아니라 p50/p95/p99, peak resource와 error rate 보고
- 95% bootstrap confidence interval 또는 exact rational score
- scoped PASS를 전체 acceptance로 승격하지 않음

### 15.2 Inventory/canonical quality

| Gate | Threshold |
|---|---:|
| Population terminal conservation | 100% |
| Accepted source hash binding | 100% |
| Canonical required-field validity | 100% |
| Accepted element provenance completeness | 100% |
| Full-population provenance completeness | ≥95% |
| JSONL/Parquet row and ID-set parity | 100% |
| Repeated-build canonical digest parity | 100% |

### 15.3 PDF extraction quality

| PDF class | NID | TEDS | MHS | Additional gate |
|---|---:|---:|---:|---|
| Born-digital local | ≥0.95 | ≥0.90 | ≥0.90 | Markdown+JSON closure, page coverage 100% |
| Complex/scanned selected local-hybrid | ≥0.90 | ≥0.85 | ≥0.85 | Korean OCR/formula/table golden omissions 없음 |

Metric이 applicable하지 않으면 `N/A`를 명시하고 PASS로 보정하지 않는다. Edu golden reviewer는 heading/table/formula/image/reading-order의 critical omission을 검사한다. Critical omission 1건도 accepted가 아니다.

OpenDataLoader upstream benchmark 수치는 dependency selection evidence이지 edu acceptance 자체가 아니다. 동일 golden corpus에서 local/hybrid 실행 결과를 다시 측정한다.

### 15.4 Retrieval and citation quality

| Metric | Golden threshold |
|---|---:|
| nDCG@10 | ≥0.80 |
| MRR@10 | ≥0.80 |
| Recall@20 | ≥0.90 |
| Important-claim citation precision | 100% |
| Important-claim citation coverage for change/decision | 100% |
| Exploration evidence quality | ≥85% |

BM25 lexical, exact-vector, LanceDB candidate와 hybrid를 같은 query set에서 비교한다. Hybrid acceptance는 threshold를 모두 만족하고 BM25 대비 nDCG@10 또는 Recall@20을 상대 5% 이상 개선하며 다른 핵심 metric을 2%보다 크게 악화시키지 않아야 한다.

Approximate vector adapter recall은 exact NumPy oracle 대비 Recall@20 ≥0.98이어야 한다.

### 15.5 External evaluation

- OpenDataLoader NID/NID-S, TEDS/TEDS-S, MHS/MHS-S
- TREC RAG 2026 development data와 RAGDoll
- official TREC result는 judgments/result가 실제 공개된 후 exact submitted run에만 claim
- RAGPerf는 MLCommons 공식 benchmark가 아닌 pinned external research framework로 분류
- RAGPerf-style embedding/index/retrieval/rerank/generation latency, throughput, CPU/GPU memory, context recall, query accuracy와 factual consistency

External code/data는 별도 checksum/license/egress-gated acquisition 후 local pinned snapshot으로 실행한다.

### 15.6 Local performance SLO

Representative corpus와 full-vault accepted snapshot에서 다음을 측정한다.

| Workload | SLO |
|---|---:|
| Warm lexical search p95 | ≤0.25 s |
| Warm graph 2-hop p95 | ≤0.25 s |
| Warm vector search p95 | ≤0.75 s |
| Warm hybrid search p95 | ≤1.50 s |
| Read/citation p95 | ≤0.30 s |
| Preflight p95 | ≤2.00 s |
| MCP cold start | ≤8 s |
| Hybrid search p99 | ≤3.00 s |
| Search error rate | <1% |
| Steady-state local RSS | ≤4 GiB |
| Single-user hybrid throughput | ≥5 queries/s |

Candidate가 quality를 높이더라도 hybrid p95가 lexical baseline의 2배를 초과하거나 memory SLO를 넘으면 자동 선택하지 않는다. Quality가 우선이지만 operational SLO도 release gate다.

Ingest/rebuild는 pages/s, elements/s, bytes/s, peak disk와 projected full-run duration을 보고한다. Representative corpus에서 full-vault 예상 시간이 24시간을 초과하거나 free-space safety margin 20%를 깨면 batch/configuration을 재설계하고 실행하지 않는다.

### 15.7 Client parity

- identical MCP request의 canonical response digest 동일
- stable source/element/relation/lineage IDs 동일
- citation set와 snapshot commit 동일
- preflight verdict와 limitation 동일
- timeout/cancellation/error code 동일
- client rendering 차이는 evidence identity를 바꾸지 않음

## 16. Test strategy

1. Schema/static gates and fixed identity/locator/relation/lineage vectors
2. Unit/property/boundary/negative fixtures
3. Filesystem syscall/fault injection for lock/create/fsync/reread/CAS/crash
4. Markdown/PDF/notebook/code/data/experiment extraction goldens
5. OpenDataLoader fast/local-hybrid routing and Markdown+JSON closure
6. Canonical/Parquet deterministic rebuild and parity
7. Relation resolution, unresolved/ambiguous/denied semantics
8. Lexical/graph/vector adapter replacement and exact-vector recall oracle
9. MCP protocol, bounds, malformed numeric token, injection, timeout, cancellation, shutdown
10. Codex/Claude/Gemini identical-task E2E
11. Benchmark reproducibility and report-schema validation
12. Backup/restore, crash recovery, retention dry run and forward rollback drill
13. Final full-vault requirement-by-requirement audit

Every PASS report states its exact scope. Empty test selection, stale fixture, prose-only success, unbound benchmark or missing negative cases cannot close a gate.

## 17. Stage acceptance sequence

### Stage 1 — Inventory and registry

- whole-vault policy-safe metadata census
- representative corpus hashes and external exact manifest
- source registry successor schema and stable allocations
- corrected ADR-002 durable publication runtime
- crash/containment/population tests
- activation remains false

### Stage 2 — Extraction and canonicalization

- every selected PDF materialized as OpenDataLoader Markdown+JSON bundle
- Markdown/notebook/code/data/experiment extractors
- source/element/relation/lineage JSONL
- deterministic typed Parquet
- quality routing and quarantine

### Stage 3 — Search and graph adapters

- SQLite FTS5 and graph adjacency
- NumPy exact vector oracle
- model bake-off and LanceDB scale candidate
- deterministic hybrid RRF
- retrieval/citation golden PASS

### Stage 4 — Read-only MCP and clients

- stable MCP SDK v1 and protocol profile
- four exact tools and snapshot pinning
- three thin client configurations
- security, bounds, injection, parity and lifecycle tests
- permanent activation still false

### Stage 5 — Evaluation and operations

- edu golden, PDF, retrieval/citation, external and RAGPerf-style reports
- three-client E2E parity
- performance/resource SLO
- telemetry/retention/backup/restore/recovery/rollback drills
- closed `release_accepted` report
- authorized final publication, client activation and live smoke

### Full-vault expansion

Vertical slice PASS 후 deterministic batches로 Stage 1B→5를 반복한다. 모든 allowed assets, unresolved terminal states와 frozen denominator를 최종 보고서에 포함한다.

## 18. Completion definition

Goal completion은 다음이 모두 current evidence로 증명될 때만 가능하다.

- Stage 1–5 implementation이 live worktree/runtime에 존재
- representative corpus와 full vault가 동일 pipeline을 통과
- 모든 allowed PDF의 accepted OpenDataLoader Markdown+JSON bundle; unresolved PDF quarantine가 하나라도 남으면 completion 불가
- full inventory/population conservation과 ≥95% provenance
- canonical/Parquet/index closure와 deterministic rebuild
- retrieval/citation/external/performance benchmark gates
- identical three-client IDs/citations/preflight
- security, backup/restore, crash, retention and rollback drills
- no source mutation, remote egress, `.omo` execution 또는 unrelated dirty-path damage
- current `release_accepted` head와 post-activation live smoke
- unresolved failures, missing evidence, false PASS 또는 scope-narrowing이 없음

Golden slice, passing unit tests, a populated registry, installed package, client version presence 또는 a single successful query alone는 completion proof가 아니다.
