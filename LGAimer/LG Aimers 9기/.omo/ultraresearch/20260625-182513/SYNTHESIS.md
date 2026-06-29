# Ultraresearch Synthesis: LG Aimers 9기 study materials

Workers: 6 spawned, 4 completed, 2 model-capacity failures; main thread covered failed lanes. Waves: 2. Sources: local vault, official public site, academy static assets, browser login surface, unauthenticated API probes. Verifications: local file hash/PDF checks, notebook parse, HTTP/API status checks.

## Executive summary

The LG Aimers 9기 local vault now has a canonical source-of-truth note at `LG Aimers 9기 강의자료 원소스.md`, linked from `LG Aimers 9기.md`. The note records the official public curriculum roster, the local material inventory, the verified file status, and the upstream academy access boundary.

The public LG Aimers homepage confirms the six Phase I modules and instructors. The private academy study URL is not a public material index: direct HTML returns a CRA shell, browser rendering redirects to `/login`, and course/content/file signed-url API probes return 401 without authentication.

## Findings by theme

### Local material state

Confirmed usable local material:

- 4 PDFs: Mathematics for ML, 지도학습, LLM Application & Evaluation, 딥러닝 자연어처리 기초와 LLM Agent.
- 1 notebook: `00_Hands_on_Tabular_ML.ipynb`.

Confirmed non-usable local placeholders:

- 6 Tabular ML PDFs are Git LFS pointer text files.
- 2 Optimization / Time-Series PDFs are Git LFS pointer text files.

### Public curriculum

The official public site `https://www.lgaimers.ai/` lists the 9기 modules and instructors. Those titles match the local `강의자료` structure.

### Academy study page

`https://academy.lgresearch.ai/study` is a private app route. It exposes static build artifacts, including `asset-manifest.json`, but the study content and file signed URLs require authentication.

## Codebase findings

- Added `LG Aimers 9기 강의자료 원소스.md`.
- Updated `LG Aimers 9기.md` to `updated: '2026-06-25'`.
- Linked the new canonical source note under `## 진행 노트`.

## Sources

1. Local file inventory: `강의자료/`
2. Hub note: `LG Aimers 9기.md`
3. Official public curriculum: `https://www.lgaimers.ai/`, accessed 2026-06-25
4. Academy study route: `https://academy.lgresearch.ai/study`, accessed 2026-06-25
5. Academy asset manifest: `https://academy.lgresearch.ai/asset-manifest.json`, accessed 2026-06-25
6. Saved HTTP artifacts: `.omo/ultraresearch/20260625-182513/http-study.html`, `http-study.headers`, `http-main-js.headers`, `asset-manifest.json`, `chunks/`

## Verified claims

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Four top-level lecture PDFs are readable PDFs | CONFIRMED | `file`, `pdfinfo`, SHA256 checks |
| The Tabular ML notebook is parseable JSON notebook | CONFIRMED | `nbformat 4`, 37 cells |
| Eight PDFs are Git LFS pointers, not real PDFs | CONFIRMED | 131-132 byte files with Git LFS pointer headers |
| `academy.lgresearch.ai/study` requires login for study content | CONFIRMED | browser redirected to `/login`; API returned 401 |
| Public unauthenticated sources expose direct 9기 lecture PDF URLs | REFUTED for this run | public search and static bundle scan found no direct PDF URLs |

## Gaps

- Actual authenticated courseId/contentId/fileId/signed URL values remain unavailable until a logged-in academy session is provided.
- The 8 Git LFS pointer PDFs cannot be restored from this folder because it is not currently recognized as a Git repository.

## Expansion trace

- Wave 1 covered local layout, file inventory, public search, static asset probing, and course roster reconciliation.
- Two external workers failed due model capacity; the main thread completed their direct HTTP/API and roster checks.
- Wave 2 expanded the static bundle lead by downloading 112 chunks and probing unauthenticated API endpoints.
- Convergence reason: no unchecked public/unauthenticated leads remain; remaining work requires authenticated academy access.
