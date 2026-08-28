# Wave 1 - Codebase Local Layout

Worker: explorer `019efe1a-0fa3-7a22-9be1-52254aa1416f`

## Findings

- `LG Aimers 9기.md` is the cohort hub and already indexes lecture material sections.
- Existing top-level notes are `LG Aimers 9기.md`, `LG Aimers 9기 평가 및 제출 가이드.md`, and `LG Aimers 9기 지원서 초안.md`.
- The correct place for a canonical source record is a top-level sibling note, not `.omo/` and not `강의자료/`.
- Before this run, `https://academy.lgresearch.ai/study` only appeared in the new `.omo` journal.

## EXPAND

- LEAD: no direct `academy.lgresearch.ai/study` content was found outside `.omo` planning state — WHY: the local vault may only contain the downloaded artifacts and not the upstream page snapshot — ANGLE: search adjacent Obsidian folders or the browser/download cache for imported academy exports or saved page captures
- LEAD: graph-facing target notes live outside the current folder under `ComputerScience/00_graph-interfaces/...` — WHY: the hub note links to those targets but they were not directly enumerated here — ANGLE: inspect the linked graph interface and evidence index notes to see whether a dedicated LG Aimers 9기 source record already exists there
- LEAD: the notebook/pdfs in `강의자료/` may have companion markdown summaries or extraction notes elsewhere in the vault — WHY: the current folder only exposes raw lecture assets, not downstream study notes — ANGLE: search for exact lecture titles plus `요약`, `정리`, `노트`, or `핵심` in neighboring vault areas
