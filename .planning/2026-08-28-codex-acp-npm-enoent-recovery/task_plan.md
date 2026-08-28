# Task Plan: Codex ACP npm ENOENT recovery

## Goal
Restore the desktop app's Codex ACP startup and verify that its configured launcher completes initialization without the broken npx cache error.

## Next Step
Deliver the verified recovery and tell the user to start a fresh Codex thread in OpenKnowledge.

## Current Phase
Phase 5

## Phases

### Phase 1: Failure capture and discovery
- [x] Record the exact reported ENOENT and ACP symptom
- [x] Inspect the npm debug log and broken cache entry
- [x] Identify the configured ACP launcher and environment
- **Status:** complete

### Phase 2: Root-cause diagnosis
- [x] Classify cache, launcher, PATH, or package-version failure
- [x] Choose the smallest reversible recovery
- **Status:** complete

### Phase 3: Contained recovery
- [x] Preserve evidence and repair only the faulty cache/config surface
- [x] Reinstall or pin the launcher only if directly supported
- **Status:** complete

### Phase 4: Testing and verification
- [x] Run the exact launcher outside the app
- [x] Verify ACP initialization or equivalent protocol startup
- [x] Confirm no new npm ENOENT log is produced
- **Status:** complete

### Phase 5: Delivery
- [x] Record root cause, changes, and verification evidence
- [x] Deliver fresh-thread instruction; application restart is not required by the verified launcher state
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Start with live launcher and log evidence | The npx cache path is a symptom; launcher ownership determines the safe fix. |
| Quarantine the exact cache directory instead of synthesizing missing files | The interrupted install also lacks `.bin` and lock metadata, so a manual manifest would preserve corruption. |
| Validate with a real ACP SDK initialize request | Process startup alone would not prove protocol readiness. |

## Errors Encountered
| Error | Resolution |
|-------|------------|
| init-session.sh was not executable | Invoked the same provided script through `sh`; initialization succeeded. |
| Broad `rg` over `~/.codex` generated an oversized result | Exclude session archives and search only app-owned bounded files. |
| Non-TTY session closed stdin before an interactive handshake could be written | Replaced it with an SDK client that owns the child process pipes. |
| A preliminary interactive probe remained running after stdin closed | Terminated only that diagnostic npm process tree after the successful SDK check. |
