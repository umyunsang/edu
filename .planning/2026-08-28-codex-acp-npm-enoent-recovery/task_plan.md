# Task Plan: OpenKnowledge ACP npm ENOENT recovery

## Goal
Restore both Codex and Claude ACP startup in OpenKnowledge and verify each registry launcher completes initialization without broken npx cache errors.

## Next Step
Capture the new Claude failure, identify its exact registry package, and inspect only cache `fca12915ff656968` before recovery.

## Current Phase
Phase 6

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

### Phase 6: Claude failure capture and diagnosis
- [ ] Inspect the reported npm log and partial cache structure
- [ ] Confirm OpenKnowledge's exact Claude ACP package and launcher cwd
- [ ] Choose the smallest reversible recovery
- **Status:** in_progress

### Phase 7: Claude contained recovery
- [ ] Quarantine only the proven partial Claude npx cache
- [ ] Rebuild it with the exact registry-pinned launcher
- **Status:** pending

### Phase 8: Claude ACP verification
- [ ] Verify rebuilt cache metadata and executable link
- [ ] Complete a real ACP initialize handshake
- [ ] Confirm no new npm ENOENT and no diagnostic processes remain
- **Status:** pending

### Phase 9: Claude delivery
- [ ] Record root cause, recovery, and scoped verification evidence
- [ ] Deliver fresh Claude-thread instruction
- **Status:** pending

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
