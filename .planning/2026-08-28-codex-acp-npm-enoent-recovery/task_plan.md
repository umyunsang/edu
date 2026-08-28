# Task Plan: Codex ACP npm ENOENT recovery

## Goal
Restore the desktop app's Codex ACP startup and verify that its configured launcher completes initialization without the broken npx cache error.

## Next Step
Wait for launcher/toolchain audits, then quarantine the single partial npx cache and rerun the registry-pinned ACP command.

## Current Phase
Phase 2

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
- [ ] Preserve evidence and repair only the faulty cache/config surface
- [ ] Reinstall or pin the launcher only if directly supported
- **Status:** in_progress

### Phase 4: Testing and verification
- [ ] Run the exact launcher outside the app
- [ ] Verify ACP initialization or equivalent protocol startup
- [ ] Confirm no new npm ENOENT log is produced
- **Status:** pending

### Phase 5: Delivery
- [ ] Record root cause, changes, and verification evidence
- [ ] Deliver restart instructions if the GUI must be reopened by the user
- **Status:** pending

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Start with live launcher and log evidence | The npx cache path is a symptom; launcher ownership determines the safe fix. |
| Quarantine the exact cache directory instead of synthesizing missing files | The interrupted install also lacks `.bin` and lock metadata, so a manual manifest would preserve corruption. |

## Errors Encountered
| Error | Resolution |
|-------|------------|
| init-session.sh was not executable | Invoked the same provided script through `sh`; initialization succeeded. |
| Broad `rg` over `~/.codex` generated an oversized result | Exclude session archives and search only app-owned bounded files. |
