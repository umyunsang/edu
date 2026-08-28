# Findings & Decisions

## Requirements
- Fix `initialize failed: ACP connection closed` for Codex startup.
- Resolve npm `ENOENT` for `/Users/um-yunsang/.npm/_npx/c8b015f66c7988d7/package.json`.
- Preserve unrelated project and user configuration.

## Research Findings
- Reported failure indicates an `npx` ephemeral install directory exists or is referenced while its root `package.json` is missing.
- The prior session concerned an OpenKnowledge desktop workflow, but current ownership must be verified from live configuration and logs.
- Live process evidence confirms `/Applications/OpenKnowledge.app` version 0.64.1 is running and has opened `/Users/um-yunsang/work/edu` through its bundled local server.
- The exact broken cache directory exists, was modified at 11:35, and contains no root `package.json`.
- A separate Codex CLI 0.150.1 session is healthy enough to run `codex-code-mode-host`; therefore the failing surface is specifically OpenKnowledge's ACP launch attempt, not total loss of the Codex executable.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Planning initializer lacked execute permission | Ran it with `sh` rather than changing installed skill permissions. |
| Broad recursive `rg` entered large Codex session archives and produced ~700 MB of output | Abandon that search surface; restrict subsequent searches to OpenKnowledge config/log files with file-size and path exclusions. |

## Resources
-
