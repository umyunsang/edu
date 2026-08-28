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
- OpenKnowledge's live registry cache pins Codex ACP to `@agentclientprotocol/codex-acp@1.7.0` via npx (`.ok/local/acp-registry-cache.json`, lines 211-225).
- The effective failing command was `npm exec --yes -- @agentclientprotocol/codex-acp@1.7.0` from cwd `/Users/um-yunsang/.ok/acp-npx-cwd`.
- The first install log ended abruptly during dependency extraction without a normal npm exit footer. The cache contains 1,151 package files but lacks root `package.json`, both lockfiles, and `node_modules/.bin`.
- Package/directory ownership and modes are normal; this is an interrupted npm reify state, not a permissions failure.
- OpenKnowledge server logs directly associate the reported cache path with `agentId: codex-acp` and repeated initialize failures.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Rename the exact partial cache and let npm rebuild it | Reversible, scoped to the proven faulty state, and restores all derived metadata together. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Planning initializer lacked execute permission | Ran it with `sh` rather than changing installed skill permissions. |
| Broad recursive `rg` entered large Codex session archives and produced ~700 MB of output | Abandon that search surface; restrict subsequent searches to OpenKnowledge config/log files with file-size and path exclusions. |

## Resources
-
