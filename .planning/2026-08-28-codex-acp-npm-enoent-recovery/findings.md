# Findings & Decisions

## Requirements
- Fix `initialize failed: ACP connection closed` for Codex startup.
- Resolve npm `ENOENT` for `/Users/um-yunsang/.npm/_npx/c8b015f66c7988d7/package.json`.
- Preserve unrelated project and user configuration.
- Fix Claude `initialize failed: ACP connection closed` and the missing `/Users/um-yunsang/.npm/_npx/fca12915ff656968/package.json` without disturbing the verified Codex recovery.

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
- The faulty cache was renamed to `/Users/um-yunsang/.npm/_npx/c8b015f66c7988d7.broken-20260828T1145KST`; it was not deleted.
- A fresh exact launcher run rebuilt the expected root `package.json`, `package-lock.json`, `node_modules/.package-lock.json`, and `.bin/codex-acp` symlink.
- A real ACP SDK initialize request succeeded with protocol version 1 and agent `@agentclientprotocol/codex-acp` version 1.7.0.
- The post-recovery npm log contains no `ENOENT` or npm error code.
- Claude's new failure log confirms the exact command `npm exec --yes -- @agentclientprotocol/claude-agent-acp@0.70.0`, cwd `/Users/um-yunsang/.ok/acp-npx-cwd`, Node 26.7.0, and npm 12.0.2.
- OpenKnowledge's registry cache independently pins `claude-acp` version 0.70.0 to the same npm package with no extra registry arguments.
- Cache `fca12915ff656968` has a populated `node_modules` tree but no root `package.json`, no root/package lock metadata, and no `node_modules/.bin`; this matches the proven interrupted-reify structure from the Codex incident.
- The OpenKnowledge server log associates the reported ENOENT directly with `agentId: claude-acp` and thread `f2e60e46-a587-48af-a2bc-8946c8715522`.
- The partial Claude cache contains 3,968 ordinary files and zero symlinks; its adapter manifest declares `.bin` target `claude-agent-acp` and Node `>=22`, but npm never created the executable link.
- Installed Node 26.7.0 satisfies the adapter engine requirement. Native Claude Code resolves to `/Users/um-yunsang/.local/bin/claude` version 2.1.250, so missing Claude CLI/PATH is not the current failure.
- npm's rolling log cleanup has already removed the original 10:58 install log, so the exact terminating event is unavailable; the interrupted-reify conclusion is an inference from the same complete structural signature as the proven Codex cache failure.
- The partial Claude cache was preserved as `/Users/um-yunsang/.npm/_npx/fca12915ff656968.broken-20260828T1205KST`.
- Re-running the exact pinned launcher rebuilt root manifests, lock metadata, and `.bin/claude-agent-acp`; the installation command exited cleanly with no lingering ACP process.
- A real ACP SDK initialize request succeeded with protocol version 1 and agent `@agentclientprotocol/claude-agent-acp` version 0.70.0.
- The latest post-verification npm log contains no `ENOENT` or npm error code, all diagnostic Claude ACP processes exited, and the live OpenKnowledge server remained running.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Rename the exact partial cache and let npm rebuild it | Reversible, scoped to the proven faulty state, and restores all derived metadata together. |
| Keep stale global `@openai/codex@0.120.0` untouched | PATH resolves the healthy signed Homebrew cask 0.150.1; unrelated cleanup is outside this incident. |
| Keep native Claude and OpenKnowledge configuration unchanged | The failure was before wrapper startup; both runtime and registry compatibility checks passed. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Planning initializer lacked execute permission | Ran it with `sh` rather than changing installed skill permissions. |
| Broad recursive `rg` entered large Codex session archives and produced ~700 MB of output | Abandon that search surface; restrict subsequent searches to OpenKnowledge config/log files with file-size and path exclusions. |

## Resources
-
