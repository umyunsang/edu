# Ultraresearch Synthesis: LinkedIn Project Post

Workers: 6 spawned, 4 completed, 2 silent/inconclusive · Waves: 1 plus root expansion · Sources: local repos, GitHub CLI/API, edu vault, LinkedIn docs · Verifications: CLI/env and claim ledger

## Executive Summary

The evidence supports a LinkedIn post about moving from coursework and prototypes into evidence-first AI/software execution. The strongest public anchors are `UMMAYA`, `DigitalPublishing`, and `edu`; `IlluOps` can be mentioned as research/planning, while `Primer` should inform the process framing without exposing private details.

Direct LinkedIn publishing is blocked in this environment. Official LinkedIn posting requires OAuth with `w_member_social`, no LinkedIn connector was available through Codex plugin discovery, no local LinkedIn/Composio credential was found, and browser automation would be the wrong channel under LinkedIn's own automation guidance. The deliverable is therefore a ready-to-paste Korean LinkedIn post in `DRAFT.md` plus the blocker evidence in `PUBLISHING_BLOCKED.md`.

## Findings by Theme

### 1. Public project anchors

- `UMMAYA` is a public TypeScript repo and a strong product anchor. The README describes it as a terminal AI agent for Korean public-service workflows and positions it around civic adapters, tool progress, and identity/consent/payment/authority boundaries. Evidence: `/Users/um-yunsang/UMMAYA/README.md:12-16`, `evidence/github-repos.json`.
- `DigitalPublishing` is a public visual/design anchor. The README lists a Three.js/WebGL mobile invitation project and a World Design Capital Busan prototype, including a shareable demo URL for the invitation. Evidence: `/Users/um-yunsang/Documents/DigitalPublishing/README.md:16-33`, `evidence/github-repos.json`.
- `IlluOps` is public but should be framed as research/planning rather than shipped product. Its project knowledge base says it is not a conventional application tree yet and has no live `src` or runnable implementation in the tracked root. Evidence: `/Users/um-yunsang/IlluOps/AGENTS.md:8-10`, `evidence/github-repos.json`.
- `Primer` is recent and evidence-rich but private. It supports the workflow theme because its instructions define `.omo` planning, ULW execution, evidence gate, and review gate as the authoritative workflow. Evidence: `/Users/um-yunsang/Primer/README.md:3-14`, `/Users/um-yunsang/Primer/AGENTS.md:7-23`, `evidence/github-repos.json`.

### 2. Learning archive as an evidence trail

- The `edu` vault describes itself as a three-year Computer Science and AI curriculum archive that connects theory, implementation, and external activities. Evidence: `/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu/README.md:24-40`.
- Its interface folders cover AI/ML/data, systems/infrastructure, software engineering, algorithms/graphics, and portfolio/professional humanities. Evidence: `/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu/README.md:149-158`.
- LG Aimers is the strongest external-learning proof point. The 8th-cycle note explicitly groups LLM Compression and EXAONE model-lightening materials; the 9th-cycle draft explains the shift toward runtime, reproducibility, explainability, and domain context. Evidence: `/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu/LGAimer/LG Aimers 8기/LG Aimers 8기.md:38-57`, `/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu/LGAimer/LG Aimers 9기/LG Aimers 9기 지원서 초안.md:46-64`.

### 3. Credible LinkedIn framing

- The strongest thesis is not "AI made everything faster"; it is "AI work becomes valuable when it is measurable, reproducible, reviewable, and tied to real delivery surfaces." This is supported by the local project evidence, the edu archive, and the completed framing worker.
- The post should be concise, first-person, and low-hype. It should name concrete artifacts (`UMMAYA`, `DigitalPublishing`, `edu`) and one process lesson.

### 4. Publishing feasibility

- LinkedIn's official Share on LinkedIn docs require OAuth and `w_member_social` for creating posts on behalf of an authenticated member, and text shares go through the UGC Posts API. Sources: Microsoft Learn Share on LinkedIn and API access docs.
- LinkedIn Help pages prohibit third-party software, browser plug-ins, bots, or other unauthorized automation that automates activity on LinkedIn's website. Sources: LinkedIn Help automation/prohibited software pages.
- Third-party tools such as LinkedCtl and Composio may be viable only after user authentication or OAuth setup; they are not credential-free shortcuts.
- Local checks found no `LINKEDIN_*`, `COMPOSIO_*`, `linkedctl`, `composio`, or exact Codex LinkedIn connector already configured. Evidence: `evidence/linkedin-cli-env.txt`, plugin discovery session output.

## Sources Ranked

1. `evidence/github-repos.json`: current `gh repo list` inventory for `umyunsang`.
2. `evidence/github-user.json`: authenticated GitHub profile snapshot.
3. `evidence/local-repo-git-summaries.txt`: local remotes and latest commits for `Primer`, `UMMAYA`, `IlluOps`, and `DigitalPublishing`.
4. `evidence/edu-signals.txt`: vault-wide keyword scan.
5. Local source files cited above.
6. LinkedIn official/Microsoft Learn docs:
   - https://learn.microsoft.com/en-us/linkedin/consumer/integrations/self-serve/share-on-linkedin
   - https://learn.microsoft.com/en-us/linkedin/shared/authentication/getting-access
   - https://learn.microsoft.com/en-us/linkedin/marketing/community-management/shares/posts-api?view=li-lms-2026-06
7. LinkedIn Help automation guidance:
   - https://www.linkedin.com/help/linkedin/answer/a1340567
   - https://www.linkedin.com/help/linkedin/answer/a1341387
8. Third-party wrapper references:
   - https://github.com/alexey-pelykh/linkedctl
   - https://composio.dev/toolkits/linkedin/framework/cli

## Verified Claims

See `claim-ledger.md`.

## Contradictions

- Some third-party tools advertise LinkedIn posting from CLI/MCP. This does not contradict the official requirement because those tools still require LinkedIn OAuth, a developer app, direct tokens, or managed account connection.
- Browser UI posting would be technically possible only with an authenticated human session, but automated posting through the website is not the compliant surface.

## Gaps

- No LinkedIn OAuth token, person URN, or connected managed LinkedIn account was present.
- The GitHub public-profile subagent did not return before synthesis; root `gh` evidence covered the required inventory.
- No live LinkedIn post URL exists because publishing was blocked before side effect.

## Expansion Trace

See `expansion-log.md`.
