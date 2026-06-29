# Claim Ledger

| Claim | Risk | Domains / Evidence | Counter-search | Primary? | Status |
| --- | --- | --- | --- | --- | --- |
| LinkedIn personal posting through the official route requires OAuth and `w_member_social`. | high | Microsoft Learn Share on LinkedIn; Microsoft Learn API access docs | searched CLI/plugin/browser alternatives; none removes OAuth | yes | verified |
| The official Share on LinkedIn flow uses UGC Posts API for text/profile shares. | high | Microsoft Learn Share on LinkedIn | checked Posts API and product catalog | yes | verified |
| This Codex environment has no installed LinkedIn connector/plugin surfaced by `tool_search` or plugin install discovery. | high | session tool output; `evidence/linkedin-cli-env.txt` | searched available plugins for LinkedIn; no exact install candidate | yes, local tool evidence | verified |
| The locally visible `linkedin-cli` npm package is extraction-oriented, not a publishing client. | normal | `npm view linkedin-cli` captured in `evidence/linkedin-cli-env.txt` | checked PyPI and third-party alternatives | yes, registry metadata | verified |
| PyPI `linkedin-cli` can post only after a LinkedIn app/client auth setup. | normal | PyPI project page; `evidence/linkedin-cli-env.txt` | checked official API docs | no, third-party package docs | verified |
| Browser automation should not be used to publish on LinkedIn. | high | LinkedIn Help automated activity; LinkedIn Help prohibited software | checked official API path as alternative | yes | verified |
| UMMAYA is a public terminal AI agent for Korean public-service workflows. | normal | `/Users/um-yunsang/UMMAYA/README.md:12-16`; `evidence/github-repos.json` | checked local git remote and README | yes, repo source | verified |
| DigitalPublishing has a public mobile WebGL invitation project and demo link. | normal | `/Users/um-yunsang/Documents/DigitalPublishing/README.md:16-33`; `evidence/github-repos.json` | checked local git remote and README | yes, repo source | verified |
| The `edu` vault is a three-year CS and AI curriculum map with AI/ML, software engineering, algorithms, and LG Aimers signals. | normal | `/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu/README.md:34-40`, `:154-158`, `:202-215`; `evidence/edu-signals.txt` | searched archive for project/AI/LGAimer signals | yes, local vault source | verified |
| Primer is recent and evidence-rich, but private; public post copy should not expose private implementation details. | high | `evidence/github-repos.json`; `/Users/um-yunsang/Primer/README.md:3-14`; `/Users/um-yunsang/Primer/AGENTS.md:7-23` | checked visibility and local README | yes, local and GitHub evidence | verified |

Unresolved claims:
- None needed for the final post. Any exact LinkedIn post URL remains unresolved because publishing is blocked by missing OAuth/connector.

Refuted claims:
- "A credential-free LinkedIn CLI/plugin can publish this post from the current environment." Refuted by local plugin/CLI checks and official OAuth requirements.
