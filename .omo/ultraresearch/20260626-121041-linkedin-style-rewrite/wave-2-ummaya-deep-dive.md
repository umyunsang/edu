# Wave 2: UMMAYA Deep Dive

UMMAYA is a Python + Node terminal AI agent for Korean public-service workflows. The accurate public framing is "bounded civic/public-service agent", not generic chatbot and not official government system.

## Strong Claims

- UMMAYA uses a small verb surface: `find`, `locate`, `check`, `send`, `document`.
- The project focuses on public-service outcomes where the user should not need to know the owning ministry, portal, credential rail, or API.
- Trust boundaries are first-class: hidden authority, fake completion, credential bypass, and unlabeled mock behavior are explicitly disallowed.
- `check`, `send`, and `document` are gated; `find` and `locate` are the safer automatic surfaces.
- Published distribution exists through npm, a project Homebrew tap, and release artifacts, but metadata still labels the project alpha.

## Safe LinkedIn Framing

Best angle:

> UMMAYA ports a Claude Code-like interaction model into civic/public-service workflows, but refuses to pretend it has authority it does not have.

Use:

- "alpha"
- "terminal AI agent"
- "bounded actions"
- "visible handoff"
- "trust boundary"

Avoid:

- Official government system.
- Fully production-ready.
- All workflows are live.
- Official Homebrew cask acceptance.

## Proof Links

- https://github.com/umyunsang/UMMAYA
- https://ummaya-docs.pages.dev/en/
- https://github.com/umyunsang/UMMAYA/blob/main/assets/ummaya-demo.mp4
- https://ummaya-docs.pages.dev/en/trust/what-ummaya-will-not-do/
