# Ultraresearch Synthesis: LinkedIn Profile Claim Review

## Scope

Adversarial review of proposed LinkedIn direction:

`AI/ML & Software Engineering Student | LLM/MLOps · Agentic AI · GovTech | UMMAYA · GovOn · DigitalPublishing`

Evidence boundary: public-source logic, public web surfaces, and prior local evidence summaries only. No private repository data was browsed.

## Verdict

Fail as written. The direction is directionally accurate but too dense and implies broader production, organizational, and MLOps maturity than the public evidence safely supports.

## Safe Claims

- AI/ML and software engineering student.
- Built UMMAYA as an open-source terminal AI agent for Korean public-service workflows.
- UMMAYA uses K-EXAONE through FriendliAI Serverless and exposes public-service tool families.
- GovOn Hugging Face model/dataset work exists publicly under the user namespace.
- DigitalPublishing includes a public repository and a live demo link for the mobile wedding invitation project.
- IlluOps may be described only as planning/reference/evidence work, not as shipped product work.

## Risky Claims

- `MLOps` in the headline: risky unless tied to public experiment tracking, reproducible training/eval artifacts, or a public W&B report.
- `GovTech` in the headline: acceptable as interest/project domain, but should not imply government affiliation or official service authority.
- `GovOn` without qualifier: risky because it can imply organizational ownership or official deployment.
- `W&B public GovOn URL`: remove unless the exact public URL is linked and accessible.
- `IlluOps` in About without qualifier: remove or soften to planning/evidence/reference.
- `live demo` for DigitalPublishing: safe only as `live web demo`, not production or client deployment.

## Recommended Wording

Headline:

`AI/ML & Software Engineering Student | LLM Apps · Tool-Calling Agents · Civic/Public-Service AI | UMMAYA · GovOn HF · DigitalPublishing`

About wording:

`I build student-led AI and software projects around Korean public-service workflows, LLM tool-calling, and web publishing. My current work includes UMMAYA, an open-source terminal AI agent for Korean public-service workflows using K-EXAONE through FriendliAI; GovOn-related Hugging Face models/datasets for Korean civil-complaint assistance; and DigitalPublishing, a public web publishing/course project with a live mobile invitation demo. I also maintain IlluOps planning/reference/evidence work for cross-agent image workflow research, which I keep clearly separate from shipped product claims.`

## Sources Checked

- Public GitHub UMMAYA README and model/license section.
- Public Hugging Face GovOn model and dataset pages.
- Public GitHub DigitalPublishing README and demo link.
- Public GitHub IlluOps repository metadata.
- Public search for W&B GovOn-specific URL; no concrete public GovOn W&B URL surfaced.

## EXPAND

- LEAD: W&B profile exists but no GovOn-specific public run/report URL surfaced — WHY: headline/About should not mention W&B as evidence unless it is directly clickable and public — ANGLE: search exact `wandb.ai/<entity>/<project>` URLs or inspect W&B public profile manually.
- LEAD: GovOn public roadmap and Hugging Face model pages exist separately — WHY: profile should distinguish personal HF artifacts from GovOn organization/runtime claims — ANGLE: compare exact ownership/role wording across GovOn README, issues, HF model cards, and GitHub org membership.
- LEAD: DigitalPublishing demo link exists in README but live browser behavior was not re-tested in this pass — WHY: `live demo` is safe only if URL renders now — ANGLE: open `https://ourseason.pages.dev/` and capture current desktop/mobile screenshots.
