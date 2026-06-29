# LinkedIn Drafts Adversarial Review

## Verdict

FAIL for immediate posting quality.

The drafts are not factually broken: public GitHub links resolve, private `Primer` is not leaked, and no placeholder markers remain. The weakness is credibility and conversion. Several claims are linked only to broad repo roots, while the specific artifacts that prove the claim are one click deeper or not linked at all.

## Findings

### P1: DigitalPublishing post does not link the actual artifact it describes

- File: `.omo/ultraresearch/20260626-110924-linkedin-project-post/MULTI_POST_DRAFTS.md:54`
- Current issue: The post claims a WebGL/mobile invitation experience but links only the parent repository at line 63.
- Evidence: `DigitalPublishing/README.md:20` names the specific folder; `DigitalPublishing/README.md:29-33` gives the live demo URL; `mobile-wedding-unrolling-invitation/README.md:13-14` gives both the live demo and folder link.
- Risk: A LinkedIn reader lands on a broad class repo and may not find the exact work. The claim is true but under-supported.
- Fix: Add the live demo `https://ourseason.pages.dev/` and folder link `https://github.com/umyunsang/DigitalPublishing/tree/main/mobile-wedding-unrolling-invitation` to post 3.

### P1: UMMAYA post links install surface but not evaluation surface

- File: `.omo/ultraresearch/20260626-110924-linkedin-project-post/MULTI_POST_DRAFTS.md:42`
- Current issue: The links are `UMMAYA` repo and Homebrew tap only.
- Evidence: `UMMAYA/README.md:7` exposes docs; `UMMAYA/README.md:20-24` exposes a demo GIF/MP4; `UMMAYA/README.md:64-70` explains first-run credential and boundary behavior.
- Risk: For a public-service AI agent claim, evaluators need docs/demo before an install tap. Homebrew tap is secondary.
- Fix: Add `https://ummaya-docs.pages.dev/en/` and the demo MP4/GIF link; keep Homebrew tap only if the post is explicitly about distribution.

### P1: Evidence-first post implies public QA/review evidence that readers cannot inspect

- File: `.omo/ultraresearch/20260626-110924-linkedin-project-post/MULTI_POST_DRAFTS.md:112`
- Current issue: The line groups "QA logs, review records, deployed screens" as if they are all linked from the post.
- Evidence: Much of the QA/review evidence for this task lives under local `.omo/evidence`, which is not a public social-proof surface. `IlluOps/AGENTS.md:46-50` also says local `.omo` is ignored/private control-plane state.
- Risk: The post can sound over-claimed because the public reader cannot verify those logs.
- Fix: Rephrase to "공개 저장소, 로컬 검증 기록, 배포된 화면..." or link only public-safe evidence such as `IlluOps/references/claim_support_matrix.*` if that is the intended proof surface.

### P1: IlluOps link overstates product maturity unless framed as planning/evidence

- File: `.omo/ultraresearch/20260626-110924-linkedin-project-post/MULTI_POST_DRAFTS.md:124`
- Current issue: `IlluOps` is linked in an engineering-process post but the body does not explain that it is not a shipped app.
- Evidence: `IlluOps/AGENTS.md:8-10` says it is a planning, evidence, and reference repo with no live `src/`, package manifest, CI workflow, or runnable product implementation in the tracked root.
- Risk: A reader may interpret the link as a finished LLM harness product, then click into a repo that explicitly says it is not runnable yet.
- Fix: Either remove the IlluOps link from this post, or rewrite the mention as "IlluOps처럼 아직 제품 이전 단계의 planning/evidence repo도 검증 가능한 경계로 관리한다."

### P2: LG Aimers post lacks a direct source link

- File: `.omo/ultraresearch/20260626-110924-linkedin-project-post/MULTI_POST_DRAFTS.md:100`
- Current issue: It links only `edu` root and GitHub profile, not the LG Aimers notes.
- Evidence: Relevant public note candidates exist at:
  - `https://github.com/umyunsang/edu/blob/main/LGAimer/LG%20Aimers%208%EA%B8%B0/LG%20Aimers%208%EA%B8%B0.md`
  - `https://github.com/umyunsang/edu/blob/main/LGAimer/LG%20Aimers%209%EA%B8%B0/LG%20Aimers%209%EA%B8%B0.md`
- Risk: The post asks readers to trust a specific learning claim but gives only a broad archive link.
- Fix: Link the note or LGAimer folder. Avoid linking certificates or course PDFs directly unless intentionally public.

### P2: Portfolio post has too many equal-weight links

- File: `.omo/ultraresearch/20260626-110924-linkedin-project-post/DRAFT.md:15`
- Current issue: The representative post lists five links with equal visual weight.
- Risk: LinkedIn typically gives one dominant preview; too many links dilute the call-to-action.
- Fix: Pick one primary link, probably GitHub profile or UMMAYA, then mention "관련 프로젝트: ..." with 2-3 secondary links.

### P2: "Portfolio Thread" label is platform-mismatched

- File: `.omo/ultraresearch/20260626-110924-linkedin-project-post/MULTI_POST_DRAFTS.md:3`
- Current issue: "Thread" suggests X/Twitter-style sequencing, not a LinkedIn standalone post set.
- Fix: Rename to "Portfolio Post" or "Portfolio Overview".

## Checks That Passed

- Public GitHub links returned HTTP 200:
  - `https://github.com/umyunsang`
  - `https://github.com/umyunsang/UMMAYA`
  - `https://github.com/umyunsang/DigitalPublishing`
  - `https://github.com/umyunsang/edu`
  - `https://github.com/umyunsang/IlluOps`
  - `https://github.com/umyunsang/homebrew-ummaya`
- `Primer` private repo link is not present in the drafts.
- Placeholder/private/token scans found no matches for `TODO`, `TBD`, `PLACEHOLDER`, `Primer`, `.omo/`, `token`, `LINKEDIN`, or `COMPOSIO`.
- Each multi-post draft is short enough for LinkedIn posting.

## Recommended Fix Order

1. Add deep links for UMMAYA docs/demo and DigitalPublishing live demo/folder.
2. Reframe or remove IlluOps from public-facing links unless explicitly described as planning/evidence.
3. Replace unverifiable "QA logs/review records" wording with a public/private-safe distinction.
4. Add direct LG Aimers note links or remove the specificity from that post.
5. Reduce the representative post to one primary CTA link.
