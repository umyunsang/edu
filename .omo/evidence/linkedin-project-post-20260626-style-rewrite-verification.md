# LinkedIn Project Post Style Rewrite Verification

Date: 2026-06-26 KST

## Scope

Verified rewritten LinkedIn-ready drafts:

- `.omo/ultraresearch/20260626-110924-linkedin-project-post/DRAFT.md`
- `.omo/ultraresearch/20260626-110924-linkedin-project-post/MULTI_POST_DRAFTS.md`

## Checks

### Leakage and Placeholder Scan

Command:

```sh
rg -n "TODO|TBD|PLACEHOLDER|수정 필요|\\.omo/|Authorization|token|api_key|w_member_social|PRIVATE|COMPOSIO" \
  .omo/ultraresearch/20260626-110924-linkedin-project-post/DRAFT.md \
  .omo/ultraresearch/20260626-110924-linkedin-project-post/MULTI_POST_DRAFTS.md
```

Result: no matches.

### LinkedIn Character and Hashtag Bounds

- Recommended single post: 1070 chars, 3 hashtags.
- Multi-post 1: 834 chars, 3 hashtags.
- Multi-post 2: 728 chars, 3 hashtags.
- Multi-post 3: 839 chars, 3 hashtags.
- Multi-post 4: 686 chars, 3 hashtags.
- Multi-post 5: 826 chars, 3 hashtags.
- Multi-post 6: 671 chars, 3 hashtags.

All posts are below LinkedIn's 3,000-character post text limit and use no more than 3 hashtags.

### Public URL Status

All URLs in the rewritten drafts returned HTTP 200:

- https://huggingface.co/umyunsang/govon-civil-adapter
- https://github.com/umyunsang
- https://ummaya-docs.pages.dev/en/
- https://wandb.ai/umyun3/GovOn
- https://ourseason.pages.dev/
- https://github.com/umyunsang/edu
- https://huggingface.co/umyunsang
- https://github.com/umyunsang/UMMAYA
- https://github.com/umyunsang/UMMAYA/blob/main/assets/ummaya-demo.mp4
- https://ummaya-docs.pages.dev/en/trust/what-ummaya-will-not-do/
- https://huggingface.co/datasets/umyunsang/govon-civil-response-data
- https://github.com/umyunsang/DigitalPublishing/tree/main/mobile-wedding-unrolling-invitation
- https://github.com/umyunsang/DigitalPublishing
- https://github.com/umyunsang/edu/blob/main/LGAimer/LG%20Aimers%208%EA%B8%B0/LG%20Aimers%208%EA%B8%B0.md
- https://github.com/umyunsang/edu/blob/main/LGAimer/LG%20Aimers%209%EA%B8%B0/LG%20Aimers%209%EA%B8%B0.md
- https://github.com/umyunsang/IlluOps
- https://github.com/umyunsang/IlluOps/blob/main/references/claim_support_matrix.tsv

### Posting Surface

Direct LinkedIn publishing remains blocked in this environment:

- No LinkedIn posting connector/tool is exposed by `tool_search`.
- No installable LinkedIn plugin was present in the plugin candidate list.
- Browser automation was not used as a posting fallback.

Output status: ready-to-paste drafts, not a completed LinkedIn publication.
