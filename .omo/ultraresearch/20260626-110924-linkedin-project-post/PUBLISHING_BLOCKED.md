# LinkedIn Publishing Status

Status: blocked before external side effect.

I did not publish to LinkedIn from this environment.

## Why

The compliant official path requires a LinkedIn developer/OAuth setup with `w_member_social`. Current local checks found no configured LinkedIn token, no Composio connection, and no exact Codex LinkedIn connector. The available `linkedin-cli` registry hits do not provide a credential-free publish path.

I also did not use browser automation to post, because LinkedIn Help states that third-party software, browser plug-ins, bots, and other unauthorized automation must not automate website activity, including creating or sharing posts.

## Evidence

- Local CLI/env check: `evidence/linkedin-cli-env.txt`
- GitHub/user project evidence: `evidence/github-repos.json`, `evidence/github-user.json`
- Official Share on LinkedIn docs: https://learn.microsoft.com/en-us/linkedin/consumer/integrations/self-serve/share-on-linkedin
- Official API access docs: https://learn.microsoft.com/en-us/linkedin/shared/authentication/getting-access
- LinkedIn automation policy help: https://www.linkedin.com/help/linkedin/answer/a1340567
- LinkedIn prohibited software help: https://www.linkedin.com/help/linkedin/answer/a1341387
- Third-party OAuth CLI example: https://github.com/alexey-pelykh/linkedctl
- Managed OAuth CLI example: https://composio.dev/toolkits/linkedin/framework/cli

## Publishable artifact

Use `DRAFT.md` as the ready-to-paste LinkedIn post. Actual direct publishing can proceed only after one of these is available:

- A LinkedIn OAuth access token with `w_member_social` and a known person URN.
- A configured managed LinkedIn connector/CLI account, such as Composio, already authenticated by the user.
- Manual user posting in the LinkedIn UI.
