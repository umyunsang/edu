# Wave 1: LinkedIn Publishing Feasibility

Worker: Librarian `019f01b2-6b87-7b72-87bc-c553f61d6d81`

## Key Findings

- Official personal posting requires LinkedIn OAuth and the `w_member_social` permission.
- The self-serve Share on LinkedIn flow posts through the UGC Posts API.
- Local environment checks did not find a configured LinkedIn/Composio CLI or LinkedIn token.
- Third-party CLIs/MCP servers exist, but the credible ones still require a LinkedIn developer app, OAuth, or managed account connection.
- Browser automation is not a compliant publishing path under LinkedIn Help guidance.

## Evidence

- `evidence/linkedin-cli-env.txt`
- LinkedIn Share on LinkedIn docs: https://learn.microsoft.com/en-us/linkedin/consumer/integrations/self-serve/share-on-linkedin
- LinkedIn API access docs: https://learn.microsoft.com/en-us/linkedin/shared/authentication/getting-access
- LinkedIn automated activity help: https://www.linkedin.com/help/linkedin/answer/a1340567
- LinkedIn prohibited software help: https://www.linkedin.com/help/linkedin/answer/a1341387
- LinkedCtl docs: https://github.com/alexey-pelykh/linkedctl
- Composio LinkedIn CLI docs: https://composio.dev/toolkits/linkedin/framework/cli

## EXPAND

- DEAD END: Credential-free official LinkedIn CLI. No official source found.
- DEAD END: Browser automation publishing. LinkedIn help pages prohibit automated website activity.
- LEAD: Official API or managed OAuth connector can publish after user-provided auth. WHY: this is the only compliant path. ANGLE: request `w_member_social` token or Composio/LinkedIn connected account if the user wants actual posting.
