# Wave 1 - Web Public Search

Worker: librarian `019efe1a-9e90-72e0-a539-38b0df8c7700`

## Findings

- `https://www.lgaimers.ai/` is the strongest public source for the LG Aimers 9기 curriculum roster.
- The official public roster lists six modules: Tabular ML, Optimization and Decision-Focused Learning / Time-Series Analysis, 딥러닝 자연어처리 기초와 LLM Agent, Mathematics for ML, LLM Application & Evaluation, and 지도학습.
- Public search did not find 9기 lecture PDF URLs or a public `academy.lgresearch.ai/study` material index.

## EXPAND

- LEAD: direct logged-in or JS-loaded `study` route on `lgaimers.ai`/`academy.lgresearch.ai` — WHY: the public homepage exposes the curriculum but not the file URLs — ANGLE: inspect authenticated network calls or page JS for `/study`, `/api`, and download endpoints
- LEAD: hidden PDF/asset URLs for 9기 lecture materials on LG CDNs — WHY: course materials may be served from a non-indexed CDN even if the study page is private — ANGLE: search `site:lgresearch.ai filetype:pdf "LG Aimers 9기"` and inspect source for asset path patterns
- LEAD: mirrored lecture notes by participants for each module title — WHY: community blogs already echo the module names and may reveal filenames or screenshots — ANGLE: search each module title with `tistory`, `naver`, `velog`, and `github` plus `pdf`/`pptx`/`slide`
