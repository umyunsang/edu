# Wave 1 - Web Static Assets

Worker: librarian `019efe1a-d77b-7061-ab9c-92363cd83e9f`

## Findings

- `https://academy.lgresearch.ai/study` returns a Create React App HTML shell.
- The public shell references `/static/js/main.400d8c09.js` and `/static/css/main.5ab3849a.css`.
- `https://academy.lgresearch.ai/asset-manifest.json` is publicly exposed and lists hashed JS chunks.
- `robots.txt`, `sitemap.xml`, `manifest.json`, service-worker guesses, and source-map guesses fall back to the SPA HTML shell.
- Static JS exposes route strings `/study`, `/study/:courseId`, and `/studyroom/:courseId`, but no unauthenticated lecture-material index.

## EXPAND

- LEAD: route-to-chunk mapping by dynamic import ID — WHY: `asset-manifest.json` lists many hashed chunks but route ownership was not fully mapped — ANGLE: fetch representative chunk files and grep for route/component names
- LEAD: public lecture-material endpoints beyond the SPA shell — WHY: no unauthenticated lecture index was found yet, but `/v1/common/files/images/` suggests adjacent file APIs may exist — ANGLE: inspect bundle for other `/v1/common/files/*` or `/download/*` patterns
