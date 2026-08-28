# Wave 2 - Main API And Browser Expansion

Owner: main orchestrator

## Browser proof

- In-app browser navigation to `https://academy.lgresearch.ai/study` rendered the academy app and redirected to `https://academy.lgresearch.ai/login`.
- The visible login page contains e-mail and password fields, confirming the current browser session is not authenticated.

## Static expansion

- Downloaded `asset-manifest.json` and 112 JS assets into `.omo/ultraresearch/20260625-182513/chunks/`.
- Relevant endpoint strings found in chunks include:
  - `/v1/portal/courses/my-courses`
  - `/v1/portal/courses/`
  - `/v1/portal/courses/main/`
  - `/v1/portal/contents/`
  - `/v1/portal/vod/info`
  - `/v1/common/files`
  - `/v1/common/files/`
  - `/v1/common/files/{fileId}/signed-url` pattern inferred from chunk code

## API proof

Unauthenticated probes returned:

- `/api/v1/auth-me`: 401 `RT_AUTHENTICATION_FAILURE`
- `/api/v1/portal/courses/my-courses`: 401 `RT_AUTHENTICATION_FAILURE`
- `/api/v1/portal/courses/certificates`: 401 `RT_AUTHENTICATION_FAILURE`
- `/api/v1/portal/contents/`: 401 `RT_AUTHENTICATION_FAILURE`
- `/api/v1/common/files/1/signed-url`: 401 `RT_AUTHENTICATION_FAILURE`
- `/api/v1/portal/codes`: 400 `RT_NOT_EXIST` for missing workspace ID

## EXPAND

- none — public/static and unauthenticated API routes are exhausted; the remaining lead requires a logged-in academy session.
