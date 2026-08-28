# Wave 1: Callable Tool Surface

## Findings

`tool_search` exposed the `mcp__codex_apps__google_drive` namespace. The namespace description is "Search and work with files from Google Drive, Docs, Sheets, and Slides."

Representative available actions included:

- `_recent_documents`
- `_create_file`
- `_export_file`
- `_upload_file`
- `_import_document`
- `_import_spreadsheet`
- `_import_presentation`
- `_update_file`
- `_delete_file`
- `_get_document_text`
- `_batch_update_document`
- `_batch_update_spreadsheet`
- `_get_presentation`
- `_batch_update_presentation`
- comment read/write actions

## Runtime Verification

Read-only connector call:

```json
{"tool":"mcp__codex_apps__google_drive._recent_documents","args":{"top_k":1,"require_viewed_by_user":true}}
```

Observed result:

```json
{"results":[]}
```

The empty result is acceptable for this verification because the pass criterion was that the installed connector action is callable and returns a structured response without an installation or authorization error.

## EXPAND

none - callable Google Drive tools are already available.
