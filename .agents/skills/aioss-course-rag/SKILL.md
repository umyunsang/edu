# AIOSS Course RAG

Use this skill when working in `ComputerScience/4-1_AIOSS` and the task requires retrieving or citing the course PDFs, class markdown notes, or sample package content.

## Workflow

1. Start in the AIOSS folder:
   `cd "/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu/ComputerScience/4-1_AIOSS"`
2. Refresh the local index when PDFs, markdown notes, or samples changed:
   `python3 tools/aioss_eval/build_rag_index.py --root .`
3. Query with concrete exam-oriented terms:
   `python3 tools/aioss_eval/rag_query.py "GitHub Actions CI testing"`
4. Cite results by local path, week, and chunk. If the source is a PDF with weak extraction, say that OCR/layout parsing should be used before relying on the passage.
5. For modern RAG or LLMOps implementation choices, combine the course result with current primary documentation or official project docs.

## Retrieval Policy

- Prefer class PDFs and `md/` notes for course concepts.
- Prefer current primary sources for external tools, APIs, and open source ecosystem details.
- Do not invent missing PDF content. If extraction is garbled, mark it as an extraction risk and fall back to visual/OCR review.

## Upgrade Path

- Baseline: local SQLite FTS5 lexical retrieval.
- Better layout: Docling parsing for PDFs and tables.
- Better relevance: hybrid dense plus sparse retrieval with reranking.
- Better evaluation: answer-level faithfulness and context relevance checks with RAGAS or TruLens.
