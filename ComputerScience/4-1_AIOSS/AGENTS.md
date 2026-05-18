# AIOSS Codex Working Rules

This folder is the working area for AI OSS practical exam preparation. Follow the root Obsidian vault rules, plus the AIOSS-specific rules below.

## Purpose

- Prepare for practical exams that evaluate GitHub Flow, CI, testing, LLMOps, and open source development workflows.
- Keep every change evidence-driven: baseline result, implementation, verification command, and short learning note.
- Treat the course PDFs and markdown notes as the primary class source. Use current official or primary-source documentation for modern tooling details.

## Source and Report Language

- Source code, workflow files, script comments, and developer-facing templates should be written in English unless the data itself is Korean course content.
- Study notes and user-facing reports may be written in Korean.

## RAG and Course Reference Rules

- Cite course material by file name, week, and page or section whenever possible.
- Prefer local course PDFs and `md/` notes for class concepts.
- For fast local lookup, use the AIOSS RAG tools:
  - Build or refresh: `python3 tools/aioss_eval/build_rag_index.py --root .`
  - Query: `python3 tools/aioss_eval/rag_query.py "GitHub Actions CI testing"`
- If PDF text extraction is weak, mark it as an OCR/layout issue instead of inventing content.

## Sample Practice Loop

For every sample exercise:

1. Read `sample/SAMPLE_PROBLEMS.md` and `sample/SUCCESS_SAMPLE.md`.
2. Run a baseline check and save the failure mode.
3. Complete the minimal TODO version.
4. Run deterministic checks:
   - `python3 tools/aioss_eval/sample_eval.py --target sample/sample-solutions-minimal --label minimal`
   - `ruff check sample/sample-solutions sample/sample-solutions-minimal tools/aioss_eval`
   - `actionlint sample/sample-solutions-minimal/sample-2-ci-basics/.github/workflows/ci.yml`
5. Record RED, GREEN, and REFACTOR evidence in the sample notes or generated evaluation report.

## Evaluation Standard

Use multiple evaluation lenses:

- Functional correctness: functions and tests pass.
- CI syntax and reproducibility: workflow parses with `actionlint` and uses explicit setup steps.
- Shift-left quality: failing tests are written before or alongside the implementation.
- Open source readiness: PR template, rollback plan, checklist, and evidence are complete.
- RAG readiness: local retrieval produces cited snippets from course assets.

Do not treat the provided complete solution as automatically correct. Run the same gates against it and fix concrete defects when they are in scope.
