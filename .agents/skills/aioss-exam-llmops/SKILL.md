# AIOSS Exam LLMOps

Use this skill when preparing, implementing, or evaluating AIOSS practical exam exercises that involve GitHub Flow, CI, testing, RAG, LLMOps, or open source development workflow.

## Operating Loop

1. Read `sample/SAMPLE_PROBLEMS.md`, `sample/SUCCESS_SAMPLE.md`, and local `AGENTS.md`.
2. Run a baseline check before editing:
   `python3 tools/aioss_eval/sample_eval.py --target sample/sample-solutions-minimal --label minimal-baseline`
3. Implement the smallest correct change in the minimal sample.
4. Verify with:
   - `python3 tools/aioss_eval/sample_eval.py --target sample/sample-solutions-minimal --label minimal`
   - `ruff check sample/sample-solutions sample/sample-solutions-minimal tools/aioss_eval`
   - `actionlint sample/sample-solutions-minimal/sample-2-ci-basics/.github/workflows/ci.yml`
5. Compare against `sample/sample-solutions`, but still run gates against the solution because sample code can contain lint or evidence defects.
6. Record the learning in Korean study notes when the user asks for a report.

## Evaluation Lenses

- Functional correctness: code satisfies explicit tests and example cases.
- CI reproducibility: workflow uses checkout, runtime setup, dependency installation, and a real verification command.
- Shift-left testing: tests exist before or alongside implementation and capture the expected behavior.
- Open source readiness: PR template, rollback plan, and checklist are complete.
- Evidence quality: command outputs, failure modes, and fixes are traceable.

## Modern LLMOps Defaults

- Keep local, deterministic gates separate from cloud LLM calls.
- Use retrieval for source grounding before generating fixes or reports.
- Add semantic/vector retrieval only after lexical retrieval is working and measurable.
- Evaluate RAG with context relevance, groundedness, and answer relevance before optimizing model prompts.
