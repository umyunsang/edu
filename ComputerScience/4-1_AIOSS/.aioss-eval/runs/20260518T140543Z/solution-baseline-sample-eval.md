# AIOSS Sample Evaluation: solution-baseline

- Target: `/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu/ComputerScience/4-1_AIOSS/sample/sample-solutions`
- Score: 5/8

## Checks
- PASS `sample1-functional-greeting`: Greeting cases passed.
- FAIL `sample1-pr-template-readiness`: Template still needs completed checklist, rollback plan, or TODO removal.
- PASS `sample2-workflow-content`: Workflow contains expected CI steps.
- PASS `sample2-actionlint`: actionlint passed.
- FAIL `sample3-test-intent`: Expected add and subtract assertions were not both found.
- PASS `sample3-pytest`: .....                                                                    [100%]
  5 passed in 0.00s
- FAIL `sample3-ruff`: F401 [*] `pytest` imported but unused
   --> tests/test_calculator.py:2:8
    |
  1 | """계산기 테스트"""
  2 | import pytest
    |        ^^^^^^
  3 |
  4 | from app.calculator import add, subtract
    |
  help: Remove unused import: `pytest`
  
  Found 1 error.
  [*] 1 fixable with the `--fix` option.
- PASS `todo-marker-scan`: No TODO or NotImplementedError markers in practice files.
