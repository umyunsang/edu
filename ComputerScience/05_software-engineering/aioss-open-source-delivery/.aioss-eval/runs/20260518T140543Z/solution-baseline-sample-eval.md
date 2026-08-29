up:: [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/notes/01. 알고리즘 설계 — 효율성·분할정복·동적계획법·탐욕·NP.md|기말고사_정리]]
related:: [[ComputerScience/05_software-engineering/aioss-open-source-delivery/.aioss-eval/runs/20260518T142831Z/pre-push-minimal-sample-eval|pre-push-minimal-sample-eval]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/.aioss-eval/runs/20260518T141147Z/minimal-verify-sample-eval|minimal-verify-sample-eval]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/.aioss-eval/runs/20260518T140702Z/minimal-final-sample-eval|minimal-final-sample-eval]]

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
