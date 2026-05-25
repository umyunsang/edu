# AIOSS Sample Evaluation: minimal-baseline

- Target: `/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu/ComputerScience/4-1_AIOSS/sample/sample-solutions-minimal`
- Score: 2/8

## Checks
- FAIL `sample1-functional-greeting`: NotImplementedError('TODO: 이름이 비어 있으면 Guest, 아니면 이름을 포함한 인사말을 반환하세요.')
- FAIL `sample1-pr-template-readiness`: Template still needs completed checklist, rollback plan, or TODO removal.
- FAIL `sample2-workflow-content`: Missing: setup python action
- PASS `sample2-actionlint`: actionlint passed.
- FAIL `sample3-test-intent`: Expected add and subtract assertions were not both found.
- FAIL `sample3-pytest`: FF                                                                       [100%]
  =================================== FAILURES ===================================
  __________________________ test_add_positive_numbers ___________________________
  
      def test_add_positive_numbers():
          # TODO: add(2, 3) == 5 검증 테스트를 작성하세요.
  >       raise NotImplementedError("TODO: 양수 덧셈 테스트를 작성하세요.")
  E       NotImplementedError: TODO: 양수 덧셈 테스트를 작성하세요.
  
  tests/test_calculator.py:6: NotImplementedError
  ________________________ test_subtract_positive_numbers ________________________
  
      def test_subtract_positive_numbers():
          # TODO: subtract(10, 3) == 7 검증 테스트를 작성하세요.
  >       raise NotImplementedError("TODO: 양수 뺄셈 테스트를 작성하세요.")
  E       NotImplementedError: TODO: 양수 뺄셈 테스트를 작성하세요.
  
  tests/test_calculator.py:11: NotImplementedError
  =========================== short test summary info ============================
  FAILED tests/test_calculator.py::test_add_positive_numbers - NotImplementedEr...
  FAILED tests/test_calculator.py::test_subtract_positive_numbers - NotImplemen...
  2 failed in 0.01s
- PASS `sample3-ruff`: All checks passed!
- FAIL `todo-marker-scan`: sample-1-collaboration/PR_TEMPLATE.md:4: - TODO: 어떤 파일을 수정했는지 작성
  sample-1-collaboration/PR_TEMPLATE.md:5: - TODO: 어떤 기능을 추가했는지 작성
  sample-1-collaboration/PR_TEMPLATE.md:8: - TODO: 로컬에서 어떤 방식으로 확인했는지 작성
  sample-1-collaboration/PR_TEMPLATE.md:11: - TODO: 문제 발생 시 되돌리는 방법 작성
  sample-1-collaboration/PR_TEMPLATE.md:14: - [ ] TODO: 코드 동작 확인
  sample-1-collaboration/PR_TEMPLATE.md:15: - [ ] TODO: 자체 리뷰 완료
  sample-1-collaboration/PR_TEMPLATE.md:16: - [ ] TODO: 문서 또는 설명 보강
  sample-1-collaboration/app/greeting.py:3: raise NotImplementedError("TODO: 이름이 비어 있으면 Guest, 아니면 이름을 포함한 인사말을 반환하세요.")
  sample-2-ci-basics/.github/workflows/ci.yml:15: - name: TODO checkout step
  sample-2-ci-basics/.github/workflows/ci.yml:16: run: 'echo "TODO: actions/checkout 단계를 추가하세요."'
  sample-2-ci-basics/.github/workflows/ci.yml:18: - name: TODO setup python
  sample-2-ci-basics/.github/workflows/ci.yml:19: run: 'echo "TODO: Python 3.10 설정 단계를 추가하세요."'
  sample-2-ci-basics/.github/workflows/ci.yml:21: - name: TODO install dependencies
  sample-2-ci-basics/.github/workflows/ci.yml:22: run: 'echo "TODO: ruff 설치 단계를 추가하세요."'
  sample-2-ci-basics/.github/workflows/ci.yml:24: - name: TODO run lint
  sample-2-ci-basics/.github/workflows/ci.yml:25: run: 'echo "TODO: ruff check 실행 단계를 추가하세요."'
  sample-3-testing/TDD_CYCLE.md:4: - TODO: 먼저 실패하는 테스트를 작성합니다.
  sample-3-testing/TDD_CYCLE.md:7: - TODO: 테스트를 통과시키는 최소 구현을 작성합니다.
  sample-3-testing/TDD_CYCLE.md:10: - TODO: 테스트가 유지되는 범위에서 코드를 정리합니다.
  sample-3-testing/app/calculator.py:6: raise NotImplementedError("TODO: add 함수를 구현하세요.")
