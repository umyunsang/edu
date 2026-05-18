"""
샘플 3: TDD 사이클 완전 가이드

## Phase 1: RED (테스트 실패)
---
$ pytest tests/test_calculator.py

FAILED tests/test_calculator.py::TestAdd::test_add_positive_numbers
ModuleNotFoundError: No module named 'app.calculator'

→ 테스트는 작성했지만 구현이 없어 실패 ❌

## Phase 2: GREEN (테스트 통과)
---
app/calculator.py에 add() 함수 구현

$ pytest tests/test_calculator.py

passed in 0.05s ✅

→ 모든 테스트 통과 ✅

## Phase 3: REFACTOR (개선)
---
- 함수에 docstring 추가
- 테스트 추가 (test_add_zero)
- 코드 정리

$ pytest tests/test_calculator.py

passed in 0.06s ✅

→ 품질 향상 + 테스트 유지 ✅

## 핵심 포인트
1. 테스트를 먼저 작성 (개발 가이드)
2. 최소한의 구현으로 통과
3. 리팩토링하면서 기능 보완

이 순서가 TDD의 핵심입니다!
"""

# 실행 예시
if __name__ == "__main__":
    import subprocess
    
    # pytest 실행
    result = subprocess.run(["pytest", "tests/test_calculator.py", "-v"], 
                          capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
