# 샘플 3: 테스트 기초 (TDD 사이클)

## 📌 문제 설명
TDD (Test-Driven Development) 사이클을 경험합니다:  
**Red → Green → Refactor** 사이클을 1회 수행합니다.

## 🎯 요구사항 (3단계)

### Step 1: 실패하는 테스트 먼저 작성 (RED)
파일: `tests/test_calculator.py`

```python
def test_add_positive_numbers():
    from app.calculator import add
    result = add(2, 3)
    assert result == 5, "2 + 3은 5여야 합니다"


def test_add_with_negative():
    from app.calculator import add
    result = add(-1, 4)
    assert result == 3, "-1 + 4는 3이어야 합니다"
```

실행:
```bash
pytest tests/test_calculator.py
```

예상 결과:
```
❌ FAILED - app.calculator 모듈을 찾을 수 없음
```

### Step 2: 테스트를 통과시키는 최소 구현 (GREEN)
파일: `app/calculator.py`

```python
def add(a: int, b: int) -> int:
    """두 수를 더합니다."""
    return a + b
```

다시 실행:
```bash
pytest tests/test_calculator.py
```

예상 결과:
```
✅ PASSED - 2 passed in 0.05s
```

### Step 3: 테스트 추가 + 정리 (REFACTOR)
새 테스트 추가: `tests/test_calculator.py`

```python
def test_add_zero():
    from app.calculator import add
    result = add(0, 5)
    assert result == 5, "0 + 5는 5여야 합니다"
```

코드 정리 (불필요한 임포트 제거 등):

```python
# app/calculator.py
"""계산기 모듈"""

def add(a: int, b: int) -> int:
    """두 수를 더합니다.
    
    Args:
        a: 첫 번째 수
        b: 두 번째 수
    
    Returns:
        두 수의 합
    """
    return a + b
```

최종 실행:
```bash
pytest tests/test_calculator.py
```

예상 결과:
```
✅ PASSED - 3 passed in 0.06s
```

## ✅ 성공 기준
- [ ] 테스트 파일 생성 (2개 이상 테스트)
- [ ] pytest 실패 → 성공 흐름 확인
- [ ] 최소 2개 커밋으로 기록 (test 작성, 구현)

## 커밋 메시지 예시
```
commit 1: test: add calculator tests (RED)
commit 2: feat: implement add function (GREEN)
commit 3: docs: improve docstring (REFACTOR)
```

## 💾 풀이 보기
[-> app/calculator.py](./app/calculator.py)  
[-> tests/test_calculator.py](./tests/test_calculator.py)

