"""계산기 모듈"""


def add(a: int, b: int) -> int:
    """두 수를 더합니다.
    
    Args:
        a: 첫 번째 수
        b: 두 번째 수
    
    Returns:
        두 수의 합
    
    Examples:
        >>> add(2, 3)
        5
        >>> add(-1, 4)
        3
    """
    return a + b


def subtract(a: int, b: int) -> int:
    """두 수를 뺍니다."""
    return a - b
