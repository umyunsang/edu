"""계산기 테스트"""

from app.calculator import add, subtract


class TestAdd:
    """add 함수 테스트"""
    
    def test_add_positive_numbers(self):
        """양수 덧셈"""
        result = add(2, 3)
        assert result == 5
    
    def test_add_with_negative(self):
        """음수 포함 덧셈"""
        result = add(-1, 4)
        assert result == 3
    
    def test_add_zero(self):
        """0 포함 덧셈"""
        result = add(0, 5)
        assert result == 5


class TestSubtract:
    """subtract 함수 테스트"""
    
    def test_subtract_positive(self):
        """양수 뺄셈"""
        result = subtract(10, 3)
        assert result == 7
    
    def test_subtract_negative_result(self):
        """음수 결과 뺄셈"""
        result = subtract(3, 10)
        assert result == -7
