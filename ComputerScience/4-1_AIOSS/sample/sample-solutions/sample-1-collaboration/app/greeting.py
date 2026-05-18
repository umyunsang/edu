def get_greeting(name: str) -> str:
    """사용자 이름을 받아 인사말을 반환합니다."""
    if not name or len(name) == 0:
        return "Hello, Guest!"
    return f"Hello, {name}!"
