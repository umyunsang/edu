def get_greeting(name: str) -> str:
    """Return a greeting for a user name, or Guest when the name is empty."""
    if not name:
        return "Hello, Guest!"
    return f"Hello, {name}!"
