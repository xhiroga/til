"""
mypy wrong.py
"""

def two_time(n: int) -> int:
    return n * 2

if __name__ == '__main__':
    m = two_time('hi')
    print(m)