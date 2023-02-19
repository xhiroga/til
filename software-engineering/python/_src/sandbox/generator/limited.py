"""
ジェネレータは配列のように扱うことができる。
"""

def limited():
    yield 1
    yield 2
    yield 3

for i in limited():
    print(i)