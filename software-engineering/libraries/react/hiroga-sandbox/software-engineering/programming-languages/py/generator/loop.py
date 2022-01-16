"""
無限配列のように扱える。
0
1
2
3
4
5
6
...
"""

import time


def endless():
    n = 0
    while True:
        yield n
        n += 1

for i in endless():
    time.sleep(1)
    print(i)