"""
groupbyの返り値ってなんだっけ？
結果: 

"""

from itertools import groupby

some_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i, grouped_items in groupby(some_list, lambda num: num // 2):
    print(i, grouped_items)
