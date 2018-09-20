"""
結論から: 
forループは、基本的にはtupleを引数にもらうんだけど、勝手に展開してくれることもある。
しかし、groupbyはユースケース的に？もともと展開用のタプルを受け取るつもりで作ってない。だから引数１個のlambdaを引数にとる。
"""


from itertools import groupby

some_list = ['a', 'b', 'c', 'd', 'e']


# 検証: groupbyの引数にenumerateを渡したら、関数がとる値は一体何になっているのか？
for i, grouped_items in groupby(enumerate(some_list), lambda tuple: tuple[0]//2):  # 結果：tupleが返ってくる。
    print(i, [item for item in grouped_items])
"""
0 [(0, 'a'), (1, 'b')]
1 [(2, 'c'), (3, 'd')]
2 [(4, 'e')]
"""

# 参考：以下でもtupleになることが確認できる。
[i for i in enumerate(some_list)]  # [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')]


# 検証2: じゃあ、groupbyが引数を二つとる関数を引数にとったら？
"""
for i, items in groupby(enumerate(some_list), lambda x, y: print('x, y', x, y)):
    print('i, items', i, items)
"""
# 結果: エラーになる
# TypeError: <lambda>() missing 1 required positional argument: 'y'

# 検証3: じゃあ、groupbyの返り値が一つだけだったら？
for i in groupby(enumerate(some_list), lambda tuple: tuple[0]//2):
    print(i)

# 結果: tupleがもらえる
"""
(0, <itertools._grouper object at 0x10c4b1cc0>)
(1, <itertools._grouper object at 0x10c4b1cf8>)
(2, <itertools._grouper object at 0x10c4b1cc0>)
"""
