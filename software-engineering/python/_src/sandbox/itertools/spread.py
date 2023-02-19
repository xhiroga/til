"""
疑問：スプレッド構文みたいに、引数を展開して受け取れないか？
結果：jsみたいに関数側で展開するのではなく、呼び出し側で（＝実行時に）unpackする。
"""

some_list = ['a', 'b', 'c', 'd', 'e']


def spread(i, item):
    print(i)


for tuple in enumerate(some_list):
    spread(*tuple)
