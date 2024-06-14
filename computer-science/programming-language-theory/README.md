# プログラミング言語論 (programming language theory)

## 字句解析

## 構文解析

## 抽象構文

## 意味解析

## GC (garbage collect)

## オブジェクト指向

## 関数型

## ループ

配列などの要素を順番に処理するにあたって、インデックスをインクリメントするのは冗長だが、配列をコピーしてpop()させるのはメモリの使用量が多い。そこで、現在が何周目かの状態を持ったオブジェクトを生成し、配列の要素の取り出しを任せよう、という発想で使うのがイテレータである。Pythonにおいては、マジックメソッドである`__iter__`を実装したクラスをイテラブルという。

また、配列の要素の取り出しが自分でカスタムできるなら、そもそも本当に配列を参照する必要がない。例として、次の2つのコードは同じ挙動を示す。

```python
class OneTwoThree:
    def __init__(self):
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index <= 3:
            return self.index
        else:
            raise StopIteration


n_list = [1, 2, 3]
for n in n_list:
    print(n)

one_two_three = OneTwoThree()
for n in one_two_three:
    print(n)
```

配列から値を取り出すような、イテレーション毎の処理がインデックス以外は同じであるケースでは、イテレータが適している。一方で、「初回のループのみ初期化処理を行う」「内部状態がインデックスからは予測不能である」ようなケースでは、直感的な処理を状態と分岐処理に分けて書く必要があり、やや儀礼的になる。そこで、どこまで実行したかを記録し、次回呼び出す際には続きから実行するためのキーワードがあり、それが`yield`である。

```python
def one_two_three():
    yield 1
    yield 2
    yield 3
```

## 同期処理

<!-- セマフォ -->

## 非同期処理

## 並行・並列処理

<!-- コルーチン, ゴルーチンとの違いも -->

## References

[型システム入門 −プログラミング言語と型の理論](https://amzn.to/4elvECy)
