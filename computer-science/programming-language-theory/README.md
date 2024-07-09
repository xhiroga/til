# プログラミング言語論 (programming language theory)

ページ構成にあたって次の情報源を参考にした。

- [コンピュータシステムの理論と実装](https://amzn.to/3RRkRGI)
- [最新コンパイラ構成技法](https://amzn.to/3xJRDCI)
- [型システム入門 −プログラミング言語と型の理論](https://amzn.to/4elvECy)
- [Programming Language Design and Implementation](https://www.springerprofessional.de/en/programming-language-design-and-implementation/23739088)
- [Concept of Programming Languages](https://www.sci.brooklyn.cuny.edu/~chuang/books/sebesta.pdf)
- [Foundations of Programming Languages](https://link.springer.com/book/10.1007/978-3-319-70790-7)
- [Cluade🔐](https://claude.ai/chat/bd1635ce-06a1-4ac4-8174-66569a915c73)

## プログラミング言語の歴史

## コンパイラ

### 字句解析

### 構文解析

<!-- - Earley法
- LL法
- LR(Left-to-right, Rightmost derivation)文法 -->

#### 式の評価

数式やプログラムを記述する方法として、次の3つが挙げられる。

- 前置記法 (ポーランド記法, PN, Polish notation)
- 中置記法 (infix notation, IN)
- 後置記法 (逆ポーランド記法, RPN, reverse Polish notation)

### 抽象構文

抽象構文木 (AST, abstract syntax tree) は、プログラムの構文構造を表す木構造のデータ構造である。構文解析の結果として生成され、意味解析やコード最適化などのコンパイラの後続のフェーズで使用される。

### 意味解析

### 中間表現

### 最適化

### コード生成

## 設計と実装

### 構文 (syntax)

### メモリ管理 (memory management)

GC (garbage collect) は、プログラムが動的に割り当てたメモリのうち、もはや使用されていないメモリを自動的に解放するメモリ管理手法である。プログラマがメモリの割り当てと解放を手動で管理する必要がなくなり、メモリリークなどのバグを防ぐことができる。

### スコープ, 関数, パラメータ渡し (scopes, functions and parameter passing)

<!-- > [!NOTE] ラムダ式と無名関数と匿名関数って同じ意味で使っていいの？ -->

### 制御構造 (control structures)

#### ループ

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

配列から値を取り出すような、イテレーション毎の処理がインデックス以外は同じであるケースでは、イテレータが適している。一方で、「初回のループのみ初期化処理を行う」「内部状態がインデックスからは予測不能である」ようなケースでは、直感的な処理を状態と分岐処理に分けて書く必要があり、やや儀礼的になる。そこで、どこまで実行したかを記録し、次回呼び出す際には続きから実行できるジェネレータがあり、Pythonでは`yield`演算子が相当する。

```python
def one_two_three():
    yield 1
    yield 2
    yield 3
```

### 型 (types)

### モジュール化 (modularization)

## パラダイム

### オブジェクト指向プログラミング (OOP, object oriented programming)

関数の返り値で`this`を返すことをfluent interfaceと呼び、メソッドチェーンによる簡潔な記述が可能になる。

### 関数型プログラミング (functional programming)

### 論理プログラミング (logical programming)

### 並行 / 並列 / 非同期プログラミング (concurrent / parallel / asynchronous programming)

<!-- セマフォ -->

<!-- コルーチン, ゴルーチンとの違いも -->

### ドメイン固有言語 (DSL, domain specific language)

### Foreign function interface (FFI)

FFIとは異なるプログラミング言語によって書かれたモジュールを呼び出すための仕組みを言う。例えばPythonには`ctypes`ライブラリが存在し、C/C++で書かれた共有オブジェクトライブラリを読み込んでメソッド等の実行を可能にする。元々C/C++向けに書かれたコードを`ctypes`で呼び出す場合は、データ型の変換等は`ctypes`が担う。しかし、C/C++でPython向けのライブラリを実装する場合は、C/C++からPythonのメモリにアクセスする必要がある。そのような場合には、`<Python.h>`を用いるとよく、これもFFIと言える。

## 未分類 (uncategorized)

- リアルタイム性: WIP
- JITコンパイラ: WIP
- クロージャ（閉包）: WIP
- 第一級オブジェクト: WIP
- 末尾関数: WIP
- 参照透明性: WIP
- 計測における不確かさ: WIP
