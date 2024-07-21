# プログラミング言語論 (programming language theory)

ページ構成にあたって次の情報源を参考にした。

- [コンピュータシステムの理論と実装](https://amzn.to/3RRkRGI)
- [最新コンパイラ構成技法](https://amzn.to/3xJRDCI)
- [型システム入門 −プログラミング言語と型の理論](https://amzn.to/4elvECy)
- [Programming Language Design and Implementation](https://www.springerprofessional.de/en/programming-language-design-and-implementation/23739088)
- [Concept of Programming Languages](https://www.sci.brooklyn.cuny.edu/~chuang/books/sebesta.pdf)
- [Foundations of Programming Languages](https://link.springer.com/book/10.1007/978-3-319-70790-7)
- [Claude🔐](https://claude.ai/chat/bd1635ce-06a1-4ac4-8174-66569a915c73)

## プログラミング言語の歴史

## コンパイラ

広義のコンパイラは、ソース言語で書かれたプログラムをターゲット言語のプログラムに変換するプログラムである。例えばC言語のプログラムは、GCCというコンパイラによって機械語に変換される。機械語はISAによって異なるので、コンパイラはオプションなどで対象のアーキテクチャを指定することができる。GCCの例は次の通り。

```shell
gcc -march=rv64gc example.c -o example
```

TypeScriptをJavaScriptに変換するような、ターゲット言語が機械語ではないコンパイルもある。これは狭義のコンパイルと区別するために、トランスパイルとも呼ばれる。

### 字句解析 (lexical analysis)

コンパイラの構文解析は、トークナイザ(tokenizer)とパーサー(parser)の2つのモジュールに分けられる。

### 構文解析, 抽象構文

プログラムをトークンに字句解析した次は、抽象構文木へとパースを行う。抽象構文木 (AST, abstract syntax tree) は、プログラムの構文構造を表す木構造のデータ構造である。構文解析の結果として生成され、意味解析やコード最適化などのコンパイラの後続のフェーズで使用される。

プログラミング言語の文法は、文脈自由文法と呼ばれる生成規則によって規定される。コンパイラの実装に依存しない規則で文法を定義できることは、プログラミング言語の動作から環境差異を除くことに繋がる。文脈自由文法は、状態とスタックから入力が規則に則っている（受理）かを判定する。状態だけでなくスタックを持つ文脈自由文法を適用することで、括弧の数を記憶することができる。

構文木を構成するための様々なアルゴリズムがある。再帰下降構文解析(recursive descent parsing)では、パーサを再帰的なメソッドで構成する。例えば`parseWhileStatement()`は`while`トークンを解析し、残りのトークンを次のメソッドに渡す、という流れでパースできる。

文脈自由文法はプログラミング言語の文法をシンプルに記述できるが、効率的な解析のためにはまだ柔軟に感じられる。それは、文脈自由文法を非決定性PDAで定義できることからも分かるように、トークンを認識する状態遷移が一意に定まらないことである。例えば次のような文を考えたい。四則演算には計算の優先順位があるため、`b`を読んだ段階で`b`を`+`の子ノードにしてよいか判断することができない。言い換えれば、パーサのどのメソッドを適用してよいか判断できない。

```python
expression = a + b * c ** d
```

そこで、トークンを1つ読んだ段階でパーサのメソッドが予測可能になるような制約を加えた文法をLL(1)文法と呼ぶ。実際のプログラミング言語の設計では、LL(1)文法を基本としつつ、部分的にLL(1)文法を逸脱した文脈自由文法に沿うことが多い。

<!-- - Earley法
- LR(Left-to-right, Rightmost derivation)文法 -->

### 意味解析

### 中間表現 (IR, intermediate representation)

機械語はIAによって異なる。ここで、ターミナルから実行するアプリケーションの開発者を考える（Webアプリはブラウザ上で動くため、すでにプラットフォーム非依存であり例えとして分かりづらい）。開発者がアプリケーションを公開するにあたって、IntelやAMDな複数のプラットフォームごとにコンパイルするのは大変である。そこで、もしプラットフォーム非依存の抽象化された機械語があれば、利用者がそれを機械語に通訳しながら利用することで、問題を解決できる。少し違うが、AWSのAPIを呼ぶプログラムの代わりにTerraformを使うようなものである。このようなプラットフォーム非依存の表現を中間表現といい、中間表現を機械語に通訳するプログラムを仮想マシン(VM)という。VMは、メモリ管理やスレッド管理などのプラットフォーム依存の処理を引き受けることもある。コンパイラにおいて、中間表現にコンパイルするまでをフロントエンド、中間表現から機械語をバックエンドと呼ぶ。

VMには、次のような種類がある。なお、厳密にはLLVMはVMではなくコンパイラのバックエンドだが、IRからコード生成を行うプログラムとしてまとめて表にした。

| VM/Compiler        | IR                    | Source Language                      | Note               |
| ------------------ | --------------------- | ------------------------------------ | ------------------ |
| JVM                | Java Bytecode         | Java, Kotlin, Scala, Groovy, Clojure | - 強力な最適化とGC |
| .NET CLR           | CIL                   | C#, F#, Visual Basic .NET            | - Microsoftが開発  |
| GraalVM            | Truffle AST, Graal IR | Java, JavaScript                     | - Oracleが開発     |
| LLVM               | LLVM IR               | C, C++, Rust, Swift, Objective-C     | - コンパイラ基盤   |
| V8, Wasmer, etc... | WebAssembly bytecode  | C, C++, Rust, AssemblyScript         | - ブラウザ内で動作 |

### 最適化

### コード生成

構文木から機械語やVMコードへの変換を考える。機械語やVMコードは、基本的に動詞と目的語からなる。重要なのはその語順である。数式やプログラムを記述する方法として、次の3つが挙げられる。

- 前置記法 (ポーランド記法, PN, Polish notation)
- 中置記法 (infix notation, IN)
- 後置記法 (逆ポーランド記法, RPN, reverse Polish notation)

機械語やVMコードでは後置記法を基に拡張した形式を用いられることが多い。

### コンパイル戦略

プログラムの実行にあたって、ソース言語を予め機械語に翻訳し、実行可能なバイナリファイルを生成することをAOTコンパイルという。一方で、プログラムの実行時にコンパイルする方式をJITコンパイルという。JITコンパイルの例としては、Rubyにおいて高速化を目的として有効化されるYJITや、JavaScriptの高速化のためにブラウザに搭載されたJITコンパイラ等がある。また、機械語へのコンパイラを用いずに、プログラムからVMコードを生成し、実行時にはVMがVMコードに対応する機械語に通訳して実行する戦略もあり、インタプリタ型と呼ばれる。いくつかの言語について、中間コード・機械語のそれぞれの観点から、コンパイル戦略をまとめた。

![プログラミング言語のコンパイル戦略](https://i.gyazo.com/thumb/3079/e6d65ee7f9d55990574dd1434fbbb466-heic.jpg)

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
- クロージャ（閉包）: WIP
- 第一級オブジェクト: WIP
- 末尾関数: WIP
- 参照透明性: WIP
- 計測における不確かさ: WIP
