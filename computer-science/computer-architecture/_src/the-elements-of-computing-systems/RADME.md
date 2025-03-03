# [コンピュータシステムの理論と実装](https://amzn.to/4dJ5RDS)

<https://nand2tetris.github.io/web-ide>

## 1章 ブール論理

- nビットの論理ゲートは、1ビットの論理ゲートの内部を複製しただけなので、省略しました。
- 多入力マルチプレクサ、多入力デマルチプレクサも省略しました。

### 正準表現（Canonical Representation）

ブール関数を表す方法として、真理値表ではなくブール式（例: $f(A,B,C)=(A∧B)∨(¬A∧C)$）がある。

中でも、特定のルールに従ったブール式を正準表現と呼ぶ。（[コンピュータシステムの理論と実装](https://amzn.to/4dJ5RDS)では正準表現と訳されているが、ググってみた感じ訳語が定まっていないようだ）

ブール式が正準表現である場合、それは積和標準形（Sum of Products, SOP）か和積標準形（Product of Sums, POS）のいずれかの形式をとる。
