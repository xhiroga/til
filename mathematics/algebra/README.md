# 代数学

## Linear Algebra

線形代数学。

## Vector

ベクトル。19世紀に様々な数学者が独自の定義を発表する中で、ポーランドの高校の数学教師だったグラスマンの理論が現在に続いている。[^vector]

[^vector]: [地球惑星数理演習 ベクトルとテンソル - 吉田茂生](https://www.zotero.org/groups/4682218/hiroga-scholar/collections/MA5LXYUI/items/2D27XYT8/item-details)

### Norm （ノルム）

ベクトルの長さ。ピタゴラスの定理を用いて、各次元の2乗の和のルート2で求める。

![ノルム](/images/ノルム.svg)

### Dot Product （内積）

0でない2つのベクトルを$\vec{a}, \vec{b}$とする。1点$O$を定め、$\vec{a}=\vec{OA}, \vec{b}=\vec{OB}$となる点$A,B$を取る。このとき、半直線$OA, OB$のなす角Θのうち、0<=Θ<=180であるものをベクトル$\vec{a}, \vec{b}$のなす角という。このとき、積$|\vec{a}||\vec{b}|cosΘ$をベクトルの内積といい、記号$\vec{a}\cdot \vec{b}$で表す。

![内積](/images/内積.svg)

物理学的には、力と移動距離から仕事量を求める際に用いる。図（内積と仕事）では、50mの間200Nが水平方向に変換されただけの力を発揮した、ということになる。内積は交換法則が成り立つが、図の例で言えば200Mを50N（が水平方向に変換されただけ）の力も同じ仕事量、ということになる。

![内積と仕事](/images/内積と仕事.svg)

数学的には、２つのベクトルから数を生成する関数はどんなものか、という問いに対する答えから生まれたものらしい。内積には交換法則が成り立つが、同時代にハミルトンが定義した四元数の積では成り立たず、それも複雑で広まらなかったとされる要因かもしれない。

内積（Inner Product）という用語について、グラスマンは次のように述べている。[^grassmann]

[^grassmann]: [数学に関する質問とその背景の数学 - 竹野茂治](https://www.zotero.org/groups/4682218/hiroga-scholar/collections/MA5LXYUI/items/QPBQHP69/item-details)

> 内積の値は −→ a と −→b が垂直ならば 0 で、その値が正になるためには、−→b が少し −→ a の『内側』の方に入らないといけないから、これを『内積』と呼ぶ。

- Dot Productの他に、Scalar Product, Inner Productともいう。
- 高校数学では$\vec{a}$と矢印で表すことが多いが、大学数学では**a**のように太字で表すことが多い。これは、大学数学より先のベクトルは多次元量であり、平面や立体のように矢印で考えづらいからと言われている（[参考](https://detail.chiebukuro.yahoo.co.jp/qa/question_detail/q1332986798)）

### コサイン類似度

ベクトルの内積をノルムの積で割ったもの。つまり、2つのベクトルのなす角Θに対するcosの値である。

[Desmos](https://www.desmos.com/calculator/pm7m6hdypq)([🔐](https://www.desmos.com/calculator/x7bw9a2yue))も参照。

## Matrix（行列）

### 行列の積

![行列の積](/images/行列の積.svg)

### その他の参考文献

- 🔐[数学B 改訂版 - 数研出版](https://www.zotero.org/groups/4682340/hiroga-books/items/8ZRH3IKI/item-details)
- [線形代数キャンパス・ゼミ 改訂11](https://amzn.to/3V7QlJe)
- [仕事と運動エネルギーとの関係 - 理数系学習サイト kori](https://physkorimath.xyz/w/)
