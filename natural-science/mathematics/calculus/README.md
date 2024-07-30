# 微分積分学 (calculus)

はじめに用語を定義しておく。表の通り、微分という言葉は微分係数を求める際も、導関数を求める際も用いられる。

| 演算     | 求める対象                           |
| -------- | ------------------------------------ |
| 微分     | 微分係数                             |
| 微分     | 導関数                               |
| 不定積分 | 原始関数                             |
| 定積分   | ある区間内の原始関数の増分（＝面積） |

また、線形性・積・商・合成関数の扱いについて、よく使うので表としてまとめておく。

| 演算     | 線形性（係数）                                  | 線形性（和・差）                                                                | 積                                                               | 商                                                                             | 合成関数の扱い                               |
| -------- | ----------------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------- |
| 微分     | $ (cf(x))' = cf'(x) $                           | $ (f(x) \pm g(x))' = f'(x) \pm g'(x) $                                          | $ (f(x)g(x))' = f'(x)g(x) + f(x)g'(x) $                          | $ \left( \frac{f(x)}{g(x)} \right)' = \frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2} $ | $ (f(g(x)))' = f'(g(x))g'(x) $               |
| 不定積分 | \( \int{cf(x)}\,dx = c\int{f(x)}\,dx \)         | \( \int{(f(x) \pm g(x))}\,dx = \int{f(x)}\,dx \pm \int{g(x)}\,dx \)             | 積分した結果を関数の積を含む式と見做すと、部分積分法が利用できる | -                                                                              | 置換積分の公式が対応しそう（よくわからない） |
| 定積分   | \( \int_a^b{cf(x)}\,dx = c\int_a^b{f(x)}\,dx \) | \( \int_a^b{(f(x) \pm g(x))}\,dx = \int_a^b{f(x)}\,dx \pm \int_a^b{g(x)}\,dx \) | -                                                                | -                                                                              | -                                            |

## 微分 (differential calculus)

### 微分の表記

導関数には次の書き方がある。

- $f'(x)$
  - ラグランジュの記法
  - プライム記号(′)を用いて表す。しかし簡単のため、シングルクォート(')を用いることが多いようだ。
  - 実際Desmosでプライム記号を入力すると、「'′'記号を理解できません。」と言われる
- $\frac{dy}{dx}x$
  - ライプニッツの記法
  - 合成関数の微分をそれぞれの関数の微分で置き換えるとき（連鎖律）に見やすくなる。

### 微分係数 (derivative)

微分係数または導関数のこと。たぶん日本人が「微分」と呼ぶのと同じ感覚でラフに使われる。正確には、微分係数は"Differential Cofficient"、導関数は"Derived Function"。

微分係数は定数、導関数は関数と性質の異なるものだが、個人的に次の2つの理由から混同してしまう。

- 式で表すと絵面が同じ
- y = x ^ 2 で簡単のために x = 1 を例に用いると、導関数のグラフと接線のグラフの傾きが一致してしまう。

```python
import matplotlib.pyplot as plt
import numpy as np

# 2次関数の場合
def quadrafic_case(func_ax: plt.Axes, derived_ax: plt.Axes):
    func_ax.set_xticks(np.arange(-4, 4+1, 1))
    func_ax.set_yticks(np.arange(-20, 16+1, 1))
    func_ax.grid()
    func_ax.autoscale(False)

    derived_ax.set_xticks(np.arange(-4, 4+1, 1))
    derived_ax.set_yticks(np.arange(-20, 16+1, 1))
    derived_ax.grid()
    derived_ax.autoscale(False)

    x = np.linspace(-4, 4, 100)

    # 元の関数
    function_x = lambda x: x * x
    f_x = function_x(x)
    func_ax.plot(x, f_x, color='blue')

    # y=f(x)の導関数、f`(x)
    derived_function_of_function_x = lambda a: a * 2
    differential_cofficients = derived_function_of_function_x(x)
    derived_ax.set_ylabel('Differential Cofficient', color='red')
    derived_ax.plot(x, differential_cofficients, color='green')

    # 接線
    tangent_at_x_equal_2 = x * derived_function_of_function_x(1) - 1
    func_ax.plot(x, tangent_at_x_equal_2, color='red')

    tangent_at_x_equal_3 = x * derived_function_of_function_x(2) - 4
    func_ax.plot(x, tangent_at_x_equal_3, color='red')

    tangent_at_x_equal_4 = x * derived_function_of_function_x(3) - 9
    func_ax.plot(x, tangent_at_x_equal_4, color='red')


# 3次関数の場合
def cubic_case(ax: plt.Axes):
    # 関数内で宣言した関数がローカルな関数にならないため、Lambda式で定義する。
    function_x = lambda x: x * x * x
    
    derived_function_of_function_x = lambda a: 3 * a ** 2

    differential_cofficient_of_function_x_at_x_equal_2 = lambda horizontal: horizontal * derived_function_of_function_x(2)

    x = np.linspace(-4, 4, 100)
    f_x = function_x(x)
    tangent_at_x_equal_2 = differential_cofficient_of_function_x_at_x_equal_2(x) - 16

    ax.set_xticks(np.arange(-4, 4+1, 1))
    ax.set_yticks(np.arange(-64, 64+1, 1))
    ax.grid()
    ax.autoscale(False)
    ax.plot(x, f_x, color='blue')
    ax.plot(x, tangent_at_x_equal_2, color='red')

fi, axes = plt.subplots(2, 2, figsize = (7, 14), tight_layout=True)
quadrafic_case(axes[0][0], axes[0][1])
cubic_case(axes[1][0])

plt.show()
```

### References

- [微分係数と導関数](https://rikeilabo.com/differential-coefficient)
- [Matplotlib 軸周り完璧マスターガイド - 軸・軸目盛・目盛り線の設定](https://www.yutaka-note.com/entry/matplotlib_axis)

### 微分の性質

- 線形性
  - $\frac{d}{dx}\{f(x)+g(x)\} = f'(x) + g'(x)$
  - $\frac{d}{dx}\{cf(x)\} = cf'(x)$
- Leibniz則（ライプニッツ則）: $\frac{d}{dx}\{f(x)g(x)\} = f'(x)g(x) + f(x)g'(x)$
- 合成関数の微分: $\frac{d}{dx}f(g(x)) = g'(x)f'(g(x))$
  - この書き方は$f'(g(x))$が$g(x)$に関する微分だということが初見で分かりづらいので好みではない

線形性という言葉について補足する。$f(x)$のような関数（更には写像）について、任意の$x, y, a$に対して、$f(ax) = af(x), f(x+y) = f(x) + (y)$が成り立つとき、線形関数（一次関数）のような挙動になるから線形性を持つ、という。

合成関数の微分について、次の通り証明と日本語の書き下しを添える。

$$
\begin{align}
\lim_{h \to 0} \frac{f(g(x+h)) - f(g(x))}{h}
= \lim_{h \to 0} \frac{f(g(x+h)) - f(g(x))}{g(x+h) - g(x)} \cdot \frac{g(x+h) - g(x)}{h}
= f'(g(x))g'(x)
\end{align}
$$

これは、「外部関数$f(g(x))$の内部変数$x$に対する変化率を知りたいのであれば、まず$x$に対する内部関数の変化率を求め、次にその変化率に対する外部関数を求めて、最後にかけ合わせなさい」と言っています。

### 初等関数の微分

個人的によく使うものを順次記載する。

- $f(x) = \log{f(x)} \implies f'(x) = \frac{f'(x)}{f(x)} (f(x) > 0)$
  - $f(x)$の対数を取ってるんだから、微分も$f'(x)$よりは小さくなるだろう、という感じで覚えている。
- $(f(x)*g(x))' = f'(x)*g(x)+f(x)*g'(x)$
  - 積の微分の公式
  - このあと、部分積分の公式を導くのに使う
- $\{\frac{f(x)}{g(x)}\}' = \frac{f'(x)*g(x) - f(x)*g'(x)}{\{g(x)\}^2}$
  - 商の微分
- $y' = \frac{dy}{dx} = \frac{dy}{dt} * \frac{dt}{dx}$
  - 合成関数の微分
  - 連鎖律ともいう
  - ライプニッツ記法で書いておくと分かりやすくて良い

### Taylor展開

ある関数がn回微分できるとき、その関数は地点$x_0$において1階微分, 2階微分, ..., n階微分の重み付きの和で近似できる。
<!-- TODO: 厳密性 -->
とはいえ、近似できるのは$x_0$の付近だけであり、ある閾値を超えると近似式が暴れ出す。その閾値をダランベールの収束半径と呼ぶ。[Desmos](https://www.desmos.com/calculator/gzneazvqb1)[🔐](https://www.desmos.com/calculator/thy0smdtik)を参照。

<!-- TODO: なぜダランベールの判定法でTaylor展開を求めたことになるのか？ -->

ちなみに、それぞれの項を足して近似するということは、それぞれの項はものすごく小さいのでは？という疑問が出てくる。グラフを書いて確かめるとその通りで、平べったい関数になっている。[Desmos](https://www.desmos.com/calculator/rkhguo3wkf)[🔐](https://www.desmos.com/calculator/0j5dqoy1pu)を参照。

また、$(0,t)$付近でTaylor展開することをマクローリン展開と呼ぶ。

### 微分法と関数のグラフ

グラフ上の2点を結んだ線分が常にグラフの上側にある関数を上に凸な関数という。しかし、漢字の凹凸の形と実際のグラフの形は正反対であり、誤解を招く。[^cho_2007]分かりにくいためか、マセマでは「上に凸」「下に凸」で統一されている。
[^cho_2007]: [凸している凹関数の困惑解消と実際応用](https://aulib.repo.nii.ac.jp/record/579/files/KJ00005242082.pdf)

$f(x)$が区間内で二階微分可能なとき、下に凸なら$f''(x) \ge 0$、上に凸なら$f''(x) \le 0$である。

#### Jensenの不等式

上に凸な関数について、次の不等式が成立する。

$$
\begin{align}
\sum_i \lambda_i f(x_i) \ge f(\sum_i \lambda_i x_i)
\end{align}
$$

また、下に凸な関数については、不等号を逆向きにした不等式が成立する。

[DesmosによるJensenの不等式のグラフ](https://www.desmos.com/calculator/ipbkvs3c3r)も参照。

### 級数と一様収束

## 偏微分 (partial derivative)

## リーマン積分 (riemann integral)

微分によって求めた導関数が変化率を求める式を、ある地点での導関数の値が傾きを表すように、関数が作る面積を求めることを定積分、その面積を求める式を求めることを不定積分と呼ぶ。

### 不定積分の表記

$\int f(x)dx=F(x)+C$

ここでなぜ$dx$が現れるのか疑問だったが、「高さが$f(x)$、底辺が$dx$（限りなく小さい$x$）である長方形を足し合わせ（$\int$）なさい」と読めばよい。

### 不定積分の基本公式・応用公式

個人的によく使うものを順次追加する。

- $\int \frac{f'(x)}{f(x)} = \log{f(x)} + C$
  - 元の関数が$f'(x)$より小さいのだから、積分した結果も$f(x)$寄りは小さいだろう、という感じで覚えている。
- $\int \frac{1}{\sqrt{x^2+a}}\,dx = \ln{|x+\sqrt{x^2+a}|}$
  - ナントカ分の1（-1乗、単位分数）になっているので積分結果はlogだろう、と察せる
- $\int \sqrt{x^2+a} \, dx = \frac{1}{2}(x\sqrt{x^2+a}+a\ln{|x+\sqrt{x^2+a}|})$
  - だいたい1乗の積分なので、だいたい2乗になっていることは分かる

#### 部分積分

微分の積の公式から導く。

#### 置換積分

内側の関数の引数から見た外側の関数の変化のようなガタガタした式を、複数の関数の変化として記述し直す、というようなイメージで捉えている。

### 定積分の表記

- $\int_a^b{f(x)}\,dx$
  - 積分記号、またはインテグラルと読む。
  - なぜSなのかは分からない。合計（Sum）だから？
- $[F(x)]_a^b$
  - インテグラル表記から原始関数の差の表記にするときに間に挟む
  - 上限と下限を使って評価を開始しますよ、ということを明示するためにやっているらしい
  - 確かに、引き算ではなく無限回の和（$f(x)*h$を無限に足す）で求めることも概念上できるが、有限時間で計算できないんだから他に選択肢ないわけで、この表記挟む理由は...?
- $\int_a^b{f(x)}\,dx = \lim_{n \to \infty} \sum_{i=1}^{n} f(x_i^*) \Delta xx$
  - リーマン和
  - 各部分での関数の和
  - 区間表記と対になる…と個人的に考えている。

### 定積分の公式

個人的によく使うものを順次追加する。

### Darboux（ダルブー）の定理による定式化

### 広義積分

### 多重積分

## 参考

- [微分積分キャンパス・ゼミ 改訂9](https://amzn.to/3ywWc37)
- [東京大学工学教室 - 基礎系数学 微積分](https://amzn.to/4bRVgVE)
