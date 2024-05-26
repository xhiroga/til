# Calculus（微分積分学）

## Differential calculus（微分）



### Derivative

微分係数または導関数のこと。たぶん日本人が「微分」と呼ぶのと同じ感覚でラフに使われる。正確には、微分係数は"Differential Cofficient"、導関数は"Derived Function"。

微分係数は定数、導関数は関数と性質の異なるものだが、個人的に次の2つの理由から混同してしまう。

- 式で表すと絵面が同じ
- y = x ^ 2 で簡単のために x = 1 を例に用いると、導関数のグラフと接線のグラフの傾きが一致してしまう。

```{python}
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

## dy/dx

yをxで微分する、と読む。y=x^2のように変数がyとxだけなら「微分する」だけでも通じるが、y=ax^2+bx+cのような複数の変数がある場合は困ってしまう。そのための表し方。

### 微分の性質

- 線形性
  - $\frac{d}{dx}\{f(x)+g(x)\} = f'(x) + g'(x)$
  - $\frac{d}{dx}\{cf(x)\} = cf'(x)$
- Leibniz則: $\frac{d}{dx}\{f(x)g(x)\} = f'(x)g(x) + f(x)g'(x)$
- 合成関数の微分: $\frac{d}{dx}f(g(x)) = g'(x)f'(g(x))$

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

## Partial derivative（偏微分）

## Riemann integral（リーマン積分）

## 参考

- [微分積分キャンパス・ゼミ 改訂9](https://amzn.to/3ywWc37)
- [東京大学工学教室 - 基礎系数学 微積分](https://amzn.to/4bRVgVE)
