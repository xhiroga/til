# Probability and statistics (確率・統計)

## Expected Value, Variance, and Standard Deviation（期待値・分散・標準偏差）

- 期待値
- 分散
- 標準偏差
- 偏差
  - 平均や中央など、基準値との差のこと。特に基準を示さない場合は平均との差だと思って良さそうだ。
- Mean Absolute Deviation, MAD（平均絶対偏差）
  - 偏差（平均との差）の絶対値の平均。分散のように2乗しないため、一見便利そうに見える。
  - しかし、微分不可能なため、実は最適化や解析で扱いが難しいことがある...らしい。

## Moments and Moment Generating Functions（k次のモーメント・モーメント母関数（積率母関数））

確率分布の性質を知るためには、期待値や分散、歪度、尖度が役に立つ。これらを導出するためには、確率変数$X$の$k$乗の期待値が便利である。
これを$k$次のモーメントと呼ぶ。

また、それらを体系的に扱うためのツールとしてモーメント母関数がある。

## Discrete probability distribution（離散型確率分布）

## Bayes' Theorem（ベイズの定理）

[ベイズの定理を面積と計算グラフで表す](https://hiroga.hatenablog.com/entry/2024/05/07/111028)を参照。

## Binomial Distribution, Binomial theorem（二項分布, 二項定理）

$(a+b)^3$のような2項の累乗を、次の式で表せることを二項定理という。

$(a+b)^n = \sum_{k=0}^n \binom{n}{k}$

（ここで$\binom{n}{k}$とは${}_nC_k$を指す。計算方法は、「n通り、n-1通り、...から1つ選ぶをk回繰り返し、重複を避けるためk!通りで割る」なので、$\frac{\prod_{i=0}^{k-1}(n-i)}{k!}$である）

それ自体は中学校の数学で習うのだが、この式はそのまま、「成功か失敗かのいずれかの結果になる事象をn回行い、k回成功する確率」を求めるために利用できる。
