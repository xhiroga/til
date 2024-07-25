# パターン認識と機械学習 (Pattern Recognition and Machine Learning)

パターン認識と機械学習 [上](https://amzn.to/3yhqd7j)[下](https://amzn.to/4cMIQiI)

## 第9章 混合モデルとEM

### EMアルゴリズムが確かに尤度関数を極大化することの証明

観測変数$X$、潜在変数$Z$を持つモデルを考える。目的関数は次の対数尤度関数である。

$\ln p(x_1,x_2,...x_n|\theta) = \ln \prod_{n=1}^N p(x_n|\theta) = \sum_{n=1}^N \ln p(x_n|\theta) = \ln p(X|\theta)$

この対数尤度関数に含まれる確率を潜在変数Zの確率に対する周辺確率と見なすと、乗法定理によってXとZの同時確率$p(X,Z|\theta)$と事後確率$p(Z|X, \theta)$に分解できる。以降、対数周辺尤度、対数同時尤度、対数事後尤度と呼ぶ。

$\ln p(X|\theta) = \ln p(X,Z|\theta) - \ln p(Z|X,\theta)$

右辺の項について、集合を用いた表記は尤度であることを忘れがちなので、次の通り総積・総和を用いても示しておく。

$\ln p(X,Z|\theta) = \ln \prod_{n=1}^N p(x_n,z_n|\theta) = \sum_{n=1}^N \ln p(x_n,z_n|\theta)$

$\ln p(Z|X,\theta) = \ln \prod_{n=1}^N p(z_n|x_n,\theta) = \sum_{n=1}^N \ln p(z_n|x_n,\theta)$

$x_n$に対して$z_n$が一意に定まる場合、つまり完全データ集合$\{X,Z\}$が与えられている場合は、最尤推定によって$\theta$を計算できる。しかし$Z$は潜在変数であり与えられていない。

ここで、対数周辺尤度を対数同時尤度と対数事後尤度に分解した等式について、おもむろにxが与えられた上でのzの条件付き確率（負担率）である$q(z)$を用いて期待値を取る。と言っても、左辺の対数周辺尤度$\ln p(X|\theta)$はzを含まないため、そのままで良い。

<!-- おもむろではない説明がしたい -->

$\ln p(X|\theta) = \sum_Z q(Z) \ln p(X,Z|\theta) - \sum_Z q(Z) \ln p(Z|X,\theta)$

総和を用いても示す。

$\ln p(X|\theta) =
\sum_{n=1}^N (\sum_{k=1}^K  q(z=k|x_n) \ln p(x_n,z=k|\theta)) -
\sum_{n=1}^N (\sum_{k=1}^K  q(z=k|x_n) \ln p(z=k|x_n,\theta))$

等式を変形し、$\ln p(X|\theta)$がパタメータの更新によって増加することを示す。

$\ln p(X|\theta) = \sum_Z q(Z) \ln p(X,Z|\theta) - \sum_Z q(Z) \ln p(Z|X,\theta) + \sum_Z q(Z) \ln q(Z) - \sum_Z q(Z) \ln q(Z)$

$= \sum_Z q(Z) \ln \frac{p(X,Z|\theta)}{q(Z)} - \sum_Z q(Z) \ln \{\frac{p(Z|X,\theta)}{q(Z)}\}$

$= \sum_Z q(Z) \ln \frac{p(X,Z|\theta)}{q(Z)} + \sum_Z q(Z) \ln \{\frac{q(Z)}{p(Z|X,\theta)}\}$

変形した等式の右辺の1項目を$\mathcal{L}(q,\theta)$で表し、変分下限(ELBO, Evidence Lower Bound)と呼ぶ。また、2項目は$q(z)$を真の分布、$p(Z|X,\theta)$を近似分布と見做したKLダイバージェンス$KL(q||p)$である。従って、次の関係が成り立つ。

$\ln p(X|\theta) = \mathcal{L}(q,\theta) + KL(q||p)$

この式の意味を考え、次にこの式を用いてEMアルゴリズムを定義する。まず$\mathcal{L}(q,\theta)$は、$q(z)$が$p(Z|X)$と一致した際に$\ln p(X)$と等しい。次に$KL(q||p)$は常に非負である。

Eステップでは、$\theta$を固定して$q(z)$を更新する。$\ln p(X|\theta)$は$q(z)$に依存しないから、Eステップは対数尤度を据え置く。

Mステップでは、Eステップでより確からしくなった$q(z)$を用いて$\theta$を更新する。しかしながら、zの分布から求められるzの期待値は、潜在変数zの真の値とは当然一致しない（カテゴリカル分布等ではなく多項式を用いても完全一致しないことの方が多いだろう）したがって、$\ln p(X|\theta)$と$\mathcal{L}(q,\theta)$との間には差が生じる。その差が新たなKLダイバージェンスである。
