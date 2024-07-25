# パターン認識と機械学習 (Pattern Recognition and Machine Learning)

パターン認識と機械学習 [上](https://amzn.to/3yhqd7j)[下](https://amzn.to/4cMIQiI)

## 第9章 混合モデルとEM

### EMアルゴリズムが確かに尤度関数を極大化することの証明

観測変数$X$、潜在変数$Z$を持つモデルを考える。目的関数は次の対数尤度関数である。

$\ln p(X|\theta) = \ln \prod_{n=1}^N p(x_n|\theta) = \sum_{n=1}^N \ln p(x_n|\theta)$

この対数尤度関数に含まれる確率を潜在変数$Z$の確率に対する周辺確率と見なすと、条件付き確率の定義により$X$と$Z$の同時確率$p(X,Z|\theta)$と事後確率$p(Z|X, \theta)$に分解できる。

$\ln p(X|\theta) = \ln p(X,Z|\theta) - \ln p(Z|X,\theta)$

右辺の項について、総和を用いて次のように表すこともできる。

$\ln p(X,Z|\theta) = \sum_{n=1}^N \ln p(x_n,z_n|\theta)$

$\ln p(Z|X,\theta) = \sum_{n=1}^N \ln p(z_n|x_n,\theta)$

$x_n$に対して$z_n$が一意に定まる場合、つまり完全データ集合${X,Z}$が与えられている場合は、最尤推定によって$\theta$を計算できる。しかし$Z$は潜在変数であり与えられていない。

ここで、$Z$の真の分布が未知であるため、近似分布$q(Z)$を導入する。$q(Z)$は$X$が与えられた上での$Z$の条件付き確率（負担率）を表す。この$q(Z)$を用いて、先ほどの等式の期待値を取る。と言っても、左辺の対数周辺尤度$\ln p(X|\theta)$はzを含まないため、そのままで良い。

$\ln p(X|\theta) = \sum_Z q(Z) \ln p(X,Z|\theta) - \sum_Z q(Z) \ln p(Z|X,\theta)$

総和を用いても示す。

$\ln p(X|\theta) =
\sum_{n=1}^N (\sum_{k=1}^K  q(z=k|x_n) \ln p(x_n,z=k|\theta)) -
\sum_{n=1}^N (\sum_{k=1}^K  q(z=k|x_n) \ln p(z=k|x_n,\theta))$

次に、この等式を変形し、$\ln p(X|\theta)$がパラメータの更新によって増加することを示す。

$\ln p(X|\theta) = \sum_Z q(Z) \ln p(X,Z|\theta) - \sum_Z q(Z) \ln p(Z|X,\theta) + \sum_Z q(Z) \ln q(Z) - \sum_Z q(Z) \ln q(Z)$
$= \sum_Z q(Z) \ln \frac{p(X,Z|\theta)}{q(Z)} - \sum_Z q(Z) \ln \frac{p(Z|X,\theta)}{q(Z)}$
$= \sum_Z q(Z) \ln \frac{p(X,Z|\theta)}{q(Z)} + \sum_Z q(Z) \ln \frac{q(Z)}{p(Z|X,\theta)}$

変形した等式の右辺の第1項を$\mathcal{L}(q,\theta)$で表し、変分下限(ELBO, Evidence Lower Bound)と呼ぶ。また、第2項は$q(Z)$を真の分布、$p(Z|X,\theta)$を近似分布と見做したKLダイバージェンス$KL(q||p_\theta)$である。ここで$p_\theta$は$p(Z|X,\theta)$を表す。従って、次の関係が成り立つ。

$\ln p(X|\theta) = \mathcal{L}(q,\theta) + KL(q||p_\theta)$

この式を用いてEMアルゴリズムを以下のように定義する：

- Eステップ：$\theta$を固定して$q(Z)$を更新する。具体的には、$q(Z) = p(Z|X,\theta)$とする。これにより$KL(q||p_\theta)$が0になり、$\ln p(X|\theta) = \mathcal{L}(q,\theta)$となる。
- Mステップ：Eステップで得られた$q(Z)$を用いて$\mathcal{L}(q,\theta)$を$\theta$について最大化する。これにより、$\ln p(X|\theta)$の下限を上げる。

EMアルゴリズムの各反復において、以下のことが保証される：

- Eステップでは、$q(Z)$を$p(Z|X,\theta)$に設定する。これにより$KL(q||p_\theta)$は0になるが、$\ln p(X|\theta)$自体は$q(Z)$に依存しないため変化しない。むしろ、$\mathcal{L}(q,\theta)$が$\ln p(X|\theta)$に等しくなる。
- Mステップでは$\mathcal{L}(q,\theta)$が増加するため、$\ln p(X|\theta)$も必ず増加する。

したがって、EMアルゴリズムは各反復で$\ln p(X|\theta)$を単調増加させ、局所最適解に収束する。ただし、$\mathcal{L}(q,\theta)$の最大化が解析的に行えない場合は、数値的な最適化手法を用いる必要がある。
この証明により、EMアルゴリズムが対数尤度関数を確実に極大化することが示された。
