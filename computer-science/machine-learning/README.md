# Machine Learning（機械学習）

## Machine Learning（機械学習）（深層学習を除く）

機械学習には次のようなモデルがあります。

| モデル | 分類 | 説明 | 代表的なモデル | 応用例 |
|---|---|---|---|---|
| 線形回帰 | 教師あり学習 | 最も単純な回帰手法であり、入力に対して線形の関係を仮定する | - | 家屋価格の予測、人口増加にP伴う犯罪発生数の予測 |
| ロジスティック回帰 | 教師あり学習 | 二値分類問題や多クラス分類問題に使用され、確率的な分類を行う | - | 顧客の購入意向の予測、スパムメールの判別 |
| 決定木 | 教師あり学習| インスタンスの属性値による条件分岐によって、目的変数を予測する | ランダムフォレスト | データの可視化、健康診断の結果からの疾患予測 |
| k近傍法[^k-NN] | 教師あり学習 | もっとも近いk個のデータ点を参照して、入力データに最も近いカテゴリを予測する | k最近傍法 | 推薦システム |
| サポートベクターマシン | 教師あり学習 | 分類と回帰に使用され、高次元空間での線形および非線形分類を行う | - | 手書き数字の認識、がんの検出 |
| ニューラルネットワーク | 教師あり・なし両方 | 生物の神経細胞を真似したモデル。多層のノードが、ニューロンのように信号を処理・伝達する。 | CNN[^cnn]、RNN[^rnn]、オートエンコーダ、Transformer、GAN[^gan]など | 画像認識、音声認識、自然言語処理、自動運転 |

[^k-NN]: k-nearest neighbor algorithm, k-NN
[^cnn]: Convolutional neural network, 畳み込みニューラルネットワーク
[^rnn]: Recurrent neural network, 再帰型（回帰型）ニューラルネットワーク
[^gan]: Generative Adversarial Network, 敵対的生成ネットワーク

### k近傍法

教師あり学習の手法の一つ。ラベルが未知の入力データに対して、入力データと全データの間の距離を測定する。例えば、環境（気温・湿度・風速）から天気を予測する分類問題なら、気温・湿度・風速の3次元空間でのデータ間の距離を測定する。距離はユークリッド距離を使うことが多いが、原理的にはマンハッタン距離などでも構わない。

<!-- ### 最小二乗法 -->

<!-- ### 最尤推定法 -->

<!-- ### パーセプトロン -->

<!-- ### ロジスティック回帰とROC曲線 -->

<!-- ### k平均法 -->

<!-- ### EMアルゴリズム -->

<!-- ### ベイズ推定 -->

## Neural Network（ニューラルネットワーク） & Deep Learning（深層学習）

深い[^deep]層を持つニューラルネットワークをディープニューラルネットワークといい、ディープニューラルネットワークを用いた機械学習を深層学習と呼ぶ。

狭義の機械学習ではハンドクラフトで特徴量の抽出を行っていたが、深層学習ではモデルに任せている。

<!-- ここもっと良い書き方ある -->

従来は層を深くすると、学習が進まなくなる課題があった。誤差逆伝播法などのテクニックの導入によって、深い層のネットワークでも学習が進むようになった。

<!-- あれ、逆に誤差逆伝播法以前ってどうやってフィードバックしてたんだっけ？ -->

[^deep]: 何層からが深いかの厳密な定義はない。多くの研究者が、CAPが2より多い場合に深いと考えているという主張がある。[^sugiyama_2019]初期の深層学習の論文[^hinton_2006]では隠れ層が3層あるため、3層以上を深いともいえる。
[^sugiyama_2019]: [Human Behavior and Another Kind in Consciousness](https://amzn.to/3VpevAm) / [Google Books](https://books.google.com/books?id=9CqQDwAAQBAJ&pg=PA15)
[^hinton_2006]: [A fast learning algorithm for deep belief nets](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)

### Neural Network（ニューラルネットワーク）

![ニューラルネットワークの構造](https://miyabi-lab.space/assets/imgs/blog/upload/images/nn_fig17.001_ut1523588254.jpeg)
> [初心者必読！MNIST実行環境の準備から手書き文字識別までを徹底解説！ - MIYABI Lab](https://miyabi-lab.space/blog/10)

## CAPs(Credit Assignment Paths)

### RNN

### LSTM

### オートエンコーダ

次元削減や特徴抽出で便利なモデル。非線形の構造を持つデータにも有効な点が、PCAなどの従来の削減手法と異なる。1980年代にHintonらによって紹介されたとされる。[^Autoencoders_Unsupervised_Learning_and_Deep_Architectures]
（検索した限り、元論文には"autoencoder"という単語はないようだ。）[^Learning_internal_representations_by_error_propagation]

[^Autoencoders_Unsupervised_Learning_and_Deep_Architectures]: [Baldi, P. (2012, June). Autoencoders, unsupervised learning, and deep architectures. In Proceedings of ICML workshop on unsupervised and transfer learning (pp. 37-49). JMLR Workshop and Conference Proceedings.](http://proceedings.mlr.press/v27/baldi12a/baldi12a.pdf)
: [Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1985). Learning internal representations by error propagation. California Univ San Diego La Jolla Inst for Cognitive Science.](https://cs.uwaterloo.ca/~y328yu/classics/bp.pdf)

2000年代に研究が再燃し、画像のノイズ除去などに用いられるようになる。

### Transformer

#### Self-Attention

#### Encoder/Decoder

## Activation Function

活性化関数。非線形性（non-linearity）の1つ。

### Sigmoid function

シグモイド関数。ギリシア文字Σの語末系ςに似ていることから、*Sigma(シグマ)*+*-oid(~状のもの)*でシグモイドと呼ぶ。

#### Standard Sigmoid Function

```{python}
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x: int):
    return 1 / (1 + np.e ** -x)

x = np.linspace(-10, 10, 100)
y = sigmoid(x)
 
fig = plt.figure(figsize = (10, 5))
plt.plot(x, y)
plt.show()
```

### Softmax Function

ソフトマックス関数とは、数値の配列を確率の配列に変換する関数。

```{python}
import numpy as np

def softmax(x: float):
  # 2乗でもよいが、数値が大きくなった時に急激に差をつけられる点・計算コストが少ない点から、自然対数を用いるらしい。
  return np.exp(x) / np.sum(np.exp(x))

x = np.array([3, 1, 2])
y = softmax(x)

print(y)  # [0.66524096 0.09003057 0.24472847]
```

### ReLU

レルー（ランプ関数、正規化線形ユニット（Rectified Linear Unit））は、主にディープニューラルネットワークの中間層で用いられる活性化関数。

```{python}
import matplotlib.pyplot as plt
import numpy as np

def relu(x: int) -> int:
  return np.maximum(x, 0)

x = np.linspace(-4, 4, 100)
y = relu(x)
fig = plt.figure(figsize = (10, 5))
plt.plot(x, y)
plt.show()
```

### GELU

ガウス誤差線形ユニット（Gaussian Error Linear Unit）は、Transformer系のモデルでも採用される活性化関数。

## Gradient Descent

機械学習の訓練中に使用される最適化アルゴリズム[^optimizer]の一つ。

[^optimizer]: [【最適化手法】SGD・Momentum・AdaGrad・RMSProp・Adamを図と数式で理解しよう。](https://kunassy.com/oprimizer/)を参照。

訓練中の予測結果と実際の値の誤差を各パラメータに戻し、パラメータを更新することで、誤差が最小になるようにパラメータを更新していく。

## 教師あり学習

## 転移学習

深層学習モデルが学習を通じてデータの表現とタスク固有の学習を行っていることに着目し、すでに学習済みのモデルの重みを用いて新たなタスクの学習を行うこと。

|学習の分類|目的|手段|
|-|-|
|事前学習|より良い表現を得る|自己教師あり学習が多い?|
|ファインチューニング|個別のタスク特化する|教師あり学習?|

<!-- 自己教師あり学習が「多い」か要確認 -->

事前学習が教師あり学習の場合は、汎用的な表現を得ることそのものが学習目的ではないため、例えば牧草が写っていたら牛と分類してしまうようなショートカット学習が行われることがあり、これがファインチューニングの妨げになる。

## 自己教師あり学習

ネットワーク構造の工夫より大量のデータを用意することでモデルの性能が向上すること、また転移学習により汎用的なモデルが固有タスクにおいても性能を発揮することを背景として、大量のラベル無しデータから学習する方法が模索された。

あらかじめ存在する事実のデータから、学習のためのデータを自分で作成する手法が自己教師あり学習である。例えば、テキストの一部をマスクしてその単語を当てるとか、画像を加工した上で、加工前後の画像を見比べて同じであれば高い報酬を与えるなどである。

具体的な手法については、Masked Language Modeling、対比学習を参照すること。なお、自己教師あり学習は、BERTの論文では教師なし事前学習とも呼ばれていた。

評価の方法としては、ラベル付き分類データを用いて埋め込みを取得し（全結合層で変換する手前の値）、k近傍法を用いるもの、シンプルに下流タスク用のヘッドを取り付けて性能を測るもの、下流タスクのための層を加えてフルパラメータのファインチューニングを行うものなどがある。

### 対比学習（対照学習）

事前学習としての表現学習に用いられる手段の一つ。エンコーダとしての多層ニューラルネットワークと射影ヘッドからなるモデルに対して、ミニバッチで複数のデータ拡張された画像を与える。

対比学習の手法の1つ、SimCLRではInfoNCE損失関数([Desmos](https://www.desmos.com/calculator/nh1ntozu9o)[[🔐](https://www.desmos.com/calculator/mbn55ivvh6)])を用いる。
ミニバッチ内の同じ画像のデータ拡張から得たベクトルのコサイン類似度は近くなるように、そうでないベクトルのコサイン類似度は遠くなるように学習を進める。(正例のコサイン類似度/負例のコサイン類似度合計)にマイナスを付けて損失にするが、exponentialを取ってから自然対数を取り直す一工夫が入っている（正例・負例の部分を逆数にした方が、マイナスが取れて式がシンプルでは？）

## 半教師あり学習

## References

- [深層学習による画像認識の基礎](https://amzn.to/3wMy8sQ)
