# 機械学習 (machine learning)

ページの構成の際、次の情報源を参考にした。

- [機械学習 (東京大学工学教程)](https://amzn.to/4bS8SjZ)
- パターン認識と機械学習 [上](https://amzn.to/3yhqd7j)[下](https://amzn.to/4cMIQiI)
- [ITエンジニアのための機械学習理論入門](https://amzn.to/3yPDrrU)
- [ゼロから作るDeep Learning](https://amzn.to/4ddQgLL)
- [ゼロから作るDeep Learning ❷](https://amzn.to/3zSlbif)
- [ゼロから作るDeep Learning ❹](https://amzn.to/3zR4KCO)
- [ゼロから作るDeep Learning ❺](https://amzn.to/46g6mSB)
- [深層学習 改訂第2版](https://www.kspub.co.jp/book/detail/5133323.html)
- [深層学習による画像認識の基礎](https://amzn.to/3wMy8sQ)
- [Claude🔐](https://claude.ai/chat/a92b1ea8-5cd4-454d-99db-a2a35e7f8571)

<!-- ページの設計思想
- 主に基盤になっている技術によって章を分けた（機械学習 → ニューラルネットワーク → 大規模事前学習）
- ただし、強化学習と生成モデルは個別の章とした。これはゼロつくを参考にしやすいようにしたため。
- それぞれの章が似通った節を持つようにした（アルゴリズム | (学習手法 → アーキテクチャ) → 評価方法）。ただし、章ごとに参考にした本の影響を大きく受ける。
 -->

## 基礎知識

ベクトルの成分は、通常列ベクトルで表す。これは普段横書きをしている身からすると、あるいはプログラミングで配列を定義するときを考えると意外に思われるが、そうなっている。

> [!NOTE]
行は集合の要素のために取っておきたかった、という意図なら、個人的には納得がいく。

## 機械学習 (machine learning) （深層学習を除く）

機械学習の手法は、パラメトリックとノンパラメトリックに大別されます。

- パラメトリックモデルは、次の手順で推定します[^nakai_2021]
  1. パラメータを含むモデルを設定する
  2. パラメータを評価する基準を決める
  3. 最良の評価を与えるパラメータを決定する
- ノンパラメトリックは、パラメータを設定しません。十分なデータがあり、かつ事前知識が少ない場合に適しています。

[^nakai_2021]: [[改訂新版]ITエンジニアのための機械学習理論入門](https://amzn.to/3yPDrrU)

パラメトリック手法において、パラメーターを評価する基準を求める関数をobjective function（目的関数）といいます。また、最良の評価を与えるパラメータを決めるためには最適化のアルゴリズムを決める必要があります。

| 手法               | パラメトリック | 目的関数 | 最適化アルゴリズム | タスク         |
| ------------------ | -------------- | -------- | ------------------ | -------------- |
| 最小二乗法         | パラメトリック | 誤差関数 | 解析的に求まる     | 回帰           |
| 最尤推定法         | パラメトリック | 尤度関数 | 解析的に求まる     | 回帰           |
| 単純パーセプトロン | パラメトリック | 誤差関数 | 確率的勾配降下法   | 分類           |
| ロジスティック回帰 | パラメトリック | 尤度関数 | IRIS法             | 分類           |
| k平均法            |                | 二重歪み |                    | クラスタリング |
| k近傍法            |                |          |                    | 分類           |
| EMアルゴリズム     |                |          |                    |                |
| SVM                |                |          |                    |                |
| 決定木             |                |          |                    |                |

### 機械学習のタスクと分類

### 機械学習の学習手法

モデルの評価にあたって、性能が過学習によるものではないことを示す必要がある。そこで、データセットを学習用とテスト用にランダムに分けることが考えられる。また、ハイパーパラメータの調整用に、さらに検証データを分けることもある。

#### MinMax法

WIP

### 機械学習のアルゴリズム

#### 線形回帰モデル

##### 最小二乗法

最小二乗法において、重みを解析的に導くための式展開は次の通り。誤差関数の$\frac{1}{2}$は微分した際の係数2を相殺するための定数。

![formula](https://i.gyazo.com/thumb/3024/4dcdeb0865b6c5bf6492619d3ca455b7-heic.jpg)

なお、方程式を重みについて解くために、偏微分によって現れるデザイン行列のグラム行列が逆行列を持つことを示す必要がある。デザイン行列が 列数 ≦ 行数 のとき（つまり特徴数よりバッチサイズが大きいとき）、グラム行列は正定値であり、逆行列を持つ。

![formula](https://i.gyazo.com/thumb/2491/de921cb60b69d0a58d0e4a6eee126711-heic.jpg)

<!-- 正則化項, CS 2018-02 2 -->

##### 最尤推定

尤度関数は次の通り。なお、確率分布として正規分布を採用している。

$P = \prod_{n=1}^{N} N(t_n|f(x_n), \sigma^2)$

<!-- https://i.gyazo.com/thumb/2420/c36546337d0c0125e4fefcbfdcad1d5d-heic.jpg -->

#### 線形識別モデル

##### 単純パーセプトロン (perceptron)

パーセプトロンとは、説明変数から0か1を出力する二値分類モデルである。このモデルでは、バイアス項1を含む説明変数と重みを線形結合し、その結果にステップ関数（閾値関数）を適用する。ステップ関数は、線形結合の結果が閾値（通常0）以上なら1を、そうでなければ0を返す。ここでは、入力層と出力層が1層ずつの単層パーセプトロンを考える。

ステップ関数を使用しているため、出力の0と1の間の変化が不連続である。そのため、重みに関して偏微分して解析的に解くことができない。
また、ステップ関数を適用する前の値（線形結合の値）に対する目的変数は与えられていないため、線形結合前の値で解析的に解くこともできない（もし与えられていたら線形回帰になってしまう）。

![formula](https://i.gyazo.com/thumb/2486/69740dced1edf6f1259ccbe35f65aeff-heic.jpg)

解析的に解く代わりに、徐々に重みを正確にする方法がいくつか挙げられる。

パーセプトロン学習則では、誤分類されたデータに対して、説明変数に学習率をかけた値を直接重みから引く（または足す）。[^biopapyrus_2021]
[^biopapyrus_2021]: [パーセプトロン学習則](https://axa.biopapyrus.jp/deep-learning/perceptron-learning-rule.html)

一方で、誤分類されたデータに対して、線形結合を重みに関して偏微分し、その勾配の方向に重みを更新することで、誤差を減らす方法がある。これを勾配降下法という。データをランダムに選んで更新を行う方法を特に確率的勾配降下法と呼ぶ。

![確率的勾配降下法](https://i.gyazo.com/thumb/2120/0bbbe23b38e3d5f028ad13083d2ee8fa-heic.jpg)

##### ロジスティック回帰

ロジスティック回帰という名前だが分類のアルゴリズム。[ロジスティック回帰は回帰か分類か](https://scrapbox.io/nishio/ロジスティック回帰は回帰か分類か)も参照。

#### カーネル法

#### 疎な解を持つカーネルマシン

##### SVM

分類・回帰アルゴリズムの1つ。データを分離する超平面を探す点はパーセプトロンと同じだが、超平面付近の点（サポートベクター）とのマージンを最大化することで汎化性能を引き上げている。また、データを高次元空間に写像することで、元の空間では線形分離できないデータを分離できるようになる。

#### グラフィカルモデル, 系列データ

##### ベイジアンネットワーク

条件付き確率の依存関係を有向非巡回グラフで表したものをベイジアンネットワークという。単純な分解では全てのノード間に辺が存在するため、逆にどのノード間に辺が無いかによってその確率が特徴づけられる。[ベイジアンネットワークで見る変数の因果関係](https://www.youtube.com/watch?v=knCbMFQJXxY)も参照。

##### マルコフ確率場

確率変数どうしがお互いに依存している関係を無向グラフで表したものをマルコフ確率場という。例えば、2x2ピクセルの画像の隣接するセルどうしは、同じ色である確率が高いため、マルコフ確率場で表せる。

##### カルマンフィルタ

ロボット制御などで、センサーの情報から実世界の乗法を推測することを考える。単純な例として、温度計が24度を示し、測定誤差が±1度であることを知っているとする。また、現在までに推定した温度が22度、その不確かさが±4度とする。ここで、新たな温度と不確かさを次の通り求める。[^geolab_kalman01]
[^geolab_kalman01]: [第1回　カルマンフィルタとは](https://www.geolab.jp/documents/column/kalman01/)

1. 現在までの推定の不確かさと測定誤差の比を取る。このとき、現在までの推定の不確かさの割合をカルマンゲイン(K)とする。
2. 新しい推定値 = K x 測定値 + (1-K) x 現在までに推定した温度 とする。
3. 新しい不確かさ = (1-K) x 現在までの推定の不確かさ とする。

つまり、測定値と現在までの推定値の、不確かさによる重み付き平均で新しい推定値を表し、次に測定誤差が小さいほど新たな不確かさが小さくなるように重み付けをしている。

#### 混合モデルとEM

##### k近傍法 (k-nearest neighbor algorithm)

分類アルゴリズムの1つ。例えば、天気を予測する問題を考える。気温・湿度・風速が与えられており、過去のデータ（教師データ）が利用できるとする。このとき、入力データを気温・湿度・風速の3軸からなる3次元空間にプロットし、最も近いk個のデータのラベルで多数決を取って、そのラベルを入力データのラベルの予測とする。このアルゴリズムをk近傍法といい、特にk=1の時に最近謗法という。データセット上のすべてのテントの距離を計算する必要があるため、データのサイズが大きい時にそのまま使うと時間がかかる。

また、過去のデータをそのまま用いるのではなく、ラベル毎にデータの重心を求め、入力データから最も近い重心のラベルを予測に用いるアルゴリズムをNCC (Nearest Centroid Classifier, あえて訳せば最近傍重心法)という。

##### k平均法 (k-nearest means algorithm)

クラスタリングの手法。NCCを教師なし学習で行う。といっても、教師データがないためにラベルがないから、適切な重心がどこだか分からない。そこで、初めに適当な位置に重心をばらまく。次に、重心毎に入力データを分類し、分類されたデータの重心を新たな重心とすることを繰り返す。このアルゴリズムをk平均法という。

##### EMアルゴリズム

WIP

#### 近似推論法

#### サンプリング法

##### マルコフ連鎖モンテカルロ

円周率を近似する有名な方法にモンテカルロ法がある。円とそれに接する正方形めがけてランダムに粒を撒き、円に入った数と正方形に入った数の比から円周率を求める方法である。

マルコフ連鎖で表される状態遷移を考える。例えば、前日の天気に応じて今日の天気が決まるとする。前日の天気が分からなくても、それぞれの天気の変わりやすさ（例えば、晴れの翌日は70%晴れ、とか）が分かっていれば、分からないなりに一番当たりそうな天気を言うことができそうだ。確率過程の言葉で言えば、単純な天気のような場合は、連立方程式を解いて定常状態を求めることができる。しかし、例えば分子の動きのシミュレーションのように、状態（この場合はそれぞれの分子）が多すぎて解析的に定常状態を求められなかったらどうするか。ここでモンテカルロ法を用いる。つまり、ランダムな初期状態から求めたい時点での状態を計算することを無数に繰り返し、そこから確率分布を見出す方法である。

#### 連続潜在変数

##### 主成分分析

次元の多いデータを説明するにあたって、主な成分とそれを補う成分で説明できると全体像が分かりやすい。例えば、中学校のある学年のテストの成績のばらつきを説明する時に、「国語は〇〇で得意不得意が分かれて、数学は✗✗で〜」と全て説明するより、「総合成績で見ると3グループに分けられます。平均的なグループは、文系科目と理系科目で次のように分かれます...」のような説明の方が、頭の中で順を追って思い出しやすい。

次元削減の手法の中でも、データの特徴を最もよく表す次元から順に見出す方法として主成分分析(PCA, Principal Component Analysis)がある。[^intage_401]手順としては、射影した時にすべての次元の特徴の分散が最大になるような線を初めに引き、次にその線と直行する線の中で、同様に射影すると分散が最大になるような線を選ぶ。このようにして、データを説明できると判断できるところまで線、つまり成分を選ぶ。なお、これは相関行列の固有値分解と等価であるそうだ。
[^intage_401]: [主成分分析とは](https://www.intage.co.jp/glossary/401/)

各種成分は元の特徴量の線形結合なので、PCAは線形な手法と言える。

PCAはニューラルネットワークに対しても適用することができる。例えば、隠れ層の値を視覚化し、似たデータが近くにあるかを分析することは、モデルが適切に学習できてない場合の原因を探ることに役立つ。

##### 多次元尺度法 (MDS)

アンケートなどの5段階評価のような手法を尺度法という。ちなみに、同意の度合いを測るものをリッカート尺度法、真逆の形容詞のどちらに近いかを測るものを評定尺度法という。

データを低次元に射影するにあたって、データ間の距離（ユークリッド距離が一般的だが、それ以外でも良い）を保つように射影する方法を多次元尺度法という。

##### t-SNE

PCAのような線形な手法では、複雑な非線形構造を持つデータの局所的な関係性を保持することが困難であった。t-SNEでは、高次元空間におけるデータどうしの類似度を、データ$x_1$を選んだ後にデータ$x_2$を選ぶ条件付き確率として定義している。その結果、t-SNEでは局所的な関係性を可視化できるようになった。

##### UMAP

#### モデルの結合

##### ブースティング

##### 決定木

分類・回帰アルゴリズムの1つ。フローチャートによる分類を考える。例えば、お客さんにパソコンをオススメする。パソコン初心者で、かつ授業でも使うなら、Windowsを搭載したノートPCが良いだろう。このように、入力データのパラメータに応じて根から木をたどり、葉にたどり着いたら提案を行うモデルを決定木という。タスクによって分類木、回帰木ともいう。

決定木は性能が低いが計算コストが安い。そこで、複数の決定木を用意し、それらの重み付き多数決で予測を行うこととする。このような手法をアンサンブル学習といい、特に決定木で行う場合をランダムフォレストという。

### 機械学習の評価方法

予測の精度を測る指標は次の通り。

- Accuracy（正解率）
  - 全ての予測に対する、正しい予測の割合
- Precision（適合率）
  - $TP/(TP+FP)$
  - 件数を減らしてでも偽陽性を防ぎたい場合に良い指標になる。ホワイトリスト向き
- Recall（再現率）
  - $TP/(TP+FN)$
  - 件数を増やしてでも偽陰性を防ぎたい場合に良い指標となる。ブラックリスト向き
  - 例: セキュリティのアラート
- F-measure（F値）
  - 適合率と再現率の調和平均

前述の指標を踏まえて、機械学習の評価に用いられる指標は次の通り。

- Receiver Operating Characteristic曲線（ROC曲線）
  - 縦軸に真陽性率（=再現率）、横軸に偽陽性率を置いたグラフ
  - 再現率を上げようと何でも陽性で判定すると、同時に偽陰性も増える、というトレードオフを表している
- Recision-Recall Curve（PR曲線）
  - 縦軸に適合率、横軸に再現率を置いたグラフ

## ニューラルネットワーク (neural network), 深層学習 (deep learning)

深い[^deep]層を持つニューラルネットワークをディープニューラルネットワークといい、ディープニューラルネットワークを用いた機械学習を深層学習と呼ぶ。

狭義の機械学習ではハンドクラフトで特徴量の抽出を行っていたが、深層学習ではモデルに任せている。

<!-- ここもっと良い書き方ある -->

従来は層を深くすると、学習が進まなくなる課題があった。誤差逆伝播法などのテクニックの導入によって、深い層のネットワークでも学習が進むようになった。

<!-- あれ、逆に誤差逆伝播法以前ってどうやってフィードバックしてたんだっけ？ -->

[^deep]: 何層からが深いかの厳密な定義はない。多くの研究者が、CAPが2より多い場合に深いと考えているという主張がある。[^sugiyama_2019]初期の深層学習の論文[^hinton_2006]では隠れ層が3層あるため、3層以上を深いともいえる。
[^sugiyama_2019]: [Human Behavior and Another Kind in Consciousness](https://amzn.to/3VpevAm) / [Google Books](https://books.google.com/books?id=9CqQDwAAQBAJ&pg=PA15)
[^hinton_2006]: [A fast learning algorithm for deep belief nets](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)

### 活性化関数

活性化関数。非線形性（non-linearity）の1つ。

#### Sigmoid function

シグモイド関数。ギリシア文字Σの語末系ςに似ていることから、*Sigma(シグマ)*+*-oid(~状のもの)*でシグモイドと呼ぶ。

#### Standard Sigmoid Function

```python
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

#### Softmax Function

ソフトマックス関数とは、数値の配列を確率の配列に変換する関数。

```python
import numpy as np

def softmax(x: float):
  # 2乗でもよいが、数値が大きくなった時に急激に差をつけられる点・計算コストが少ない点から、自然対数を用いるらしい。
  return np.exp(x) / np.sum(np.exp(x))

x = np.array([3, 1, 2])
y = softmax(x)

print(y)  # [0.66524096 0.09003057 0.24472847]
```

#### ReLU

レルー（ランプ関数、正規化線形ユニット（Rectified Linear Unit））は、主にディープニューラルネットワークの中間層で用いられる活性化関数。

```python
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

#### GELU

ガウス誤差線形ユニット（Gaussian Error Linear Unit）は、Transformer系のモデルでも採用される活性化関数。

### 確率的勾配降下法 (SGD)

機械学習の訓練中に使用される最適化アルゴリズム[^optimizer]の一つ。

[^optimizer]: [【最適化手法】SGD・Momentum・AdaGrad・RMSProp・Adamを図と数式で理解しよう。](https://kunassy.com/oprimizer/)を参照。

訓練中の予測結果と実際の値の誤差を各パラメータに戻し、パラメータを更新することで、誤差が最小になるようにパラメータを更新していく。

### 誤差逆伝播法 (BP, Backpropagation)

単純パーセプトロンは特徴空間内で超平面で分離（線形分離）できる分類問題しか解けない。そのため、多層パーセプトロンが考案された。しかし、当初は多層パーセプトロンを効率的に学習させる方法が見つかっていなかった。例えば、ランダムに重みを更新する、層別に学習させる、などの方法が取られていた可能性がある。その後、ニューラルネットワークを微分可能な関数として見なし、隠れ層の誤差を、次の層の誤差を重みで偏微分して計算する誤差逆伝播法が考案された。

#### 勾配消失, 勾配爆発 (vanishing gradient problem, exploding gradient problem)

誤差逆伝播法において、誤差が伝播する内に小さくなって消失したり、または大きくなって発散することがある。この性質は、シグモイド関数を考えると想像しやすい。シグモイド関数では、数字を0~1の値に圧縮する反面、出力から引数を復元するのは難しい。誤差逆伝播法において誤差が消失または発散することを、勾配消失・勾配爆発という。

勾配消失・勾配爆発の対策として、活性化関数の差し替えがある。例えば、値の範囲を圧縮するシグモイド関数に代わって、0以上の値を保つReLUを用いることが考えられる。

また、勾配の値を正規化することも考えられる。同じ層のすべての勾配のノルムが閾値になるように調整すれば良い。これを勾配クリッピングという。

勾配消失・勾配爆発は、時系列データを扱うRNNで特に問題となる。詳しくは[LSTM](#lstm-long-short-term-memory)を参照。

#### 残差接続

### 深層学習のアーキテクチャ

#### 畳み込みニューラルネットワーク (CNN, Convolutional Neural Network)

画像の分類タスクを考える。例えば、トラとライオンを分類するとする。トラには縞模様があるから、ニューラルネットワークの下位レイヤーは模様を検出しそうだ。そのような働きを促進するため、訓練可能なフィルターを設けることを考える。具体的には、入力データと積和演算を行う適当なサイズの行列（例えば、3x3）を作り、そのフィルターを1ピクセルづつ（この間隔は調整できる）ずらしていく。このような層を畳み込み層という。

また、画像処理は1ピクセルのズレに対して鈍感であってほしい。トラの画像をトリムしたり回転させてもトラであるため。そこで、入力データを縮小することを考える。具体的には、適当なサイズ（例えば、2x2）ごとに入力データのの最大値や平均を取って、新たな行列の要素とする。この対象領域をウィンドウと呼ぶが、例えばウィンドウが2x2なら、入力に対する出力のサイズは1/4になる。このような層をプーリング層という。

#### リカレントニューラルネットワーク (RNN, Recurrent Neural Network)

ニューラルネットワークによる文章の生成を考える。文章は単語や文字をベクトル（例：one-hot encodingや単語埋め込み）に変換することでコンピュータが扱える形式になる。そのため、ベクトルを入力にベクトルを出力するニューラルネットワークを考えれば良い。

しかし、単純な実装ではトークン数とベクトルの次元の積の入力が必要となる。この方式は入力可能なトークン数に限度があるし、限度を変更する度に再度訓練が必要になる。そこで、トークンは1つづつ処理することにする。代わりに、前のトークンを入力にした隠れ層を次のトークンの入力と合わせて使う。この隠れ層は過去の情報を要約して保持する役割を果たす。これを循環的に行うことで、入力サイズは固定長でありながら、自由な長さの文字列を扱うことができる。このモデルをRNNという。

個人的な印象としては、RNNの隠れ状態の伝播と全加算器における繰り上がりは似ている。実際、RNNで全加算器を実装した記事も存在する。[^aaaaaaaaaaaaaaaaaa_2021]
[^aaaaaaaaaaaaaaaaaa_2021]: [E検定で出てくるリカレントニューラルネットワークと強化学習について](https://qiita.com/aaaaaaaaaaaaaaaaaa/items/2f6b023066b05dd1b371)

RNNは長い系列を扱える一方で、系列が長くなると過去の情報を保持しにくくなる課題もある。

#### LSTM (Long short-term memory)

RNNでは、隠れ層の状態を次のタイムステップにおける同じ隠れ層の入力に用いる（隠れ状態）ことによって状態を扱っている。しかし、隠れ状態は活性化関数を経由するため勾配消失が発生しやすい。また、次のタイムステップのために何を忘れ、何を覚えるかを明示的に学習させることができない。そこで、忘れたり、新たなタイムステップの入力を覚えるにあたって、活性化関数を用いない状態を別途設ける。これをセル状態と呼ぶ。隠れ層の出力にあたっては、セル状態と隠れ状態の2つを入力として計算する。しかし、隠れ状態とは別にセル状態も出力し、どちらも次のタイムステップの入力となる。

セル状態について、隠れ状態と入力を元に何を忘れるかを選択する部品をforgetゲートと呼ぶ。また、隠れ状態と入力を元に新たなタイムステップの情報を追加する部品をinputゲート、セル状態と隠れ状態を合わせて出力する部品をoutputゲートと呼ぶ。セル状態と3つのゲートを備えたRNNをLSTMと呼び、短期記憶を長期に渡って保持することを指す。

Elman RNN(LSTMではないシンプルなRNNを指す)やLSTMは、分類や予測・生成タスクに用いることができる。分類タスクの場合は最後のタイムステップの出力のみを用いる。

#### seq2seq

LSTMでは、系列の分類タスクや、系列のタイムステップと出力が1:1で対応するようなタスクを処理できた。次に、系列から系列を予測・生成するが、タイムステップどうしが必ずしも1:1で対応しないタスクを考える。例えば、日本語から英語への翻訳である。

系列から系列の変換タスクでは、一度全ての入力情報が隠れ状態に含まれるのを待ち、それから出力を始める。したがって、入力が完了したことを示す工夫などが求められる。そこで、入力を隠れ状態へと変換するLSTM（エンコーダ）と、隠れ状態を元に系列を生成するLSTM（デコーダ）を繋げることを考える。2つのLSTMで役割を分担することで、エンコード・デコードに特化した学習や機能の導入が簡単になる。このようなアーキテクチャをエンコーダ・デコーダモデルといい、エンコーダ・デコーダモデルを含む系列から系列への変換を行うモデルをseq2seqという。

#### seq2seq with attention

エンコーダ・デコーダモデルによる翻訳タスクを考える。固定長である。

#### グラフニューラルネットワーク (GNN)

（要レビュー）ニューラルネットワークは一般的に、データを多次元変数として捉えた上で、変数の重み付きの和を新たな次元とすることで特徴量を自動で作る。CNNでは周辺のマスの重み付き和を、Transformerでは全範囲の重み付き和を用いる。これは、入力の範囲をグラフ構造で与えることで一般化できる。物体の各点が近い点からの相互作用を受けることに着目し、GNNを用いて自然なシミュレーションを行った応用などがある。[^joisino_2024]
[^joisino_2024]: [僕たちがグラフニューラルネットワークを学ぶ理由](https://speakerdeck.com/joisino/pu-tatigagurahuniyurarunetutowakuwoxue-buli-you)

### 推論の信頼性

### 説明と可視化

### 深層学習のいろいろな学習手法

#### 継続, 追加学習

#### 知識蒸留

#### 枝刈り

#### 量子化

#### ネットワーク構造探索 (NAS)

### データが少ない場合の学習

## 強化学習 (reinforcement learning)

エージェントが環境の中で行動する毎に、新たな状態を観測し、かつ正または負の報酬を与えることで、エージェントが環境内で最も収益の良い行動を取るように導く機械学習の手法。

初めに、強化学習で用いられる用語と変数について、由来を次の通りまとめる。

- 報酬
  - 変数Rの由来はRewardと思われる
- 収益
  - ある状態における、その状態からの生涯年収のようなもの
  - 変数Gの由来はGainと思われる
- ロールアウト
  - ポリシーの更新を指す

強化学習における訓練のサイクルは次の通り。

1. エピソードの開始: 環境をリセットし、エージェントに状態を観測させ、行動を選択させる。その後、新たな状態を返し、報酬を与える
2. エピソードの終了: エピソードの終了条件を満たした場合、エピソードを終了する。環境をリセットし、次のエピソードに備える。
3. ポリシーの更新: エピソード終了時、または適切な間隔で、エージェントは方策を更新する。

また、強化学習のタスクは次の2つに分類される。

- エピソードタスク
  - 終わりのある問題。
  - 囲碁、Atariのゲームなど。
  - `gymnasium.Env#step`の返すterminatedの値がTrueの場合、それはエピソードの終了を意味する。
- 連続タスク
  - 終わりのない問題。
  - 在庫管理の問題、ロボットの操作の問題など。

学習の流れとタスクの分類を踏まえて、深層強化学習のアルゴリズムはモデルベースとモデルフリー、更にモデルフリーは価値ベースと方策ベースに分類される。

- モデルベース
- モデルフリー
  - 価値ベース
    - ロールアウトがエピソード単位に行われる傾向がある
  - 方策ベース
    - ロールアウトが固定タイムステップで行われる傾向がある

`gymnasium.Env`においては、強化学習は次のように抽象化されている。

1. 最上位のレイヤーとしてアルゴリズムがあり、ポリシーと環境に加えてハイパーパラメータを内包する
2. ポリシーとして、多層パーセプトロンやCNNなどの深層ニューラルネットワークを持つ
3. 環境は`step`メソッドを必ず持つ。`step`は行動を引数に、観測結果と報酬を返す。

### 強化学習のアルゴリズム

#### 方策勾配法 (Vanilla Policy Gradient)

#### REINFORCE

#### PPO

方策ベースの手法。

#### Actor-Critic

価値ベースかつ方策ベースの手法。

#### 世界モデル (World Models)

価値ベースかつ方策ベースの手法。

## 生成モデル (generative model)

観測データの特徴変数$x$から目的変数$y$を推測するにあたって、$P(y|x)$の条件付き確率のみを用いるモデルを識別モデル(discriminative model)と呼ぶ。逆に、目的変数$y$が与えられた際に特徴変数が$x$である尤度$P(x|y)$と、$y$が観測できる事前確率$P(y)$を用いて、同時確率$P(x,y)$を最大化する$y$を求めるモデルを生成モデル(generative model)と呼ぶ。

### 生成モデルの学習手法

#### 敵対的生成ネットワーク (GAN, Generative Adversarial Networks)

生成モデルの学習フレームワークの1つ。生成モデルでは、これまで存在しなかったような画像などを生成する場合、元画像が存在しない。そのため、ニューラルネットワークへのフィードバックにあたって、元画像と生成画像を比較して尤もらしさを算出することができない。

これに対しては、すでに訓練済みの画像分類モデル（例えばResNet）などを用いる方法が考えられる。しかし、生成したい分野の訓練済みモデルが都合良くあるとは限らない。また、分類モデルは画像が生成されたかどうかを見破ることに特化されていないため、生成された画像がある程度尤もらしくなると、それ以上の品質向上に貢献できないかもしれない。そこで、生成器(generator)と併せて識別器(discriminator)を訓練することを考える。これをGANという。

### 生成モデルのアーキテクチャ

#### オートエンコーダ

次元削減や特徴抽出で便利なモデル。非線形の構造を持つデータにも有効な点が、PCAなどの従来の削減手法と異なる。1980年代にHintonらによって紹介されたとされる。[^Autoencoders_Unsupervised_Learning_and_Deep_Architectures]
（検索した限り、元論文には"autoencoder"という単語はないようだ。）[^Learning_internal_representations_by_error_propagation]

[^Autoencoders_Unsupervised_Learning_and_Deep_Architectures]: [Baldi, P. (2012, June). Autoencoders, unsupervised learning, and deep architectures. In Proceedings of ICML workshop on unsupervised and transfer learning (pp. 37-49). JMLR Workshop and Conference Proceedings.](http://proceedings.mlr.press/v27/baldi12a/baldi12a.pdf)
[^Learning_internal_representations_by_error_propagation]: [Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1985). Learning internal representations by error propagation. California Univ San Diego La Jolla Inst for Cognitive Science.](https://cs.uwaterloo.ca/~y328yu/classics/bp.pdf)

2000年代に研究が再燃し、画像のノイズ除去などに用いられるようになる。

#### 変分オートエンコーダ

#### 拡散モデル

### 生成モデルの評価

ページ構成の際、次の情報源を参考にした。

- [ChatGPT🔐](https://chatgpt.com/c/0c69a86c-096a-4a84-a265-c6df17de88cb)

#### BLEU

BLEU (Bilingual Evaluation Understudy, 発音はBlueと同じ)[^papineni_2002]は、機械翻訳等の評価に広く利用される評価指標。

[^papineni_2002]: [BLUE: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.pdf)

#### ROUGE

ROUGE (Recall-Oriented Understudy for Gisting Evaluation, ルージュ)[^lin_2004]は、要約タスクで用いられる評価指標。参照する要約と生成した要約の一致度を測ることを試みる。

[^lin_2004]: [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)

最も基本的なROUGE-Nでは、N-gramの単位で、人手の要約と機械の要約との共起を測る。N-gramがUnigramの場合、ROUGE-1と呼ばれる。[^icoxfog417_2017]
[^icoxfog417_2017]: [ROUGEを訪ねて三千里:より良い要約の評価を求めて](https://qiita.com/icoxfog417/items/65faecbbe27d3c53d212)

ここでは、ROUGE-1で次の要約を評価する。

- 参照する要約: The artist chose a deep rouge for the lips
- 生成した要約: The lips were painted with a deep shade of red lipstick

参照する要約の単語をどれだけ当てられたかがRecallとなるため、$3/10 = 0.3$。また、生成した要約の単語がどれだけ参照元出身かがPrecisionなので、$3/8=0.375$。
したがってF1スコアを求めると、$2*(0.3*0.375)/(0.3+0.375)=0.333...$となる。

## 大規模事前学習

### 大規模事前学習の学習手法

#### 転移学習

深層学習モデルが学習を通じてデータの表現とタスク固有の学習を行っていることに着目し、すでに学習済みのモデルの重みを用いて新たなタスクの学習を行うこと。

| 学習の分類           | 目的                 | 手段                    | 例                      |
| -------------------- | -------------------- | ----------------------- | ----------------------- |
| 事前学習             | より良い表現を得る   | 自己教師あり学習が多い? |                         |
| 継続事前学習         | より良い表現を得る   | 自己教師あり学習が多い? | 日本語コーパスでLLMなど |
| ファインチューニング | 個別のタスク特化する | 教師あり学習?           |                         |

<!-- 自己教師あり学習が「多い」か要確認 -->

事前学習が教師あり学習の場合は、汎用的な表現を得ることそのものが学習目的ではないため、例えば牧草が写っていたら牛と分類してしまうようなショートカット学習が行われることがあり、これがファインチューニングの妨げになる。

#### 自己教師あり学習

ネットワーク構造の工夫より大量のデータを用意することでモデルの性能が向上すること、また転移学習により汎用的なモデルが固有タスクにおいても性能を発揮することを背景として、大量のラベル無しデータから学習する方法が模索された。

あらかじめ存在する事実のデータから、学習のためのデータを自分で作成する手法が自己教師あり学習である。例えば、テキストの一部をマスクしてその単語を当てるとか、画像を加工した上で、加工前後の画像を見比べて同じであれば高い報酬を与えるなどである。

具体的な手法については、Masked Language Modeling、対比学習を参照すること。なお、自己教師あり学習は、BERTの論文では教師なし事前学習とも呼ばれていた。

評価の方法としては、ラベル付き分類データを用いて埋め込みを取得し（全結合層で変換する手前の値）、k近傍法を用いるもの、シンプルに下流タスク用のヘッドを取り付けて性能を測るもの、下流タスクのための層を加えてフルパラメータのファインチューニングを行うものなどがある。

### 大規模事前学習のアーキテクチャ

#### BERT

#### Transformer (2017)

Transformer[^vaswani_2017]は単語間の長距離依存性を把握できるようになったニューラルネットワークである。具体的には、全単語間にAttention機構を導入したRNNである。
[^vaswani_2017]: A. Vaswani et al., “Attention is All you Need,” in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2017. Accessed: Jan. 05, 2024. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

TransformerがEncoder-Decoderブロックから構成される一方で、GPTはDecoderブロックからのみ構成される。Transformerが翻訳タスク向けに設計されたことが関係している。

Transformerでは単語の位置情報を知るため、位置埋込 (PE, Positional Encoding)を行う。単に位置インデックスを用いずに、トークンごとに計算したべクトル次元数分の波を用いることで、モデルが位置関係をより連続的に理解できる。

- Self-Attention
- Encoder/Decoder
- Multi-head attention
- Cross-Attention
- 残差結合
- 層正規化

Transformerは、様々な注意表現を学習するために異なるAttentionを何度も適用している。その結果、CNNのように同じフィルタを繰り返し適用するモデルと比較して、計算量やパラメータが多くなり、それらをメモリから読み出す頻度が上がった。

CPUの演算性能だけでなく、メモリI/Oを含めた性能を評価するためのモデルとしてルーフラインモデルがある。マシンの達成可能なFLOPSを計算するに当たり、CPUのピーク演算性能とメモリ帯域によって成約される性能の小さい方を取るもので、チャートが屋根のような形になることからそう呼ぶ。

![ルーフラインモデル](https://fukushimalab.github.io/hpc_exercise/docfig/roofline.png)

#### CLIP (2021)[^radford_2021]

[^radford_2021]: [A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision.” arXiv, Feb. 26, 2021. doi: 10.48550/arXiv.2103.00020.](https://doi.org/10.48550/arXiv.2103.00020)

大規模事前学習による画像言語モデル。画像・キャプションのペアを用いた対比学習による自己教師あり学習を行う。損失関数は次の通り（式の出典はCyCLIP[^goel_2022]）

$$
\mathcal{L}_{\text{CLIP}} =
-\frac{1}{2N} \sum_{j=1}^N \log \left[
\frac{\exp \left( \left\langle I^e_j, T^e_j \right\rangle / \tau \right)}{\sum_{k=1}^N \exp \left( \left\langle I^e_j, T^e_k \right\rangle / \tau \right)}
\right]
-\frac{1}{2N} \sum_{k=1}^N \log \left[
\frac{\exp \left( \left\langle I^e_k, T^e_k \right\rangle / \tau \right)}{\sum_{j=1}^N \exp \left( \left\langle I^e_j, T^e_k \right\rangle / \tau \right)}
\right]
$$

[^goel_2022]: [S. Goel, H. Bansal, S. Bhatia, R. A. Rossi, V. Vinay, and A. Grover, “CyCLIP: Cyclic Contrastive Language-Image Pretraining.” arXiv, Oct. 26, 2022. doi: 10.48550/arXiv.2205.14459.](https://arxiv.org/abs/2205.14459)

##### CyCLIP (2022)[^goel_2022]

CLIPは正例と近い負例・遠い負例の距離に注意を払っていないため、画像をプロンプトで分類した場合と正解ラベル付き画像を用いてk-近傍法で分類した場合で結果に差が出ることがある。

次の考え方に基づいて損失を調整することで、精度を改善することができる。具体的にはそれぞれの距離の差の二乗を損失に加えている。

1. 画像jとキャプションkの距離感は、画像kとキャプションjの距離感と同じであるべき
2. 画像jと画像kの距離感は、キャプションjとキャプションkの距離感と同じであるべき

##### PAINT (2022)[^ilharco_2022]

[^ilharco_2022]: [Patching open-vocabulary models by interpolating weights](https://arxiv.org/abs/2208.05592)

ファインチューニングによって汎化性能を失う問題に対して、ファインチューン前後の重みを線形補間した重みを用いることを提案している。これによって汎化性能と固有タスクを解く能力をある程度良いところ取りできるらしい。感想だが、ファインチューニングだけでは過学習が起きてしまう、ということを示唆しているように思える。

#### 対比学習（対照学習）

事前学習としての表現学習に用いられる手段の一つ。エンコーダとしての多層ニューラルネットワークと射影ヘッドからなるモデルに対して、ミニバッチで複数のデータ拡張された画像を与える。

対比学習の手法の1つ、SimCLRではInfoNCE損失関数([Desmos](https://www.desmos.com/calculator/nh1ntozu9o)[[🔐](https://www.desmos.com/calculator/mbn55ivvh6)])を用いる。
ミニバッチ内の同じ画像のデータ拡張から得たベクトルのコサイン類似度は近くなるように、そうでないベクトルのコサイン類似度は遠くなるように学習を進める。(正例のコサイン類似度/負例のコサイン類似度合計)にマイナスを付けて損失にするが、exponentialを取ってから自然対数を取り直す一工夫が入っている（正例・負例の部分を逆数にした方が、マイナスが取れて式がシンプルでは？）

#### VL-Checklist (2023)[^zhao_2023]

[^zhao_2023]: [T. Zhao et al., “VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations.” arXiv, Jun. 22, 2023. doi: 10.48550/arXiv.2207.00221.](https://arxiv.org/abs/2207.00221)

画像言語モデルの特性を調べるためのベンチマーク。画像のキャプションに登場する名詞や形容詞を適当に入替えても、正しく理解できているなら入れ替え後のほうがスコアが少ないべき、という考え方に基づいたテストを行う。

<!-- ## BLIP-2 -->

<!-- ## LLaVA -->

## LLM

ページの構成の際、次の情報源を参考にした。

- [LLM 大規模言語モデル講座 2023](https://weblab.t.u-tokyo.ac.jp/llm_contents/)
- [大規模言語モデル Deep Learning 応用講座 2024 Fall](https://weblab.t.u-tokyo.ac.jp/education/large-language-model/)

### LLMの学習手法

#### 事前学習

##### 事前学習データ

##### 事前学習データの前処理

LLaMAの学習に用いられたデータはWebからクロールしたデータで、CommonCrawlを初めとした4TB以上のサイズを持つ。また、GPT-3の事前学習トークン数は4100億, GPT-4は13兆トークンと言われる。

> [!NOTE] データの質を上げて量を絞ると、訓練にかかるコストも削減できるの？
> 調査中...

> [!NOTE] Webからクロールしたデータはゴミも多い。データの質に着目した量の指標はないの？
> 調査中...

データの前処理パイプラインは次の通り。ただしデータセットによって前処理の仕組みは異なる。

- Quality Filtering
- De-dup (重複排除)
- Privacy Reduction
- Tokenization

##### 事前学習の訓練

次のトークンの生成確率をひたすら予測する。数理的には、トークンの生成確率から交差エントロピーを算出し、そのミニバッチ内での平均をLossとする。

Next Token Predictionでは、一般的に1epochのみ学習させる。

<!-- TODO: [!NOTE] 誤差を測るにあたって、単語間の意味の近さも考慮するの？ -->

#### 継続事前学習

継続事前学習においても前処理のパイプラインがある。Swallow[^swallow_2023]コーパス![^swallow-corpus_2023]のパイプラインは次の通り。
![^swallow_2023](https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/A8-5.pdf)
![^swallow-corpus_2023](https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/A6-1.pdf)

1. 日本語のテキスト抽出
2. 品質フィルタリング (3兆文字 → 1兆文字)
   1. 文字数が400文字以上である
   2. 日本語の文字数が50%以上である, 他
3. 重複フィルタリング (1兆文字 → 3500億文字)
4. ホスト名フィルタリング (3500億文字 → 3100億文字)

##### 語彙拡張

<!-- TODO -->

#### スケール則

計算資源、データセットサイズ、パラメータ数を適切に引き上げることで、モデルの性能を向上させる（＝誤差を減らす）ことができる。これらの値には、次の関係が成り立つことが経験的に知られている。

$$
\begin{align}
L(X) = (\frac{X_c}{X})^\alpha
\end{align}
$$

ただし、$L(X)$は誤差、$X$は計算資源, データセットサイズ, パラメータ数のいずれか、$\alpha$は$L(X)$と$X$の両対数グラフの負の傾きを表す。

スケール則を用いることで、与えられた条件内で到達できる最小の誤差を予測することができる。また、スケール則は深層学習のタスクの種類（翻訳, 画像分類, 音声認識, etc...）を問わずに発現することが報告されている。[^Hestness_et_al_2017]
[^Hestness_et_al_2017]: J. Hestness et al., “Deep Learning Scaling is Predictable, Empirically,” arXiv.org. Accessed: Oct. 02, 2024. [Online]. Available: <https://arxiv.org/abs/1712.00409v1>

計算量の単位としてはFLOPs (Floating Point Operations, 浮動小数点演算性能)、またはPF-daysが用いられる。PF-daysとは、1Peta FLOPS(Floating points operation per second, 末尾Sが大文字であることに注意)の処理性能を持つサーバーを何日分計算に使ったかを示す量である。

また、スケール則を用いることで、計算資源に対して最適なモデルのパラメータ数とトークン数を求めることができる。Chinchilla論文では、パラメータ数:トークン数＝1:20の比通が良いとされる。

#### Fine-Tuning

LLMの訓練フローは、次の3ステップからなる。

1. Pre-Training (事前学習)
2. Supervised Fine-Tuning
3. Reinforcement Learning from Human Feedback (RLHF)

事前学習以降は広義のFine-Tuningなので、RLHFと区別する意味でSupervised Fine-Tuningと呼ばれる。

##### Instruction Tuning

Fine-Tuningの中でも、指示・回答の形式に統一したデータセットで言語モデルをFine−Tuningする手法をInstruction Tuningという。主にタスクへの適応を行っている一方で、新たに知識を獲得するのではなく事前学習で得た知識を引き出すことで改善を実現している、という説がある。

##### Parameter Efficient Fine-Tuning

大規模なモデルに対してFine-Tuningを行うと、莫大な計算リソースが必要になる。そこで、一部のパラメータや追加したパラメータのみを対象にしたパラメータ効率の良いFine-Tuningが考えられる。これをParameter Efficient Fine-Tuning (PEFT)という。

PEFTの代表的な手法としては、次の4つが存在する。

| Name                    | Description                                                      | Pros                                          | Cons                                              |
| ----------------------- | ---------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------- |
| Adapters                | Transformer内にモジュールを追加                                  | 訓練パラメータ数が小さい                      | 推論にオーバーヘッド                              |
| Soft Prompts            | モデルは変化させず、タスクのためのプロンプトをベクトル形式で学習 | モデル学習不要, 性能高い                      | 入力のContextを圧迫                               |
| Selective               | 各モジュールのバイアス項だけを学習                               | 学習データ数が小さい領域ではFull-FTより高精度 | 大規模モデルではFull-FTに対して精度が劣る         |
| Reparametrization-based | Full-FT後の重み - 現在の重みの差分のみを学習                     | タスク依存だがFull-FTと同等の精度             | タスクに寄ってはFull-FTと比較して著しい性能の劣化 |

![Lialin et al. (2023):Figure 2:Parameter-efficient fine-tuning methods taxonomy.](https://ar5iv.labs.arxiv.org/html/2303.15647/assets/img/peft_taxonomy_v3.2.jpg)

#### RLHF & Alignment

<!-- TODO -->

##### Human-in-the-Loop機械学習

広義には機械学習の訓練・運用のプロセスに人間が参加することを言うようだ。[^itmedia_2022]狭義には、継続的に能動学習を行うことで人間の知見を取り込むことを指すようにみえる。
[^itmedia_2022]: [ヒューマン・イン・ザ・ループ（HITL ：Human-in-the-Loop）とは？](https://atmarkit.itmedia.co.jp/ait/articles/2203/10/news019.html)

### LLMの推論

#### Prompting & 文脈内学習

LLMの応答の正誤は、指示文の影響を大きく受ける。代表的なPromptのテクニックは次の通り。

- CoT (Chain of Thought) (ステップバイステップで考える)
- Few-shot learning (例示する)

LLMを提供している企業のプロンプトは次の通り。

- [Claude Prompt Library](https://docs.anthropic.com/en/prompt-library/library)

#### RAG (Retrieval-Augmented Generation)

外部知識を利用したテキストの生成をRAGと呼ぶ。そのうち、関連する知識（文書）を取得する機能をRetrieverと呼ぶ。関連度合いの求め方によって、次のように分類される。

- Sparse Retriever
  - キーワード検索
  - TF-IDF
  - 埋め込みのコサイン類似度
- Neural Retriever (Dense Retriever)

また、初めにキーワード検索を行い、次にNeural Retrieverを用いるようなRetrieverも考えられる。これをRerankと呼ぶ。

検索した文書の使い方は次の通り。

1. コンテキストとして追加する (REPLUG)
2. 複数の予測のうち、得られた文書から見て尤もらしい予測を採用する (KNN-prompt)

#### Tool Augmented Language model

プログラミング言語実行環境や電卓などを利用する言語モデルをTool Augmented Language Modelといい、代表的なモデルに[Gorilla](https://gorilla.cs.berkeley.edu/)がある。

#### 推論時のスケーリング

同じモデルを使う場合でも、推論時に工夫をすることで性能を引き上げることができる。次の通り、様々なレベルの工夫が考えられる。

- Decodingによる工夫
  - Greedy Decoding
  - Beam Search
  - Random Sampling
- Promptingによる工夫
- メタ生成 (Meta-generation)アルゴリズムによる工夫
  - Parallel Search
  - Step level search
  - Refinement

### LLMの評価

ツールが公開されているほか、リーダーボードが公開されている。

- [Open LLM Leaderboard 2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) (Open LLMのリーダーボード, 1からタスクを刷新)
- [Nejumiリーダーボード](https://wandb.ai/wandb-japan/llm-leaderboard3/reports/Nejumi-LLM-3--Vmlldzo3OTg2NjM2) (日本語に特化)

## MLOps

### 能動学習

データの量が多い、専門性が高い等の理由からラベル付けのコストが高く付きそうな場合、学ぶデータに優先順位を付けるのが有向になる。アルゴリズムで新たに学習するデータを選ぶ手法を能動学習と呼ぶ。新たに学習するデータとしては、分類に迷うデータを選んだり、出現率が高いデータを選択する。

### 連合学習 (Federated Learning)

WIP
