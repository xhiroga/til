# 機械学習 (machine learning)

## 基本概念

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
| パーセプトロン     | パラメトリック | 誤差関数 | 確率的勾配降下法   | 分類           |
| ロジスティック回帰 | パラメトリック | 尤度関数 | IRIS法             | 分類           |
| k平均法            |                | 二重歪み |                    | クラスタリング |
| k近傍法            |                |          |                    | 分類           |
| EMアルゴリズム     |                |          |                    |                |
| SVM                |                |          |                    |                |
| 決定木             |                |          |                    |                |

### 最小二乗法

最小二乗法において、重みを解析的に導くための式展開は次の通り。誤差関数の$\frac{1}{2}$は微分した際の係数2を相殺するための定数。

![formula](https://i.gyazo.com/thumb/3024/4dcdeb0865b6c5bf6492619d3ca455b7-heic.jpg)

なお、方程式を重みについて解くために、偏微分によって現れるデザイン行列のグラム行列が逆行列を持つことを示す必要がある。デザイン行列が 列数 ≦ 行数 のとき（つまり特徴数よりバッチサイズが大きいとき）、グラム行列は正定値であり、逆行列を持つ。

![formula](https://i.gyazo.com/thumb/2491/de921cb60b69d0a58d0e4a6eee126711-heic.jpg)

<!-- 正則化項, CS 2018-02 2 -->

### 最尤推定法

尤度関数は次の通り。なお、確率分布として正規分布を採用している。

$P = \prod_{n=1}^{N} N(t_n|f(x_n), \sigma^2)$

<!-- https://i.gyazo.com/thumb/2420/c36546337d0c0125e4fefcbfdcad1d5d-heic.jpg -->

### パーセプトロン (perceptron)

目的関数の式は次の通り。ただし2値分類の問題とし、$f(x,y)$と0の大小関係で分類するものとする。また、誤って分類された点のみの合計とする。

$E=\sum_{n}|f(x_n, y_n)|$

また、最適化アルゴリズムである確率的勾配降下法の式は次の通り。

$W_new = W_old - \nabla E(w)$

<!-- TODO: ここでΦが出てくるように勾配ベクトルを0にするところ、自分で導出できない

なお、Φは慣例的に特徴ベクトルを表すのに用いられる。

> [!NOTE]
Φ(PHI) = FEAture ということだろうか？
-->

### ロジスティック回帰

ロジスティック回帰という名前だが分類のアルゴリズム。[ロジスティック回帰は回帰か分類か](https://scrapbox.io/nishio/ロジスティック回帰は回帰か分類か)も参照。

### k平均法

### k近傍法 (k-nearest neighbor algorithm)

教師あり学習の手法の一つ。ラベルが未知の入力データに対して、入力データと全データの間の距離を測定する。例えば、環境（気温・湿度・風速）から天気を予測する分類問題なら、気温・湿度・風速の3次元空間でのデータ間の距離を測定する。距離はユークリッド距離を使うことが多いが、原理的にはマンハッタン距離などでも構わない。

### 評価方法

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

## CAPs (Credit Assignment Paths)

### RNN

### LSTM

### オートエンコーダ

次元削減や特徴抽出で便利なモデル。非線形の構造を持つデータにも有効な点が、PCAなどの従来の削減手法と異なる。1980年代にHintonらによって紹介されたとされる。[^Autoencoders_Unsupervised_Learning_and_Deep_Architectures]
（検索した限り、元論文には"autoencoder"という単語はないようだ。）[^Learning_internal_representations_by_error_propagation]

[^Autoencoders_Unsupervised_Learning_and_Deep_Architectures]: [Baldi, P. (2012, June). Autoencoders, unsupervised learning, and deep architectures. In Proceedings of ICML workshop on unsupervised and transfer learning (pp. 37-49). JMLR Workshop and Conference Proceedings.](http://proceedings.mlr.press/v27/baldi12a/baldi12a.pdf)
: [Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1985). Learning internal representations by error propagation. California Univ San Diego La Jolla Inst for Cognitive Science.](https://cs.uwaterloo.ca/~y328yu/classics/bp.pdf)

2000年代に研究が再燃し、画像のノイズ除去などに用いられるようになる。

## Activation Function

活性化関数。非線形性（non-linearity）の1つ。

### Sigmoid function

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

### Softmax Function

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

### ReLU

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

### GELU

ガウス誤差線形ユニット（Gaussian Error Linear Unit）は、Transformer系のモデルでも採用される活性化関数。

## Gradient Descent

機械学習の訓練中に使用される最適化アルゴリズム[^optimizer]の一つ。

[^optimizer]: [【最適化手法】SGD・Momentum・AdaGrad・RMSProp・Adamを図と数式で理解しよう。](https://kunassy.com/oprimizer/)を参照。

訓練中の予測結果と実際の値の誤差を各パラメータに戻し、パラメータを更新することで、誤差が最小になるようにパラメータを更新していく。

### グラフニューラルネットワーク (GNN)

（要レビュー）ニューラルネットワークは一般的に、データを多次元変数として捉えた上で、変数の重み付きの和を新たな次元とすることで特徴量を自動で作る。CNNでは周辺のマスの重み付き和を、Transformerでは全範囲の重み付き和を用いる。これは、入力の範囲をグラフ構造で与えることで一般化できる。物体の各点が近い点からの相互作用を受けることに着目し、GNNを用いて自然なシミュレーションを行った応用などがある。[^joisino_2024]
[^joisino_2024]: [僕たちがグラフニューラルネットワークを学ぶ理由](https://speakerdeck.com/joisino/pu-tatigagurahuniyurarunetutowakuwoxue-buli-you)

## 大規模事前学習

### 転移学習

深層学習モデルが学習を通じてデータの表現とタスク固有の学習を行っていることに着目し、すでに学習済みのモデルの重みを用いて新たなタスクの学習を行うこと。

| 学習の分類           | 目的                 | 手段                    | 例                      |
| -------------------- | -------------------- | ----------------------- | ----------------------- |
| 事前学習             | より良い表現を得る   | 自己教師あり学習が多い? |                         |
| 継続事前学習         | より良い表現を得る   | 自己教師あり学習が多い? | 日本語コーパスでLLMなど |
| ファインチューニング | 個別のタスク特化する | 教師あり学習?           |                         |

<!-- 自己教師あり学習が「多い」か要確認 -->

事前学習が教師あり学習の場合は、汎用的な表現を得ることそのものが学習目的ではないため、例えば牧草が写っていたら牛と分類してしまうようなショートカット学習が行われることがあり、これがファインチューニングの妨げになる。

### 自己教師あり学習

ネットワーク構造の工夫より大量のデータを用意することでモデルの性能が向上すること、また転移学習により汎用的なモデルが固有タスクにおいても性能を発揮することを背景として、大量のラベル無しデータから学習する方法が模索された。

あらかじめ存在する事実のデータから、学習のためのデータを自分で作成する手法が自己教師あり学習である。例えば、テキストの一部をマスクしてその単語を当てるとか、画像を加工した上で、加工前後の画像を見比べて同じであれば高い報酬を与えるなどである。

具体的な手法については、Masked Language Modeling、対比学習を参照すること。なお、自己教師あり学習は、BERTの論文では教師なし事前学習とも呼ばれていた。

評価の方法としては、ラベル付き分類データを用いて埋め込みを取得し（全結合層で変換する手前の値）、k近傍法を用いるもの、シンプルに下流タスク用のヘッドを取り付けて性能を測るもの、下流タスクのための層を加えてフルパラメータのファインチューニングを行うものなどがある。

#### 対比学習（対照学習）

事前学習としての表現学習に用いられる手段の一つ。エンコーダとしての多層ニューラルネットワークと射影ヘッドからなるモデルに対して、ミニバッチで複数のデータ拡張された画像を与える。

対比学習の手法の1つ、SimCLRではInfoNCE損失関数([Desmos](https://www.desmos.com/calculator/nh1ntozu9o)[[🔐](https://www.desmos.com/calculator/mbn55ivvh6)])を用いる。
ミニバッチ内の同じ画像のデータ拡張から得たベクトルのコサイン類似度は近くなるように、そうでないベクトルのコサイン類似度は遠くなるように学習を進める。(正例のコサイン類似度/負例のコサイン類似度合計)にマイナスを付けて損失にするが、exponentialを取ってから自然対数を取り直す一工夫が入っている（正例・負例の部分を逆数にした方が、マイナスが取れて式がシンプルでは？）

### Transformer (2017)[^vaswani_2017]

[^vaswani_2017]: [A. Vaswani et al., “Attention is All you Need,” in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2017.](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

#### Self-Attention

#### Encoder/Decoder

### CLIP (2021)[^radford_2021]

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

#### CyCLIP (2022)[^goel_2022]

CLIPは正例と近い負例・遠い負例の距離に注意を払っていないため、画像をプロンプトで分類した場合と正解ラベル付き画像を用いてk-近傍法で分類した場合で結果に差が出ることがある。

次の考え方に基づいて損失を調整することで、精度を改善することができる。具体的にはそれぞれの距離の差の二乗を損失に加えている。

1. 画像jとキャプションkの距離感は、画像kとキャプションjの距離感と同じであるべき
2. 画像jと画像kの距離感は、キャプションjとキャプションkの距離感と同じであるべき

#### PAINT (2022)[^ilharco_2022]

[^ilharco_2022]: [Patching open-vocabulary models by interpolating weights](https://arxiv.org/abs/2208.05592)

ファインチューニングによって汎化性能を失う問題に対して、ファインチューン前後の重みを線形補間した重みを用いることを提案している。これによって汎化性能と固有タスクを解く能力をある程度良いところ取りできるらしい。感想だが、ファインチューニングだけでは過学習が起きてしまう、ということを示唆しているように思える。

### VL-Checklist (2023)[^zhao_2023]

[^zhao_2023]: [T. Zhao et al., “VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations.” arXiv, Jun. 22, 2023. doi: 10.48550/arXiv.2207.00221.](https://arxiv.org/abs/2207.00221)

画像言語モデルの特性を調べるためのベンチマーク。画像のキャプションに登場する名詞や形容詞を適当に入替えても、正しく理解できているなら入れ替え後のほうがスコアが少ないべき、という考え方に基づいたテストを行う。

<!-- ## BLIP-2 -->

<!-- ## LLaVA -->

## 手法

### 教師あり学習

### 教師なし学習

### 強化学習 (reinforcement learning)

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

#### 方策勾配法 (Vanilla Policy Gradient)

#### REINFORCE

#### PPO

方策ベースの手法。

#### Actor-Critic

価値ベースかつ方策ベースの手法。

#### 世界モデル (World Models)

価値ベースかつ方策ベースの手法。

## MLOps

### 能動学習

データの量が多い、専門性が高い等の理由からラベル付けのコストが高く付きそうな場合、学ぶデータに優先順位を付けるのが有向になる。アルゴリズムで新たに学習するデータを選ぶ手法を能動学習と呼ぶ。新たに学習するデータとしては、分類に迷うデータを選んだり、出現率が高いデータを選択する。

### Human-in-the-Loop機械学習

広義には機械学習の訓練・運用のプロセスに人間が参加することを言うようだ。[^itmedia_2022]狭義には、継続的に能動学習を行うことで人間の知見を取り込むことを指すようにみえる。
[^itmedia_2022]: [ヒューマン・イン・ザ・ループ（HITL ：Human-in-the-Loop）とは？](https://atmarkit.itmedia.co.jp/ait/articles/2203/10/news019.html)

### 連合学習 (Federated Learning)

WIP

### 評価

ページ構成の際、次の情報源を参考にした。

- [ChatGPT🔐](https://chatgpt.com/c/0c69a86c-096a-4a84-a265-c6df17de88cb)

#### BLEU[^papineni_2002]

[^papineni_2002]: [BLUE: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.pdf)

BLEU (Bilingual Evaluation Understudy, 発音はBlueと同じ)は、機械翻訳等の評価に広く利用される評価指標。

#### ROUGE[^lin_2004]

[^lin_2004]: [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)

ROUGE (Recall-Oriented Understudy for Gisting Evaluation, ルージュ)は、要約タスクで用いられる評価指標。参照する要約と生成した要約の一致度を測ることを試みる。

最も基本的なROUGE-Nでは、N-gramの単位で、人手の要約と機械の要約との共起を測る。N-gramがUnigramの場合、ROUGE-1と呼ばれる。[^icoxfog417_2017]
[^icoxfog417_2017]: [ROUGEを訪ねて三千里:より良い要約の評価を求めて](https://qiita.com/icoxfog417/items/65faecbbe27d3c53d212)

ここでは、ROUGE-1で次の要約を評価する。

- 参照する要約: The artist chose a deep rouge for the lips
- 生成した要約: The lips were painted with a deep shade of red lipstick

参照する要約の単語をどれだけ当てられたかがRecallとなるため、$3/10 = 0.3$。また、生成した要約の単語がどれだけ参照元出身かがPrecisionなので、$3/8=0.375$。
したがってF1スコアを求めると、$2*(0.3*0.375)/(0.3+0.375)=0.333...$となる。

## Uncategorized

- ランダムフォレスト: WIP
- カルマンフィルタ: WIP
- モンテカルロ法: WIP
- 主成分分析: WIP
- 隠れマルコフモデル: WIP
- ベイズの定理: WIP
- ベクトル量子化: WIP...次元削減との違いは？
- 分枝限定法: WIP
- MinMax法: WIP
- GRU
- 勾配消失
- 勾配爆発

## References

- [機械学習 (東京大学工学教程)](https://amzn.to/4bS8SjZ)
- [深層学習による画像認識の基礎](https://amzn.to/3wMy8sQ)
