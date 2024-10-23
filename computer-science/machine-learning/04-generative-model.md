# 生成モデル (generative model)

ページ構成の際、次の情報源を参考にした。

- [ゼロから作るDeep Learning ❺](https://amzn.to/46g6mSB)

## 概要

観測データの特徴変数$x$から目的変数$y$を推測するにあたって、$P(y|x)$の条件付き確率のみを用いるモデルを識別モデル(discriminative model)と呼ぶ。逆に、目的変数$y$が与えられた際に特徴変数が$x$である尤度$P(x|y)$と、$y$が観測できる事前確率$P(y)$を用いて、同時確率$P(x,y)$を最大化する$y$を求めるモデルを生成モデル(generative model)と呼ぶ。

## 生成モデルの学習手法

### 敵対的生成ネットワーク (GAN, Generative Adversarial Networks)

生成モデルの学習フレームワークの1つ。生成モデルでは、これまで存在しなかったような画像などを生成する場合、元画像が存在しない。そのため、ニューラルネットワークへのフィードバックにあたって、元画像と生成画像を比較して尤もらしさを算出することができない。

これに対しては、すでに訓練済みの画像分類モデル（例えばResNet）などを用いる方法が考えられる。しかし、生成したい分野の訓練済みモデルが都合良くあるとは限らない。また、分類モデルは画像が生成されたかどうかを見破ることに特化されていないため、生成された画像がある程度尤もらしくなると、それ以上の品質向上に貢献できないかもしれない。そこで、生成器(generator)と併せて識別器(discriminator)を訓練することを考える。これをGANという。

### Alignment

2024年現在、拡散モデルのAlignmentは、LLMに比べて研究されていない。拡散モデルにDPOを適用した研究があり、Diffusion-DPOという。[^Wallance_et_al_2023]
[^Wallance_et_al_2023]: B. Wallace et al., “Diffusion Model Alignment Using Direct Preference Optimization,” Nov. 21, 2023, arXiv: arXiv:2311.12908. doi: 10.48550/arXiv.2311.12908.

## 生成モデルのアーキテクチャ

### オートエンコーダ

次元削減や特徴抽出で便利なモデル。非線形の構造を持つデータにも有効な点が、PCAなどの従来の削減手法と異なる。1980年代にHintonらによって紹介されたとされる。[^Autoencoders_Unsupervised_Learning_and_Deep_Architectures]
（検索した限り、元論文には"autoencoder"という単語はないようだ。）[^Learning_internal_representations_by_error_propagation]

[^Autoencoders_Unsupervised_Learning_and_Deep_Architectures]: [Baldi, P. (2012, June). Autoencoders, unsupervised learning, and deep architectures. In Proceedings of ICML workshop on unsupervised and transfer learning (pp. 37-49). JMLR Workshop and Conference Proceedings.](http://proceedings.mlr.press/v27/baldi12a/baldi12a.pdf)
[^Learning_internal_representations_by_error_propagation]: [Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1985). Learning internal representations by error propagation. California Univ San Diego La Jolla Inst for Cognitive Science.](https://cs.uwaterloo.ca/~y328yu/classics/bp.pdf)

2000年代に研究が再燃し、画像のノイズ除去などに用いられるようになる。

### 変分オートエンコーダ

### 拡散モデル

<!-- TODO: 混合精度 -->

## 生成モデルの評価

ページ構成の際、次の情報源を参考にした。

- [ChatGPT🔐](https://chatgpt.com/c/0c69a86c-096a-4a84-a265-c6df17de88cb)

### BLEU

BLEU (Bilingual Evaluation Understudy, 発音はBlueと同じ)[^papineni_2002]は、機械翻訳等の評価に広く利用される評価指標。

[^papineni_2002]: [BLUE: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.pdf)

### ROUGE

ROUGE (Recall-Oriented Understudy for Gisting Evaluation, ルージュ)[^lin_2004]は、要約タスクで用いられる評価指標。参照する要約と生成した要約の一致度を測ることを試みる。

[^lin_2004]: [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)

最も基本的なROUGE-Nでは、N-gramの単位で、人手の要約と機械の要約との共起を測る。N-gramがUnigramの場合、ROUGE-1と呼ばれる。[^icoxfog417_2017]
[^icoxfog417_2017]: [ROUGEを訪ねて三千里:より良い要約の評価を求めて](https://qiita.com/icoxfog417/items/65faecbbe27d3c53d212)

ここでは、ROUGE-1で次の要約を評価する。

- 参照する要約: The artist chose a deep rouge for the lips
- 生成した要約: The lips were painted with a deep shade of red lipstick

参照する要約の単語をどれだけ当てられたかがRecallとなるため、$3/10 = 0.3$。また、生成した要約の単語がどれだけ参照元出身かがPrecisionなので、$3/8=0.375$。
したがってF1スコアを求めると、$2*(0.3*0.375)/(0.3+0.375)=0.333...$となる。
