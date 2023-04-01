# Natural Language Processing

自然言語処理。モデルについては[機械学習](../machine-learning/README.qmd)もご覧ください。

## LLM

### GPT-4

- [gpt\-4\.pdf](https://cdn.openai.com/papers/gpt-4.pdf)

### OPT

Open Pre-trained Transformer。2022年5月にMetaが発表したLLM。ローカルでLLMを動かしたい場合に重宝することがありそうだ。

- [\[2205.01068\] OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068)
- [metaseq/projects/OPT at main · facebookresearch/metaseq · GitHub](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)

### ChatGPT

OpenAIが公開した、GPT-3.5をベースにRLHFでFine Tuningした会話特化のLLM。  
[API経由のデータはデフォルトでモデルの学習に利用されず、逆にChatGPT経由のデータはデフォルトでモデルの学習に利用される。](https://help.openai.com/en/articles/5722486-how-your-data-is-used-to-improve-model-performance)オプトアウトも可能。

- [ChatGPT](https://chat.openai.com/chat)
- [Introducing ChatGPT](https://openai.com/blog/chatgpt)

### Facebook LLaMa

2023年2月にMeta AI Researchが発表したLLM。  
公開から１週間後に完全なデータが4chanに流出した。一方で、M1Macで動作する[llama.cpp](https://github.com/ggerganov/llama.cpp)が登場し、以後ラズパイでも動くなど消費者向けハードウェアで実行可能になっている。

- [Introducing LLaMA: A foundational, 65\-billion\-parameter language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
- [facebookresearch/llama: Inference code for LLaMA models | GitHub](https://github.com/facebookresearch/llama)
- [\[2302.13971v1\] LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971v1)

### Stanford Alpaca

Stanford大学がLLaMA-7Bをベースにファインチューニングしたモデル。

ベンチマークは次のとおり。（2023-03-15時点）  
見ての通り、すべて誤っている。

```txt
Q: What is your name?
A: My name is Joe.

Q: 111 * 111 = ?
A: 111 * 111 = 1231.

Q: 日本の都道府県を、人口が多い順に3つ挙げてください。
A: 1. Tokyo 2. Yokohama 3. Osaka
```

- [tatsu-lab/stanford_alpaca | GitHub](https://github.com/tatsu-lab/stanford_alpaca)
- [Alpaca demo](https://crfm.stanford.edu/alpaca/)

### Vicuna

ビクーニャ。名前の由来はアルパカやラマと同じ種類の動物。

### Claude

Anthoropic[^Anthoropic]が2023-03-14に発表したAIアシスタントサービス。APIでのアクセスが可能など、エンタープライズを意識していることが伺える。名前の元ネタはおそらく、[Claude Shannon (クロード・シャノン)](https://ja.wikipedia.org/wiki/%E3%82%AF%E3%83%AD%E3%83%BC%E3%83%89%E3%83%BB%E3%82%B7%E3%83%A3%E3%83%8E%E3%83%B3)。

[^Anthoropic]: 元OpenAI社員が起業。"Anthorop-"は「人類」を表す接頭語。

## Training method

### RLHF

Reinforcement Learning from Human Feedback。Fine Tuningの手法の一つ。  
余談だが、論文中にはRLFHやReinforcement Learning from Human Feedbackといった表記は出てこない。

- [\[1909.08593\] Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)
- [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)

### LoRA

Microsoftが2021年に発表した論文『[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)』で提案された手法。  
学習対象のパラメータ数を一部に限定することで、GPUメモリやストレージを削減しつつFine Tuning並の精度を達成している。

- [【インターンレポート】6.7B日本語モデルに対するLoRAチューニング](https://engineering.linecorp.com/ja/blog/lora-tuning-for-japanese-model)

### RLAIF

Reinforcement Learning for AI Fairness。Anthoropicが2022年に発表した論文『[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)』で提案された手法。有用なモデルから有害性を取り除く際に、人間の代わりに憲法によって判断するAI（CAI, Constitutional AI）を活用する。

### PEFT

Parameter-Efficient Fine-Tuning。HuggingFaceが公開している、効率的に訓練を行うためのライブラリ。

- [huggingface/peft: 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning.](https://github.com/huggingface/peft)

## Benchmark

OpenAIの公開ベンチマーク[Evals](https://github.com/openai/evals)が参考になる。

このリポジトリでは、LLMと一般的な日本語ネイティブを比較したい。テスト用のタスクをもとに、次のような質問を定めた。

```txt
# 基本となる知識問題
Q: What is the first Japanese prime minister?

# 日本語による知識問題
Q: 日本の都道府県を、人口が多い順に3つ挙げてください。

# 計算
Q: What's 111*111?

# ジョーク
Q: ミカンを使ったダジャレを教えてください。


# その他
Q: What is your name?
```

## Prompt Engineering

### Chain of Thought

LLMにReasoningタスク[^reasoning]を依頼する際に、「途中式を書いて」「ステップバイステップで考えて」と指示すること。
[^reasoning]: 論理的思考力を測るタスク。ここでは、つるかめ算など。個人的には、ICUのリベラルアーツ適正考査のようなもの（[例題](https://icu.bucho.net/icu/pastexams/SAT80.pdf)）を想像した。

- [\[2201.11903\] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

### Self-Consistency

自己無矛盾性。簡単に言えば、AIに検算させることで回答の精度を上げるやり方。  

- [\[2203.11171\] Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
