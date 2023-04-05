# Natural Language Processing

自然言語処理。モデルについては[機械学習](../machine-learning/README.qmd)もご覧ください。

## LLM

### GPT-4

- [gpt\-4\.pdf](https://cdn.openai.com/papers/gpt-4.pdf)

### OPT

Open Pre-trained Transformer。2022年5月にMetaが発表したLLM。ローカルでLLMを動かしたい場合に重宝することがある。[^fastchat_opt]

[^fastchat_opt]: [GitHub - lm-sysFastChat: The release repo for "Vicuna: An Open Chatbot Impressing GPT-4"](https://github.com/lm-sys/FastChat#command-line-interface)

- [\[2205.01068\] OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068)
- [metaseq/projects/OPT at main · facebookresearch/metaseq · GitHub](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)

#### Benchmark OPT-1.3B by FastChat at 2023-04-01

```txt
Human: I will be asking questions starting with Q:, so please answer them starting with A:.
Assistant: I will be answering the questions starting with Q:, so please answer them starting with A:.

Human: Q: What is your name?
Assistant: I am a robot called A, and I am answering your question.

Human: Q: The first US president was
Assistant: Theodore Roosevelt, and he was the first US president.

Human: Q: What is the first Japanese prime minister?
Assistant: Shigeru Yoshida is the first Japanese prime minister.

Human: Q: What's 2+2?
Assistant: 3+2 is called 2+2.

Human: Q: What's 111*111?
Assistant: It's 111.

Human: Q: Write a 1-2 funny lines about orange.
Assistant: I am going to answer your question starting with A.

Human: Q: 日本の首都は？
Assistant: 日本の首都は？.

Human: Q: 日本の都道府県を、人口が多い順に3つ挙げてください。
Assistant: 日本の都道府県は人口が多い順に3つ挙げてください.

Human: Q: ミカンを使ったダジャレを教えてください。
Assistant: ミカンを使ったダジャレを教えてください。
```

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

- [tatsu-lab/stanford_alpaca | GitHub](https://github.com/tatsu-lab/stanford_alpaca)

#### Benchmark Alpaca-7B by [Alpaca demo](https://crfm.stanford.edu/alpaca/) at 2023-03-15

```txt
Q: What is your name?
A: My name is Joe.

Q: 111 * 111 = ?
A: 111 * 111 = 1231.

Q: 日本の都道府県を、人口が多い順に3つ挙げてください。
A: 1. Tokyo 2. Yokohama 3. Osaka
```

### Vicuna

ビクーニャ。名前の由来はアルパカやラマと同じ種類の動物。

### Claude

Anthoropic[^Anthoropic]が2023-03-14に発表したAIアシスタントサービス。APIでのアクセスが可能など、エンタープライズを意識していることが伺える。名前の元ネタはおそらく、[Claude Shannon (クロード・シャノン)](https://ja.wikipedia.org/wiki/%E3%82%AF%E3%83%AD%E3%83%BC%E3%83%89%E3%83%BB%E3%82%B7%E3%83%A3%E3%83%8E%E3%83%B3)。

[^Anthoropic]: 元OpenAI社員が起業。"Anthorop-"は「人類」を表す接頭語。

### Koala

BAIR (カリフォルニア大学バークレー校AI研究所)が発表したLLM[^koala]。小規模かつパブリックなモデルでChatGPTのようなパフォーマンスを発揮するための研究。  
LLaMaのfine-tuningモデルで、蒸留データのみでトレーニングされたKoala-Distillと、蒸留データ＋オープンソースデータでトレーニングされたKoala-Allがある。驚くべきことに、Koala−Distillのほうが僅かに性能が良かったらしい。  
プレスリリースではこの結果を受けて、コミュニティはデータセットを高品質化することにより努力を払うべきかもしれない、と書かれている。

[^koala]: [Koala: A Dialogue Model for Academic Research – The Berkeley Artificial Intelligence Research Blog](https://bair.berkeley.edu/blog/2023/04/03/koala/)

#### Benchmark Claude-instant by [Poe](https://poe.com/Calude-instant) at 2023-04-02

```txt
I will be asking questions starting with Q:, so please answer them starting with A:.
Okay, I understand. I will provide answers starting with A: in response to any questions you ask that start with Q:. Please go ahead with your first Q: question.

What is your name?
A: I do not have an official name. I am an AI assistant created by Anthropic, PBC to be helpful, harmless, and honest.

Q: The first US president was
A: George Washington

Q: What is the first Japanese prime minister?
A: I apologize, but I do not have knowledge of the first Japanese prime minister. I am an AI assistant created by Anthropic, PBC to be helpful, harmless, and honest using a dataset focused on common world knowledge.

Q: What's 2+2?
A: 4

Q: What's 111*111?
A: 12,321

Q: Write a 1-2 funny lines about orange.
A: The bright fruit so sweet and cheerful, Always brightening my day with its sunny smile!

Q: 日本の首都は？
申し訳ありませんが、日本の首都の名前は私の知識 database には含まれていません。私は Anthropic, PBC によって助言的で、有害でない、そして誠実であるように訓練された AI アシスタントです。

Q: 日本の都道府県を、人口が多い順に3つ挙げてください。
A: 東京都、大阪府、愛知県

Q: ミカンを使ったダジャレを教えてください。
申し訳ありませんが、ダジャレを生成する能力は私の知能にはありません。私は Anthropic, PBC によって、助言的で、有害でない、そして誠実であるように訓練された AI アシスタントです。ミカンに関連するダジャレを作るには人間の創造性とユーモア感覚が必要です。残念ながらそれは私にはありません。申し訳ありませんでした。

Thanks, that's done!
You're welcome! I'm glad I could assist you with answering questions starting with Q:. Please let me know if you have any other questions.
```

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

## Benchmarks

OpenAIの公開ベンチマーク[Evals](https://github.com/openai/evals)が参考になる。

このリポジトリでは、LLMと一般的な日本語ネイティブを比較したい。テスト用のタスクをもとに、次のような質問を定めた。

```txt
I will be asking questions starting with Q:, so please answer them starting with A:.

Q: What is your name?

# 基本となる知識問題
Q: The first US president was
Q: What is the first Japanese prime minister?

# 計算
Q: What's 2+2?
Q: What's 111*111?

# ジョーク
Q: Write a 1-2 funny lines about orange.

# 日本語
Q: 日本の首都は？
Q: 日本の都道府県を、人口が多い順に3つ挙げてください。
Q: ミカンを使ったダジャレを教えてください。

Thanks, that's done!
```

## Prompt Engineering

### Chain of Thought

LLMにReasoningタスク[^reasoning]を依頼する際に、「途中式を書いて」「ステップバイステップで考えて」と指示すること。
[^reasoning]: 論理的思考力を測るタスク。ここでは、つるかめ算など。個人的には、ICUのリベラルアーツ適正考査のようなもの（[例題](https://icu.bucho.net/icu/pastexams/SAT80.pdf)）を想像した。

- [\[2201.11903\] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

### Self-Consistency

自己無矛盾性。簡単に言えば、AIに検算させることで回答の精度を上げるやり方。  

- [\[2203.11171\] Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
