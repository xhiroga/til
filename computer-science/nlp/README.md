# Natural Language Processing

自然言語処理。モデルについては[機械学習](../machine-learning/README.qmd)もご覧ください。

## Transformer

### Self-Attention

### Encoder/Decoder

## LLM

### GPT-4

### Facebook LLaMa

2023年月にMeta AI Researchが発表したLLM。  
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

## Prompt Engineering

### Chain of Thought

LLMにReasoningタスク[^reasoning]を依頼する際に、「途中式を書いて」「ステップバイステップで考えて」と指示すること。
[^reasoning]: 論理的思考力を測るタスク。ここでは、つるかめ算など。個人的には、ICUのリベラルアーツ適正考査のようなもの（[例題](https://icu.bucho.net/icu/pastexams/SAT80.pdf)）を想像した。

- [\[2201.11903\] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
