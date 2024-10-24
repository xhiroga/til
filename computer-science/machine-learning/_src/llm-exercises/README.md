# LLM exercises

日本語におけるLLMの5択クイズと長文要約の性能を引き上げることを目的とした演習。

## Getting Started

```shell
wsl
uv sync
```

## 戦略

- 試行回数を稼ぐため、軽いモデルを利用する。[^hatakeyama_2024_08_30]
- ライセンスに気を配りつつ、可能な限り在野のデータセットを用いる。
- 限られたGPUで実行する場合にも性能が引き出せるよう、高品質なデータセットを自動的に選別できると良い。
- PEFTの中でもSoft-Promptはぜひ試したい。
- メモリとVRAMの使用量を常に確認する[^nishio_2023]

## 技術的詳細

Fine Tuningの各ステージ[^Parthasarathy_et_al_2024]において気をつけるべき技術的詳細をまとめる。

### データ準備

#### 公開データ

- <https://tyc.rei-yumesaki.net/material/kaiwa-ai/>

#### データ生成

- [KanHatakeyama/synthetic-texts-by-llm](https://github.com/KanHatakeyama/synthetic-texts-by-llm)

### モデル準備

### 訓練

### Fine-tuning

### 評価・検証

- eval (OpenAI)

### デプロイ

### 監視

---

[^nishio_2023]: [松尾研LLMサマースクールのコンペ](https://scrapbox.io/nishio/松尾研LLMサマースクールのコンペ)
[^hatakeyama_2024_08_30]: [大規模言語モデルTanuki-8B, 8x8Bの位置づけや開発指針など](https://zenn.dev/matsuolab/articles/377f7ae8b1169e)
[^Parthasarathy_et_al_2024]: [The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs](https://arxiv.org/html/2408.13296v1)
