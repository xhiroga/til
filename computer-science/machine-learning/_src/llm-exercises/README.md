---
notebook_urls:
- SFTTrainer: https://aistudio.google.com/prompts/1PgguWNLZA0C1AYMrf5UeR0_epKkVKWkd
- Gemma2: https://aistudio.google.com/prompts/1Tja4U0X4Ix9JEO5G1a_4zJq0tcORwomw
- Nexusflow: https://aistudio.google.com/prompts/1hhNmI7Ze7slNs3DqHS4bG-VA1CuHTzZ9
---

# LLM exercises

LLMの日本語による複雑な指示・タスクを行う能力を引き上げるプロジェクト。松尾研LLM講義 2024 Fall の最終課題コンペティション（以下、コンペ）に参加している。

リソース、コードのリストアップにあたって、次の情報源を参照した。

- [awesome-japanese-llm](https://github.com/llm-jp/awesome-japanese-llm)

## Getting Started

```shell
wsl
uv sync
uv run python src/sft.py
```

## 戦略

- [x] 試行回数を稼ぐため、軽いモデルを利用する。[^hatakeyama_2024_08_30]
- [x] 訓練から自動評価までのパイプラインを構築する
- ライセンスに気を配りつつ、可能な限り在野のデータセットを用いる。
- 限られたGPUで実行する場合にも性能が引き出せるよう、高品質なデータセットを自動的に選別できると良い。
- PEFTの中でもSoft-Promptはぜひ試したい。
- メモリとVRAMの使用量を常に確認する[^nishio_2023]
- [ ] 設定のYAML化を検討
- [ ] unsloth の有無の比較
- [ ] QLoRA と LoRA の比較
- [ ] min_p sampling の検証

[^hatakeyama_2024_08_30]: [大規模言語モデルTanuki-8B, 8x8Bの位置づけや開発指針など](https://zenn.dev/matsuolab/articles/377f7ae8b1169e)
[^nishio_2023]: [松尾研LLMサマースクールのコンペ](https://scrapbox.io/nishio/松尾研LLMサマースクールのコンペ)

## リソース

本プロジェクトは成果物が公開可能であることを念頭に置くと同時に、コンペのルールに準拠する。

### LLM

スコアは[Nejumi LLMリーダーボード3](https://wandb.ai/wandb-japan/llm-leaderboard3/reports/Nejumi-LLM-3--Vmlldzo3OTg2NjM2)におけるTotal AVGを指す。

- [LLM-jp-3](https://huggingface.co/collections/llm-jp/llm-jp-3-pre-trained-models-672c6096472b65839d76a1fa)
  - パラメータ数: 1.8b, 3.7b, 13b, 172b
  - コンペ利用: 推論, モデルマージ, 蒸留, 合成データの生成
- [Gemma2](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)
  - パラメータ数: 2b, 9b, 27b (Transformers実装とPyTorch実装の2種類が存在)
  - [ライセンス](https://ai.google.dev/gemma/terms): 商用利用可能。派生する配布物は"Notice"を含む必要がある。
    - したがって、同じく継承の義務があり、かつ競合するライセンスのモデル・データセットを組み合わせた派生物の再配布ができない（CC-SAなど）
  - コンペ利用: 推論, モデルマージ, 蒸留, 合成データの生成
- [Qwen/Qwen2.5](https://huggingface.co/Qwen)
  - パラメータ数: 0.5b, 1.5b, 3b, 7b, 14b, 32b, 72b
  - [ライセンス](https://github.com/QwenLM/Qwen2.5?tab=readme-ov-file#license-agreement)
    - 3bと72bを除いてApache Apache 2.0で利用可能。
    - 3bは商用利用不可（アリババに問い合わせが必要）
    - 72bはQwenライセンスで利用可能。
  - スコア: 32bで0.7362。
  - コンペ利用: 合成データの生成のみ。
- mistralai/Mixtral
  - パラメータ数: 8x7b, 8x22b (Instructあり・なし)
  - ライセンス: Apache 2.0 license
- [Llama3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)
  - パラメータ数: 8b, 70b, 405b
  - [ライセンス](https://www.llama.com/llama3_1/license/): 
  - Llama3.2はマルチモーダルとオンデバイス対応がメインのため、ここでは3.1を取り上げた。

LLMのリストアップにあたり、次のリソースを参照した。

- [Nejumi LLMリーダーボード3](https://wandb.ai/wandb-japan/llm-leaderboard3/reports/Nejumi-LLM-3--Vmlldzo3OTg2NjM2)
- [オープン日本語LLMリーダーボード](https://huggingface.co/spaces/llm-jp/open-japanese-llm-leaderboard)

### データセット

- [kajuma/CC-news-2024-July-October-cleaned](https://huggingface.co/datasets/kajuma/CC-news-2024-July-October-cleaned)
  - Common Crawlのnewsサブセットから作成した2024年7月から10月の日本語のニュースの文章
- [ichikara-instruction: LLMのための日本語インストラクションデータ](https://liat-aip.sakura.ne.jp/wp/llmのための日本語インストラクションデータ作成/)[^Sekine_et_al_2024]
  - ライセンス: CC BY-NC-SA 4.0
- [DeL-TaiseiOzaki/Tengentoppa-sft-v1.0](https://huggingface.co/datasets/DeL-TaiseiOzaki/Tengentoppa-sft-v1.0)
  - instruction, input, output の形式に統一された複数のデータセット
  - ライセンス: オリジナルのデータセットに準拠
- [DeL-TaiseiOzaki/news_summary_2024secondhalf](https://huggingface.co/datasets/DeL-TaiseiOzaki/news_summary_2024secondhalf)
- <https://tyc.rei-yumesaki.net/material/kaiwa-ai/>

[^Sekine_et_al_2024]: 関根聡, 安藤まや, 後藤美知子, 鈴木久美, 河原大輔, 井之上直也, 乾健太郎. ichikara-instruction: LLMのための日本語インストラクションデータの構築. 言語処理学会第30回年次大会(2024)

データセットのリストアップにあたり、次のリソースを参照した。

- [LLM のデータセットまとめ](https://note.com/npaka/n/n686d987adfb1)
- [データセットのまとめ](https://zenn.dev/karaage0703/articles/5fcb217baded2e#llm)

## コード

Fine Tuningの各ステージ[^Parthasarathy_et_al_2024]において気をつけるべき技術的詳細をまとめる。
[^Parthasarathy_et_al_2024]: [The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs](https://arxiv.org/html/2408.13296v1)

### データ合成

- [WorksApplications/uzushio](https://github.com/WorksApplications/uzushio)
  - CommonCrawlのデータから日本語コーパスを抽出するためのライブラリ
- [KanHatakeyama/synthetic-texts-by-llm](https://github.com/KanHatakeyama/synthetic-texts-by-llm)

## 継続事前学習

### Fine-tuning

- [LLM-jp SFT](https://github.com/llm-jp/llm-jp-sft)
  - LLP-jpのSFTで用いられたスクリプト

### 推論

### 評価・検証

- eval (OpenAI)

### デプロイ

### 監視
