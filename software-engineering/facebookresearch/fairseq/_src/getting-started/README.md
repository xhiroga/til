# Getting Started | fairseq

## 実行する

**注意: 未完成。エラーが発生します**

```sh
mkdir -p data-bin
uv run --directory data-bin bash ../.venv/lib/python3.10/site-packages/fairseq/examples/translation/prepare-iwslt14.sh
uv run fairseq-train data-bin/iwslt14.tokenized.de-en \
  --arch tutorial_simple_lstm \
  --encoder-dropout 0.2 --decoder-dropout 0.2 \
  --optimizer adam --lr 0.005 --lr-shrink 0.5 \
  --max-tokens 12000 \
  --user-dir src/getting_started \
  --source-lang de \
  --target-lang en
```

## fairseqとは？

PyTorchを利用した、翻訳などのシーケンス処理のためのフレームワーク。読みは「フェアーシーク」が近そう（参考: [YouTube](https://www.youtube.com/watch?v=t6JjlNVuBUQ)）。

パッケージとしてimportすることもできるが、Cloneして拡張する方がメジャーに見えます。公式ドキュメントもそうですし、論文の公式実装リポジトリ内にfairseqがコミットされていることもあります。

2025年現在、別のリポジトリで [fairseq2](https://github.com/facebookresearch/fairseq2) が提供されているため、fairseq のメンテナンスはほぼ終了しています（ただし、スター数では**31k vs 1k**とfairseqが圧倒しています）

## メンタルモデル

PyTorchのラッパーです。YAMLなどでタスク・モデル・データをまとめて定義し、関数に渡すだけで訓練できます。ループや誤差逆伝播を自分で書かないで良いのはもちろん、データ前処理や貪欲デコーディングなどの前後の処理・勾配蓄積などの学習テクニックも実装してあります。ただ、翻訳関連タスクに特化している印象はあります。

タスクやモデルはfairseqの基底型を継承したクラスとして定義する必要があります。また、それらを動的にロードするために、キーと紐付けるためのレジストリという概念があります。これは単にfairseqのパッケージ内に存在するシングルトンです。次の通り、登録したクラスをレジストリで見ることができます。

```console
% uv run python -c "from fairseq.models import MODEL_REGISTRY; print(sorted(MODEL_REGISTRY.keys()))"
2025-10-12 17:59:56 | INFO | fairseq.tasks.text_to_speech | Please install tensorboardX: pip install tensorboardX
['bart', 'camembert', 'cmlm_transformer', 'convtransformer', 'dummy_model', 'fastspeech2', 'fconv', 'fconv_lm', 'fconv_self_att', 'gottbert', 'hf_gpt2', 'hubert', 'hubert_ctc', 'insertion_transformer', 'iterative_nonautoregressive_transformer', 'levenshtein_transformer', 'lightconv', 'lightconv_lm', 'lstm', 'lstm_lm', 'masked_lm', 'model_parallel_roberta', 'model_parallel_transformer', 'model_parallel_transformer_lm', 'multilingual_transformer', 'nacrf_transformer', 'nonautoregressive_transformer', 'pipeline_parallel_transformer', 'roberta', 'roberta_enc_dec', 's2spect_transformer', 's2t_berard', 's2t_conformer', 's2t_transformer', 's2ut_conformer', 's2ut_transformer', 'tacotron_2', 'transformer', 'transformer_align', 'transformer_from_pretrained_xlm', 'transformer_lm', 'transformer_ulm', 'tts_transformer', 'wav2vec', 'wav2vec2', 'wav2vec_ctc', 'wav2vec_seq2seq', 'xlmr', 'xm_transformer', 'xmod']

% uv run python -c "import getting_started; from fairseq.models import MODEL_REGISTRY; print(sorted(MODEL_REGISTRY.keys()))" 
2025-10-12 18:00:00 | INFO | fairseq.tasks.text_to_speech | Please install tensorboardX: pip install tensorboardX
['bart', 'camembert', 'cmlm_transformer', 'convtransformer', 'dummy_model', 'fastspeech2', 'fconv', 'fconv_lm', 'fconv_self_att', 'gottbert', 'hf_gpt2', 'hubert', 'hubert_ctc', 'insertion_transformer', 'iterative_nonautoregressive_transformer', 'levenshtein_transformer', 'lightconv', 'lightconv_lm', 'lstm', 'lstm_lm', 'masked_lm', 'model_parallel_roberta', 'model_parallel_transformer', 'model_parallel_transformer_lm', 'multilingual_transformer', 'nacrf_transformer', 'nonautoregressive_transformer', 'pipeline_parallel_transformer', 'roberta', 'roberta_enc_dec', 's2spect_transformer', 's2t_berard', 's2t_conformer', 's2t_transformer', 's2ut_conformer', 's2ut_transformer', 'simple_lstm', 'tacotron_2', 'transformer', 'transformer_align', 'transformer_from_pretrained_xlm', 'transformer_lm', 'transformer_ulm', 'tts_transformer', 'wav2vec', 'wav2vec2', 'wav2vec_ctc', 'wav2vec_seq2seq', 'xlmr', 'xm_transformer', 'xmod']
```

特に重要な概念はタスク・モデル・アーキテクチャです。ここでタスクは訓練・推論する処理の種類を指し、毎回の実行はジョブと呼ばれます。モデルは`torch.nn`の子孫クラスのラッパーと雑に考えて良いです。アーキテクチャはモデル＋プリセットの設定です。単にモデルを継承したモデルとして定義すれば良さそうなものですが、そうしなかったようです。

## ありがちなエラー

fairseqは古いパッケージなので、最新の環境だとエラーが発生します。

### Python3.11以降のDataclassの仕様との競合

```console
...
  File "/home/hiroga/.local/share/uv/python/cpython-3.12.7-linux-x86_64-gnu/lib/python3.12/dataclasses.py", line 852, in _get_field
    raise ValueError(f'mutable default {type(f.default)} for field '
ValueError: mutable default <class 'fairseq.dataclass.configs.CommonConfig'> for field common is not allowed: use default_factory
```

対処法は2通りあります。

1. `FairseqConfig`を修正する（参考: [xhiroga/zero-avsr](https://github.com/xhiroga/zero-avsr/blob/7609cf42c99c74a231a9c93615f42e1a2af547ff/fairseq/fairseq/dataclass/configs.py#L973)）
2. Python 3.10以前を使う

### PyTorch2.6以降のモデルロード時のデフォルト引数の変更との競合

```console
ValueError: mutable default <class 'fairseq.dataclass.configs.CommonConfig'> for field common is not allowed: use default_factory
```
対処法は3通りあります。

1. `fairseq`側の`checkpoint_utils.py`を変更する（参考: [xhiroga/zero-avsr](https://github.com/xhiroga/zero-avsr/blob/7609cf42c99c74a231a9c93615f42e1a2af547ff/fairseq/fairseq/checkpoint_utils.py#L305)）
2. PyTorch2.5以前を使う
3. 次のとおりワークアラウンドを実行する（これ直接関係あるのか...？）

```py
add_safe_globals([data.dictionary.Dictionary])
```

### fairseq-train: error: argument --arch/-a: invalid choice:

パッケージの`fairseq`のCLIが、自作モデルを認識していないことがあります。その場合、`--user-dir`オプションで渡したディレクトリがパッケージとして解釈され、`import`が実行されます。
