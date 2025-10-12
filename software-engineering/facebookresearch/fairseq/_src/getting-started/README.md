# Getting Started | fairseq

## 実行する

```sh
uv run fairseq-train data-bin/iwslt14.tokenized.de-en \
  --arch tutorial_simple_lstm \
  --encoder-dropout 0.2 --decoder-dropout 0.2 \
  --optimizer adam --lr 0.005 --lr-shrink 0.5 \
  --max-tokens 12000 \
  --user-dir src/getting_started
```

## fairseqとは？

PyTorchを利用したシーケンス処理のためのフレームワーク。読みは「フェアーシーク」が近そう（[YouTube](https://www.youtube.com/watch?v=t6JjlNVuBUQ)）。

見た感じでは、パッケージとしての使用よりも、Cloneしてモデルを拡張し、さらにCLIから呼び出すことを想定しているようだ。

論文の公式実装などでは、Cloneしたリポジトリがそのままコミットされていることもある。

## メンタルモデル

(TODO...)

- タスク
- レジストリ
- マニフェスト
- モデル = register_modelと 
- アーキテクチャ = register_model_architecture（モデルの設定プリセット）

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
