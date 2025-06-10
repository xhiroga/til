# Safetensors

Safetensors のフォーマットは 「8 byte のヘッダー長」+「UTF-8 JSON ヘッダー」+「生データ」 という極めてシンプルな仕様になっている。[HuggingFaceのドキュメント](https://huggingface.co/docs/safetensors/en/metadata_parsing)も参照。

## 実験

コードは[sandbox](./_src/sandbox)を参照。

```console
$ uv run python diy.py models/pixel-art-medium-128-v0.1.safetensors
header_len_bytes=b'\xb8^\x00\x00\x00\x00\x00\x00'
header_len=24248
header_json={'__metadata__': {'format': 'pt'}, 'transformer.transformer_blocks.0.attn.to_k.lora_A.weight': {'dtype': 'F32', 'shape': [4, 1536], 'data_offsets': [0, 24576]},... }
```
