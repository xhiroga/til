# bitsandbytes

大規模言語モデルのモデル削減を、CUDA関数をラップして呼ぶ形でコスパよく実現してくれるライブラリ。

## 実験

```console
% uv run memray run -o "logs/memray-$(date '+%Y%m%d-%H%M%S').bin" main.py
```

## トラブルシューティング

### CUDA SETUP: CUDA detection failed! Possible reasons

次のようなエラーが出る場合があります。

```
False

===================================BUG REPORT===================================
/home/hiroga/Documents/GitHub/til/software-engineering/bitsandbytes/_src/getting-started/.venv/lib/python3.13/site-packages/bitsandbytes/cuda_setup/main.py:167: UserWarning: Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes
```

bitsandbytes側に対応するCUDA関係の共通ライブラリを持っていないのが原因であることがあります。`site-packages/bitsandbytes`を見れば確実です。

```console
% ls .venv/lib/python3.13/site-packages/bitsandbytes | grep .so
libbitsandbytes_cpu.so
libbitsandbytes_cuda110.so
libbitsandbytes_cuda110_nocublaslt.so
libbitsandbytes_cuda111.so
libbitsandbytes_cuda111_nocublaslt.so
libbitsandbytes_cuda114.so
libbitsandbytes_cuda114_nocublaslt.so
libbitsandbytes_cuda115.so
libbitsandbytes_cuda115_nocublaslt.so
libbitsandbytes_cuda117.so
libbitsandbytes_cuda117_nocublaslt.so
libbitsandbytes_cuda118.so
libbitsandbytes_cuda118_nocublaslt.so
libbitsandbytes_cuda120.so
libbitsandbytes_cuda120_nocublaslt.so
libbitsandbytes_cuda121.so
libbitsandbytes_cuda121_nocublaslt.so
libbitsandbytes_cuda122.so
libbitsandbytes_cuda122_nocublaslt.so
libbitsandbytes_cuda123.so
libbitsandbytes_cuda123_nocublaslt.so
```

bitsandbytesのバージョンを上げるか下げるかすれば確認できます。ドキュメントからも確認できます。

- bitsandbytesのバージョンが`0.43.0`以上であれば[HuggingFaceのドキュメント](https://huggingface.co/docs/bitsandbytes/v0.48.1/en/installation)に書いてあります。
- bitsandbytesのバージョンが`0.42.0`以下の場合は、[PyPIのドキュメント](https://pypi.org/project/bitsandbytes/0.42.0/)に書いてあります。
