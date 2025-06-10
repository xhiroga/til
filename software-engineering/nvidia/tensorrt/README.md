# TensorRT

NVIDIAが提供する機械学習モデルの推論バックエンド。PyTorchとは統合され、コンパイルバックエンドの1つとして呼び出すことができる。

## Performance

実装は[_src/sandbox](_src/sandbox)を参照。

GPU: NVIDIA GeForce RTX 4090

```console
$ uv run python resnet50_trt_demo.py
...
--- Performance Summary ---
Input Batch Size: 8
PyTorch: 4.611 ms
PyTorch Compile (TensorRT): 1.652 ms (Speedup: 2.79x)
PyTorch Compile (Inductor): 3.499 ms (Speedup: 1.32x)
TensorRT FP32: 3.024 ms (Speedup: 1.52x)
TensorRT FP16: 2.444 ms (Speedup: 1.89x)
```
