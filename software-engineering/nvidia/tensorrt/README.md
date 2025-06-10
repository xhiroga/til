# TensorRT

NVIDIAが提供する機械学習モデルの推論バックエンド。PyTorchとは統合され、コンパイルバックエンドの1つとして呼び出すことができる。

## Performance

コードは[sandbox/_src/sandbox/resnet50_trt_demo.py](_src/sandbox/resnet50_trt_demo.py)を参照。

GPU: NVIDIA GeForce RTX 4090

```console
$ uv run python resnet50_trt_demo.py
...
--- Performance Summary ---
Input Batch Size: 8
PyTorch: 4.205 ms
PyTorch Compile: 1.696 ms (Speedup: 2.48x)
PyTorch Compile (Inductor): 3.841 ms (Speedup: 1.09x)
TensorRT FP32: 2.962 ms (Speedup: 1.42x)
TensorRT FP16: 2.056 ms (Speedup: 2.05x)
```
