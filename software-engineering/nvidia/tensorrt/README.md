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
PyTorch: 5.729 ms
PyTorch Compile: 3.409 ms (Speedup: 1.68x)
PyTorch Compile (Inductor): 3.742 ms (Speedup: 1.53x)
TensorRT FP32: 3.349 ms (Speedup: 1.71x)
TensorRT FP16: 2.340 ms (Speedup: 2.45x)
```
