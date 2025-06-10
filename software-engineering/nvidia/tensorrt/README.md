# TensorRT

ResNet50モデルを使用したTensorRTの推論高速化デモの結果です。
PyTorchでの推論、ONNXエクスポート、TensorRTエンジン（FP32およびFP16）のビルドと推論を行い、パフォーマンスを比較しました。

コードは[sandbox/_src/sandbox/resnet50_trt_demo.py](_src/sandbox/resnet50_trt_demo.py)を参照してください。
デモスクリプトおよび生成されるモデルファイル(`.onnx`, `.engine`)は、`software-engineering/nvidia/tensorrt/_src/sandbox/` ディレクトリおよびその中の `models/` サブディレクトリで扱われます。

## Performance

GPU: NVIDIA GeForce RTX 4090

```console
$ uv run python resnet50_trt_demo.py
...
Benchmarking PyTorch Compile (100 runs, batch_size=8)...
PyTorch Compile average inference time: 3.309 ms

Benchmarking TensorRT FP32 (100 runs, batch_size=8)...
TensorRT FP32 average inference time: 3.146 ms

Benchmarking TensorRT FP16 (100 runs, batch_size=8)...
TensorRT FP16 average inference time: 2.325 ms

--- Performance Summary ---
Input Batch Size: 8
PyTorch: 5.444 ms
PyTorch Compile: 3.309 ms (Speedup: 1.65x)
TensorRT FP32: 3.146 ms (Speedup: 1.73x)
TensorRT FP16: 2.325 ms (Speedup: 2.34x)
```
