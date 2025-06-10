# TensorRT

ResNet50モデルを使用したTensorRTの推論高速化デモの結果です。
PyTorchでの推論、ONNXエクスポート、TensorRTエンジン（FP32およびFP16）のビルドと推論を行い、パフォーマンスを比較しました。

コードは[sandbox/_src/sandbox/resnet50_trt_demo.py](_src/sandbox/resnet50_trt_demo.py)を参照してください。
デモスクリプトおよび生成されるモデルファイル(`.onnx`, `.engine`)は、`software-engineering/nvidia/tensorrt/_src/sandbox/` ディレクトリおよびその中の `models/` サブディレクトリで扱われます。

## パフォーマンス結果 (バッチサイズ 8 での平均推論時間)

使用されたGPU: NVIDIA GeForce RTX 4090

### PyTorch

実行コマンド (`_src/sandbox` ディレクトリ内):
```console
$ uv run python resnet50_trt_demo.py
```
実行結果のサマリー:
```
PyTorch average inference time: 4.108 ms
```

### TensorRT (FP32)

実行コマンド (`_src/sandbox` ディレクトリ内):
```console
$ uv run python resnet50_trt_demo.py 
```
実行結果のサマリー:
```
TensorRT FP32 average inference time: 2.667 ms
Speedup vs PyTorch: 1.54x
```

### TensorRT (FP16)

実行コマンド (`_src/sandbox` ディレクトリ内):
```console
$ uv run python resnet50_trt_demo.py
```
実行結果のサマリー:
```
TensorRT FP16 average inference time: 2.493 ms
Speedup vs PyTorch: 1.65x
Speedup vs TensorRT FP32: 1.07x
```

## 全体サマリー (リファクタリング後スクリプトによる)

```
--- Performance Summary ---
Input Batch Size: 8
PyTorch: 4.108 ms
TensorRT FP32: 2.667 ms (Speedup vs PyTorch: 1.54x)
TensorRT FP16: 2.493 ms (Speedup vs PyTorch: 1.65x)
  (Speedup FP16 vs FP32: 1.07x)
```
