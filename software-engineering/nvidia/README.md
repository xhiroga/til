# TensorRT

ResNet50モデルを使用したTensorRTの推論高速化デモの結果です。
PyTorchでの推論、ONNXエクスポート、TensorRTエンジン（FP32およびFP16）のビルドと推論を行い、パフォーマンスを比較しました。

コードは[sandbox](./_src/sandbox/resnet50_trt_demo.py)を参照してください。
デモスクリプトは、`software-engineering/nvidia/tensorrt/_src/sandbox` ディレクトリ内で実行されました。

## パフォーマンス結果 (バッチサイズ 8 での平均推論時間)

使用されたGPU: NVIDIA GeForce RTX 4090

### PyTorch

実行コマンド (sandbox ディレクトリ内):
```console
$ uv run python resnet50_trt_demo.py
```
実行結果のサマリー:
```
PyTorch average inference time: 4.104 ms
```

### TensorRT (FP32)

実行コマンド (sandbox ディレクトリ内):
```console
$ uv run python resnet50_trt_demo.py 
```
実行結果のサマリー:
```
TensorRT FP32 average inference time: 1.656 ms
Speedup vs PyTorch: 2.48x
```

### TensorRT (FP16)

実行コマンド (sandbox ディレクトリ内):
```console
$ uv run python resnet50_trt_demo.py
```
実行結果のサマリー:
```
TensorRT FP16 average inference time: 1.118 ms
Speedup vs PyTorch: 3.67x
Speedup vs TensorRT FP32: 1.48x
```

## 全体サマリー

```
--- Summary ---
PyTorch: 4.104 ms
TensorRT FP32: 1.656 ms (Speedup vs PyTorch: 2.48x)
TensorRT FP16: 1.118 ms (Speedup vs PyTorch: 3.67x)
TensorRT FP16 Speedup vs TensorRT FP32: 1.48x
```
