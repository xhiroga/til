# PyTorch

## Compile

実験は[sandbox](_src/sandbox)を参照。

```console
$ make run
...
[06/18/2025-13:41:03] [TRT] [W] Functionality provided through tensorrt.plugin module is experimental.
Running ResNet50 benchmark (original vs compiled & exported)...
Original model average inference time: 23.474 ms
Loading exported compiled TensorRT model from models/trt.ep...
Compiled TensorRT model loaded successfully.
Compiled model average inference time: 17.440 ms
Speedup: 1.35x

$ ls models
trt.ep
```

## Profiler

実験は[sandbox](_src/sandbox)を参照。

```console
$ make profile
```

### References (Profiler)

- https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
