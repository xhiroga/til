# [ShmuelRonen/ComfyUI-FramePackWrapper_Plus](https://github.com/ShmuelRonen/ComfyUI-FramePackWrapper_Plus)

## Parameters

- Load FramePackModel は 分割されたモデルを読み込むことができないので、事前にmergeしておく

## WIP! Error 137 (Out of memory) thrown on RTX-4090

```log
Final latent frames: 46 (Expected based on generation: 46)
Requested to load AutoencoderKL
0 models unloaded.
loaded partially 128.0 127.99981689453125 0
make: *** [Makefile:8: run] Error 137
```
