# [nvitop](https://github.com/XuehaiPan/nvitop)

インタラクティブなGPUのプロセス監視ツール。

```shell
nvitop -1 # --once, つまりインタラクティブなResource Monitorを表示しない
nvitop -C # 純粋なComputeプロセスのみを表示する
```

## Metrics

- CPU
- GPU-MEM: GPUメモリの使用率。なお、WindowsにおいてはOSがGPUメモリの割当を定めるため、プロセスごとの利用率を参照できない。[^nvidia_gpu_memory_usage]
- HOST-MEM
- GPU-SM: GPUのStreaming Multiprocessorの利用率

[^nvidia_gpu_memory_usage]: <https://forums.developer.nvidia.com/t/gpu-memory-usage-shows-n-a/169140>

## Note

- WindowsのプロセスをWSLから参照することはできない
