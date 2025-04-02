# Kijai/WanVideo

## Memo

### [workflows/kijai/wanvideo_long_T2V_example_01.json](workflows/kijai/wanvideo_long_T2V_example_01.json)

- 普通のT2V
- L40Sで7分程度

[![WanVideo2 1 T2V](http://img.youtube.com/vi/NeT2DpWUUcs/0.jpg)](https://www.youtube.com/watch?v=NeT2DpWUUcs "WanVideo2 1 T2V")

### [workflows/kongo-jun/sample_wanvideo_480p_I2V_endframe_example_01.json](workflows/kongo-jun/sample_wanvideo_480p_I2V_endframe_example_01.json)

- 混合順さんのワークフロー
- Start, End 共に指定されている
- modelがbf16のままではOOMで落ちたので、fp8に差し替えている（環境はL40S）
  - noteの記事ではRTX4090で動かされたとのことだが、可能なんだろうか...?
- Sampling 81 frames at 480x832 with 20 steps, 約5分, ピーク時VRAM約37GiB (fp8, L40S)

[![WanVideoWrapper I2V endframe](http://img.youtube.com/vi/iGbFLVoW3_U/0.jpg)](https://www.youtube.com/watch?v=iGbFLVoW3_U "WanVideoWrapper I2V endframe")

## References

- [Wan2.1のI2V関連のワークフローをComfyUIで色々動かしてみた](https://note.com/kongo_jun/n/nf9d9d2903a42)
