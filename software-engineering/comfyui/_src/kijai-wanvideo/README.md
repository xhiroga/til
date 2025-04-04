---
notebook_urls:
  - Precision/Recall: https://chatgpt.com/c/67ee0be0-7cf0-8010-9841-0f2045c7748f
---

# Kijai/WanVideo

## Memo

- 起動時に `--highvram` オプションが有効な場合、WanVideo Model Loader がOOMになることがある
  - ComfyUIの仕様で、オフロード先がGPUになるため。実質無意味（参考: [OOM with use of ComfyUI --highvram parameter affecting the offload_device selection #196
](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/196)）
- Frame数が49や77と中途半端に見えるのはなぜなんだろうか？

### [workflows/kijai/wanvideo_long_T2V_example_01.json](workflows/kijai/wanvideo_long_T2V_example_01.json)

- 普通のT2V
- L40Sで7分程度

[![WanVideo2 1 T2V](http://img.youtube.com/vi/NeT2DpWUUcs/0.jpg)](https://www.youtube.com/watch?v=NeT2DpWUUcs "WanVideo2 1 T2V")

### [workflows/kongo-jun/sample_wanvideo_480p_I2V_endframe_example_01.json](workflows/kongo-jun/sample_wanvideo_480p_I2V_endframe_example_01.json)

- 混合順さんのワークフロー（オリジナルは @kijai ）
- Start, End 共に指定されている
  - Start, Endは画像埋め込みとしてサンプラーに渡される
- End Frameに従わないケースがある
- 生成される動画の品質は @raindrop313 の WanVideoStartEndFrames と変わらないように見える
- modelがbf16のままではOOMで落ちたので、fp8に差し替えている（環境はL40S）
  - noteの記事ではRTX4090で動かされたとのことだが、可能なんだろうか...?
- Sampling 81 frames at 480x832 with 20 steps, 約5分, ピーク時VRAM約37GiB (fp8, L40S)

#### 77frames, 16fps

[![WanVideoWrapper I2V endframe - Asagi-chan hands up (77 frames, 16 fps)](http://img.youtube.com/vi/eoOeKCt3jvQ/0.jpg)](https://www.youtube.com/watch?v=eoOeKCt3jvQ "WanVideoWrapper I2V endframe - Asagi-chan hands up (77 frames, 16 fps)")

- `This is an image of an anime heroine raising her hands.`
- 手を挙げるだけのアニメーションに5秒あると、時間をゆっくりと使う印象がある
  - 具体的には、手を挙げる → 表情を変える → 手の微調整
- ところどころ残像が見えてしまっている

#### 49frames, 24fps

[![WanVideoWrapper I2V endframe - Asagi-chan hands up (49 frames, 24 fps)](http://img.youtube.com/vi/E6Nffvoc-TI/0.jpg)](https://www.youtube.com/watch?v=E6Nffvoc-TI "WanVideoWrapper I2V endframe - Asagi-chan hands up (49 frames, 24 fps)")

- `This is an image of an anime heroine raising her hands.`
- L40Sで3分程度

#### 49frames, 24fps

[![WanVideoWrapper I2V endframe - Asagi-chan hands up (49 frames, 24 fps)](http://img.youtube.com/vi/WtgZzvdYcUg/0.jpg)](https://www.youtube.com/watch?v=WtgZzvdYcUg "WanVideoWrapper I2V endframe - Asagi-chan hands up (49 frames, 24 fps)")

- `This is an image of an anime heroine quickly raising her hands. It has a clear contrast, as you would expect from an anime image.`
- 手を挙げるのが素早すぎる。プロンプトに忠実とも言える

#### 49frames, 24fps

[![WanVideoWrapper I2V endframe - Asagi-chan hands up (49 frames, 24 fps) x8](http://img.youtube.com/vi/eHdgl48cl0c/0.jpg)](https://www.youtube.com/watch?v=eHdgl48cl0c "WanVideoWrapper I2V endframe - Asagi-chan hands up (49 frames, 24 fps) x8")

- `This is an image of an anime heroine quickly raising her hands.`
- Seed値をランダムにして8通り生成した。
  - Seed値による品質のブレが非常に大きい。ギリ実用レベル〜意図しない暗転や残像まで広く、打率は良いところ0.2くらい。

#### 25frames, 24fps, Negative Prompt: 中国語

[![WanVideoWrapper I2V endframe - Asagi-chan hands up (25 frames, 24 fps, Negative Prompt: 中国語) x8](http://img.youtube.com/vi/TaTjvE1xDqc/0.jpg)](https://www.youtube.com/watch?v=TaTjvE1xDqc "WanVideoWrapper I2V endframe - Asagi-chan hands up (25 frames, 24 fps, Negative Prompt: 中国語) x8")

- L40Sで1分半程度。
- 若干打率が上がった気がする
- 25 frame "も" あると、途中の細かい動きを指示するのが難しい。
  - 1秒近い映像を作るなら、V2Vの方が細かく指示ができて良い。

#### 5frames, 24fps, Negative Prompt: English

[![WanVideoWrapper I2V endframe - Asagi-chan hands up (5 frames, 24 fps, Negative Prompt: English) x4](http://img.youtube.com/vi/7m_6GXA3hag/0.jpg)](https://www.youtube.com/watch?v=7m_6GXA3hag "WanVideoWrapper I2V endframe - Asagi-chan hands up (5 frames, 24 fps, Negative Prompt: English) x4")

- フレーム数を下げれば丁寧な補完をするかと思ったが、全然上手くいかない。
- プロンプトが「手を挙げる様子」で、すでに手が挙がっているという矛盾が良くないかもしれない。
  - プロンプトに拘らずフレーム補間をするにはLoRAの開発が必要かもしれない。

### [workflows/8co28/wanfuncn.json](workflows/8co28/wanfuncn.json)

- ノードを修正済み。WanVideoWrapper@ce6522c 以前のどこかで、WanVideo ImageToVideo Encode に対して破壊的変更が入ったため。
- Stable DiffusionのControl Netとは違い、ネットワークの追加ではなく埋め込みの追加であるにも関わらず、よく制御できているように見える。
- Positive Prompts: `anime style,deer,` と、非常にシンプルで良いのも特徴的。
  - 試してみたところ、Positiv Prompts無しでもほぼ同様に動作した。

#### [Wan2 1 Fun Control 14B fp8 - Post-apocalyptic (57frames, 16fps) x6](workflows/works/wanfuncn.json)

[![Wan2 1 Fun Control 14B fp8 - Post-apocalyptic (57frames, 16fps) x6](http://img.youtube.com/vi/WJQpodbH7xU/0.jpg)](https://www.youtube.com/watch?v=WJQpodbH7xU "Wan2 1 Fun Control 14B fp8 - Post-apocalyptic (57frames, 16fps) x6")

- オリジナル画像(ChatGPT 4o)と線路の映像から生成
- 57 frames → L40Sで3分程度
- 結論、Control映像からの影響をかなり受ける。というかむしろ、start_imageの影響は最初のフレームと画風のみ。
  - いっそノイズとかの方が良かったりするのかな？
- WanVideo Samplerのsmaplesが入力されていないが、他の画像/映像で試した感じでは、あってもなくてもあまり差を感じない。要追加検証。

## [workflows/ai-73-30/Wan-Fun视频首尾帧-KJ版.json]()

- [オリジナル](https://zhuanlan.zhihu.com/p/1890424727547381581)と比較して、カスタムノードの種類を削減。

## References

- [Wan2.1のI2V関連のワークフローをComfyUIで色々動かしてみた](https://note.com/kongo_jun/n/nf9d9d2903a42)
