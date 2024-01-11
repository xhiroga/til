# Computer Graphics

## Models

画像生成には次のようなモデルがあります。

| モデル | 説明 | 代表的なモデル | 論文 |
|---|---|---|---|
| GANs[^GANs] | 生成器(Generator)と識別器(Discriminator)の2つのネットワークを用意する。生成器が生成した画像と本物の画像をシャッフルし、識別器がチューリングテストのように偽物の画像を見分けようとすることで、より本物に近い画像を生成する。 | BigGAN, StyleGAN | [\[2001.06937\] A Review on Generative Adversarial Networks: Algorithms, Theory, and Applications](https://arxiv.org/abs/2001.06937) |
| VAE[^VAE] | 潜在空間と呼ばれる低次元空間にデータを圧縮し、そこからデータを生成する手法。Encoderは元のデータを潜在空間に変換し、Decoderは潜在空間のベクトルを元のデータに復元する。VAEでは、潜在空間が正規分布に従うように学習することで、潜在空間からランダムにサンプリングしたベクトルを入力として、多様なデータを生成することができる。 | VQ-VAE, VQ-VAE2 | [\[1312.6114\] Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) |
| Flow | GANやVAEよりも確率密度推定の正確さで優れる。 | NICE, RealNVP, [Glow](https://github.com/openai/glow) | [\[1410.8516\] NICE: Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516) |
| Diffusion | データ生成の際に、時間的なステップごとに確率的にノイズを加えながら、実在するデータを生成する手法。生成されるデータは、最終的なステップでのノイズが無いデータから始まり、逆方向にノイズを除去して生成される。 | DALL-E, Imagen, Stable Diffusion |
[^GANs]: Generative Adversarial Networks, 敵対的生成ネットワーク
[^VAE]: Variational Auto-Encoder, 変分オートエンコーダ

## U-Net

## Stable Diffusion

### Stable Diffusion Models

|名前|開発者|CLIP Score|学習データ|ライセンス|論文・公式URL|
|---|---|---|---|
| Stable Diffusion 2.1 | Stability AI | CLIP Score |LAION-Aesthetics[^LAION-Aesthetics]| ライセンス | | [Stable Diffusionローンチのお知らせ - Stability AI](https://stability.ai/blog/stable-diffusion-announcement) |
| Cool Japan Diffusion | Stability AI | CLIP Score|LAION-Aesthetics[^LAION-Aesthetics] | [Stable Diffusionローンチのお知らせ - Stability AI](https://stability.ai/blog/stable-diffusion-announcement) |

[^LAION-Aesthetics]:LAIONの画像データセットを「美しさ」に基づいてフィルタリングしたサブセット。

ベンチマーク用のパラメータは次の通り。

|Parameter|Value|
|---|---|
|Prompt|a smiling girl in a white T-shirts and jeans jumping high while bending her knees and making a V sign with her hands near her cheeks for joy,<br>a green hill and blue sky with white clouds,<br>detail facial features, full body, front view|
|Negative Prompt|artist name, copyright, error, logo, signature, text, watermark,<br>lowres, bad anatomy, worst quality, duplicate, long neck, bad hand, extra digit, fewer digit, extra legs, fewer legs,<br>out of frame|
|Sampler|Euler|
|Steps|28|
|CFG scale|12|
|Seed|3945564133|
|Face restoration|CodeFormer|
|Size|360x640|

#### Stable Diffusion 2.1

![青春ベンチマーク (Stable Diffusion 2.1)](https://bnz07pap001files.storage.live.com/y4mzBBRbLx354jUmUJQGN0u12oU4B2oC26pZAjluILfXMThcSF-5CM9ZstFIsYcMiS9FeGdz32uOjJWXbUqad3GUvFSDpuMw6wl0K99DDdb0IU1W__E1qvqdS0XQh0B8uMX2h3UCq170g3iJPSPxYaqCZtCQFgPygyb3WuqVV1ieo8?width=1080&height=1920&cropmode=none)

#### Cool Japan Diffusion 2.1.0 beta (25d0b8d594)

![青春ベンチマーク (Cool Japan Diffusion 2.1.0 beta)](https://bnz07pap001files.storage.live.com/y4mccty5BkJM__ciUc_s_v1EEJrfgBVmf0rokAu7CRifAMq1L-YwYOsLqHRpKtJCbIEXnNYPb6iYL0gVPiU_cmHW_isVu2rhcqF8kP_XAAJMfz4luTpuCt6qO4mu3EuR9900u45DvD8m0WPvluDIvF9ewLEIHQAWufUnjlyTC82J6w?width=1080&height=1920&cropmode=none)

#### Mitsua Diffusion One (2da4fb419f)

![青春ベンチマーク (Mitsua Diffusion One)](https://bnz07pap001files.storage.live.com/y4mln0UOc7tVsQ2uLe3NlCcrI7k7tR-3zgH7eVGgT3KbYnJ-jm6CQfhwv2qChG7770kTVkTfavMd6izVqTROWhRjpEraJDM_97FcJlysP8BMSb6yl4PVkuofCPqo5HkiMApKTunKplFdBIb-pzlWerSCFgEHQQRToPQwIunVM9-fyY?width=1080&height=1920&cropmode=none)

### NPの原理

<https://twitter.com/11t_game/status/1630837011384012800?s=20>

### モデル破損について

[Stable Diffusionモデルの破損チェックと修復について｜まゆひら｜note](https://note.com/mayu_hiraizumi/n/n09a7d6ec5678)

[arenatemp/stable\-diffusion\-webui\-model\-toolkit: A Multipurpose toolkit for managing, editing and creating models\.](https://github.com/arenatemp/stable-diffusion-webui-model-toolkit)

## Control Net[^control-net]

[^control-net]: [[2302.05543] Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)

## References

- [Stable Diffusion を基礎から理解したい人向け論文攻略ガイド【無料記事】](https://ja.stateofaiguides.com/20221012-stable-diffusion/)
