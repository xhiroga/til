# musubi-tuner

## Note

- Step数を**概算する**計算式は次のとおり。
  - 動画枚数 * フレーム数 * 1/4 * 1/latent_window_size * max_train_epochs
  - 1/4 は ピクセルフレーム > VAEフレームへの時間方向の圧縮率
  - latent_window_size は デフォルトで9
  - 例: 5枚 * 73フレーム * 1/4 * 1/9 * 16 epoch =  480
