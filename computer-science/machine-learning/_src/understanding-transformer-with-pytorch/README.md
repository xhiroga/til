[Python(PyTorch)で自作して理解するTransformer](https://zenn.dev/yukiyada/articles/59f3b820c52571)

```powershell
conda env create understanding-transformer-with-pytorch
conda run -n understanding-transformer-with-pytorch poetry run python src/train.py
```

## 学んだこと

- Decoderがニューラルネットワークの中でどこに位置するのかわかった
- Decoderの訓練時の入力と出力が2組のトークンの配列で、片方はもう片方を1つづつずらしたものであることがわかった
- 訓練データをミニバッチに分割し、その数だけイテレーションを回すことと、Epochはその処理自体を何回繰り返すかを表すことがわかった
