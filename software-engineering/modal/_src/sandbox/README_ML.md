# Modalで機械学習モデルを効率的に動かす方法

Modalで機械学習モデルを使う際に毎回モデルをダウンロードするのを避けて、高速化する方法を説明します。

## 主要な最適化手法

### 1. Modal Volume（推奨）
- **概要**: モデルファイルを永続化ストレージに保存
- **利点**: モデルのダウンロードが初回のみ、高速なアクセス
- **適用場面**: 一般的な機械学習モデル全般

### 2. Container Idle Timeout
- **概要**: コンテナをメモリに保持してモデルロード時間を短縮
- **利点**: モデルが既にメモリにロード済み状態を維持
- **適用場面**: 頻繁にアクセスされるモデル

### 3. GPU最適化
- **概要**: GPU利用時のメモリ効率化
- **利点**: 大きなモデルでもメモリ効率良く動作
- **適用場面**: Transformer系の大きなモデル

## 実装例

### シンプルな例（scikit-learn）
```bash
modal run simple_ml_example.py
```

### 高度な例（GPU + Transformers）
```bash
modal run ml_model_example.py
modal run advanced_ml_example.py
```

## ファイル構成

- `simple_ml_example.py`: scikit-learnを使ったシンプルな例
- `ml_model_example.py`: HuggingFace Transformersの基本例
- `advanced_ml_example.py`: GPU使用 + 複数モデル管理の高度な例

## 主要なポイント

### 1. Volume の使用
```python
# ボリュームを定義
model_volume = modal.Volume.from_name("ml-models", create_if_missing=True)

# 関数でボリュームをマウント
@app.function(volumes={"/models": model_volume})
def my_function():
    # モデルファイルは /models/ に保存される
    pass
```

### 2. モデルの存在チェック
```python
if not os.path.exists(model_path):
    # ダウンロードと保存
    model.save_pretrained(model_path)
    model_volume.commit()  # 重要：変更をコミット
```

### 3. GPU使用時の最適化
```python
@app.function(
    gpu="t4",  # GPU指定
    container_idle_timeout=300,  # アイドル時間を設定
)
```

### 4. クラスベースのアプローチ
```python
@app.cls(container_idle_timeout=300)
class MyModel:
    def __enter__(self):
        # モデルをメモリにロード（一度だけ）
        self.model = load_model()
        return self
    
    @modal.method()
    def predict(self, data):
        # 予測実行
        return self.model.predict(data)
```

## 実行手順

1. **最初にモデルをセットアップ（一度だけ）:**
   ```bash
   modal run simple_ml_example.py
   ```

2. **その後の予測は高速:**
   - モデルは既にVolumeに保存済み
   - ダウンロード時間なしで即座に利用可能

## パフォーマンス比較

| アプローチ | 初回実行時間 | 2回目以降 | GPU使用 | メモリ効率 |
|------------|-------------|-----------|---------|------------|
| 毎回ダウンロード | 遅い（数分） | 遅い（数分） | ○ | △ |
| Volume使用 | 遅い（初回のみ） | 高速（数秒） | ○ | ○ |
| Class + Idle | 遅い（初回のみ） | 超高速（ミリ秒） | ○ | ◎ |

## トラブルシューティング

### よくある問題
1. **Volume のコミットを忘れる**
   - 解決策: `volume.commit()` を必ず呼ぶ

2. **メモリ不足**
   - 解決策: `torch_dtype=torch.float16` で精度を下げる

3. **タイムアウト**
   - 解決策: `timeout` パラメータを調整

### デバッグ方法
```python
# ボリュームの内容を確認
@app.function(volumes={"/models": model_volume})
def list_models():
    import os
    return os.listdir("/models")
```

## 次のステップ

1. **シンプルな例から開始**: `simple_ml_example.py`
2. **自分のモデルに適用**: モデル名と処理を変更
3. **GPU が必要な場合**: `advanced_ml_example.py` を参考
4. **本番環境での最適化**: クラスベースのアプローチを検討 