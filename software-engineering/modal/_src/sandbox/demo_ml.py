import modal

# アプリケーションの定義
app = modal.App("sandbox-demo-ml")

# モデル保存用のボリューム
model_volume = modal.Volume.from_name("sandbox-demo-ml-models", create_if_missing=True)

# 軽量なML用のイメージ
ml_image = modal.Image.debian_slim().pip_install(["scikit-learn", "joblib", "numpy"])

@app.function(image=ml_image, volumes={"/models": model_volume})
def train_model():
    import joblib, numpy as np, os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    model_path = "/models/classifier.joblib"
    if os.path.exists(model_path):
        return "モデル既存"
    
    # データ生成・モデル訓練
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, model_path)
    model_volume.commit()
    return "訓練完了"

@app.function(image=ml_image, volumes={"/models": model_volume})
def predict(features):
    import joblib, numpy as np
    model = joblib.load("/models/classifier.joblib")
    return {"prediction": int(model.predict([features])[0])}

if __name__ == "__main__":
    with app.run():
        print("訓練:", train_model.remote())
        print("予測:", predict.remote([1.0, 2.0, 3.0, 4.0])) 
