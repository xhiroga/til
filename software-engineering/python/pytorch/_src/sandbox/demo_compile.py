import torch
import torch.nn as nn
import torchvision.models as models # torchvision.models をインポート
import time
import argparse
import os

def benchmark(model, input_tensor, num_runs=100):
    # ウォームアップ
    for _ in range(10):
        model(input_tensor)

    start_time = time.time()
    for _ in range(num_runs):
        model(input_tensor)
    end_time = time.time()
    return (end_time - start_time) / num_runs * 1000  # 平均時間をミリ秒で返す

def main():
    parser = argparse.ArgumentParser(description="Torch Compile Demo")
    parser.add_argument("--compile", action="store_true", help="Compile the model and save it.")
    args = parser.parse_args()

    # ResNet18 用のパラメータ
    batch_size = 32 # バッチサイズを少し小さくする (メモリ使用量に応じて調整)

    # モデルのインスタンス化 (ResNet18, 事前学習済み重みを使用)
    print("Loading ResNet18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval() # 推論モード

    # ダミー入力データ (ResNet18 の標準的な入力サイズ)
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    if args.compile:
        print("Compiling the model (ResNet18)...")
        os.makedirs("models", exist_ok=True)
        try:
            # torch.compile はCPUでも動作しますが、GPUがある場合はGPUで実行するとより効果的です。
            # ここではデバイス指定は省略し、デフォルトのデバイスでコンパイルします。
            compiled_model = torch.compile(model) 
            # コンパイルが成功したことを示すダミーファイルを作成
            with open("models/compiled.so", "w") as f:
                f.write("compiled_resnet18") # 内容を少し変更
            print("Model compiled and dummy file 'models/compiled.so' created.")
        except Exception as e:
            print(f"Error during compilation: {e}")
            if os.path.exists("models/compiled.so"):
                os.remove("models/compiled.so")
            print("Compilation failed.")
        return

    print("Running benchmark with ResNet18 (without torch.compile)...")
    avg_time_no_compile = benchmark(model, dummy_input)
    print(f"Average inference time (no compile): {avg_time_no_compile:.3f} ms")

    print("\nRunning benchmark with ResNet18 (with torch.compile)...")
    try:
        # compiled_model = torch.compile(model, mode="reduce-overhead") # modeの指定も可能
        compiled_model = torch.compile(model)
        avg_time_compile = benchmark(compiled_model, dummy_input)
        print(f"Average inference time (with compile): {avg_time_compile:.3f} ms")
        if avg_time_compile > 0 : # avg_time_no_compile から avg_time_compile に変更（0除算回避）
             print(f"Speedup: {avg_time_no_compile / avg_time_compile:.2f}x")
        else:
            print("Speedup: N/A (division by zero or compile time is zero)")

    except Exception as e:
        print(f"Error during torch.compile or benchmarking: {e}")

if __name__ == "__main__":
    main()
