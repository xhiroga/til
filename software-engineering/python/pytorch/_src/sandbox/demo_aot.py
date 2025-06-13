import torch
import torch.nn as nn
import time
import argparse
import os
# from torch._export import aot_compile # torch.export を使用するためコメントアウト
from torch.export import export, load, save as export_save # export_save を正しくインポート
import torchvision.models as models # torchvision.models をインポート

def get_model():
    # weights=None でランダム初期化、ResNet50_Weights.DEFAULT で事前学習済み
    weights = models.ResNet50_Weights.DEFAULT # 常に事前学習済み重みを使用
    model = models.resnet50(weights=weights) # ResNet50 に変更
    # モデルの出力を確認・調整する場合 (例: 分類タスクのクラス数)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 10) # 例: 10クラス分類
    return model

def benchmark(model, input_tensor, num_runs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    # ウォームアップ
    for _ in range(10):
        model(input_tensor)

    start_time = time.time()
    for _ in range(num_runs):
        model(input_tensor)
    end_time = time.time()
    return (end_time - start_time) / num_runs * 1000  # 平均実行時間 (ms)

def main():
    parser = argparse.ArgumentParser(description="PyTorch AOT Autograd Demo with ResNet50") # 説明を更新
    parser.add_argument("--export", action="store_true", help="Export the model to an .so file")
    args = parser.parse_args()

    # ResNet50用の入力サイズ
    batch_size = 64
    dummy_input = torch.randn(batch_size, 3, 224, 224) # ResNet50の標準的な入力形状

    model = get_model() # pretrained 引数を削除
    model.eval() # 推論モードに設定

    if args.export:
        print("Exporting ResNet50 model...") # モデル名を更新
        os.makedirs("models", exist_ok=True)
        try:
            exported_program = export(model, (dummy_input,))
            print(f"Exported program type: {type(exported_program)}")
            export_save(exported_program, "models/exported.pt2")
            print("Model exported to models/exported.pt2")

        except ImportError:
            print("torch.export is not available. Please use PyTorch 2.0 or later.")
        except Exception as e:
            print(f"Error during model export: {e}")

    else:
        print("Running ResNet50 benchmark...") # モデル名を更新

        original_model_time = benchmark(model, dummy_input)
        print(f"Original model average inference time: {original_model_time:.3f} ms")

        exported_model_path = "models/exported.pt2"

        if os.path.exists(exported_model_path):
            try:
                print(f"Loading exported model from {exported_model_path}...")
                loaded_exported_program = load(exported_model_path) # torch.export.load を使用
                print("Model loaded successfully.")

                # benchmark 関数は nn.Module を期待するため、ラッパーを作成
                class ExportedProgramWrapper(nn.Module):
                    def __init__(self, program_to_wrap): # 引数名を変更して明確化
                        super().__init__()
                        self.program_module = program_to_wrap.module() # .module() を呼び出して保持
                    def forward(self, x):
                        return self.program_module(x) # 保持しているモジュールを呼び出す

                exported_model_for_benchmark = ExportedProgramWrapper(loaded_exported_program)

                exported_model_time = benchmark(exported_model_for_benchmark, dummy_input)
                print(f"Exported model average inference time: {exported_model_time:.3f} ms")

                if original_model_time > 0 and exported_model_time > 0:
                    speedup = original_model_time / exported_model_time
                    print(f"Speedup: {speedup:.2f}x")
                elif original_model_time <= 0: # 0以下の場合をまとめて処理
                    print("Original model time is zero or negative, cannot calculate speedup.")
                else: # exported_model_time <= 0 の場合
                    print("Exported model time is zero or negative, cannot calculate speedup meaningfully.")

            except ImportError:
                print("torch.export.load is not available. Cannot load the exported model.")
            except FileNotFoundError:
                print(f"Exported model not found at {exported_model_path}. Please run with --export first.")
            except Exception as e:
                print(f"Error loading or benchmarking exported model: {e}")
                print(f"Please ensure '{exported_model_path}' was created correctly via '--export'.")

        else:
            print(f"Exported model not found at {exported_model_path}.")
            print("Please run with --export first to generate the model file.")

if __name__ == "__main__":
    main()
