import argparse
import os
import time

import torch
import torch.nn as nn
import torchvision.models as models
from torch.export import export, load, save


def benchmark(model, input_tensor, num_runs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)
    for _ in range(10):  # Warmup
        model(input_tensor)

    start_time = time.time()
    for _ in range(num_runs):
        model(input_tensor)
    return (time.time() - start_time) / num_runs * 1000  # ms


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch AOT Autograd Demo with ResNet50"
    )
    parser.add_argument(
        "--export", action="store_true", help="Export the model to an .pt2 file"
    )
    args = parser.parse_args()

    batch_size = 64
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()

    exported_model_path = "models/exported.pt2"

    if args.export:
        try:
            print("Compiling the model with torch.compile...")
            compiled_model = torch.compile(model, backend="inductor")
            print("Model compiled successfully.")

            print(f"Exporting compiled ResNet50 model to {exported_model_path}...")
            exported_program = export(compiled_model, (dummy_input,))
            save(exported_program, exported_model_path)
            os.makedirs(os.path.dirname(exported_model_path), exist_ok=True)
            print("Compiled model exported successfully.")
        except Exception as e:
            print(f"Error during compiled model export: {e}")
            if os.path.exists(exported_model_path):
                os.remove(exported_model_path)
            return
    else:
        if not os.path.exists(exported_model_path):
            print(
                f"Exported compiled model not found at {exported_model_path}. Please run with --export first."
            )
            return

        print("Running ResNet50 benchmark (original vs compiled & exported)...")
        original_model_time = benchmark(model, dummy_input)
        print(f"Original model average inference time: {original_model_time:.3f} ms")

        try:
            print(f"Loading exported compiled model from {exported_model_path}...")
            loaded_exported_program = load(exported_model_path)
            print("Compiled model loaded successfully.")

            class ExportedProgramWrapper(nn.Module):
                def __init__(self, program_to_wrap):
                    super().__init__()
                    self.program_module = program_to_wrap.module()

                def forward(self, x):
                    return self.program_module(x)

            exported_model_for_benchmark = ExportedProgramWrapper(
                loaded_exported_program
            )
            exported_model_time = benchmark(exported_model_for_benchmark, dummy_input)
            print(
                f"Compiled and Exported model average inference time: {exported_model_time:.3f} ms"
            )

            if original_model_time > 0 and exported_model_time > 0:
                speedup = original_model_time / exported_model_time
                print(f"Speedup: {speedup:.2f}x")
            else:
                print(
                    "Cannot calculate speedup due to zero or negative inference time."
                )
        except Exception as e:
            print(f"Error loading or benchmarking exported model: {e}")


if __name__ == "__main__":
    main()
