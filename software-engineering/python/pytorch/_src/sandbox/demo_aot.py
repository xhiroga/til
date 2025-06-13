import argparse
import os
import time

import torch
import torch_tensorrt
import torchvision.models as models

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")


def benchmark(model, input_tensor, num_runs=100):
    input_tensor = input_tensor.to("cuda")
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

    exported_model_path = "models/exported_trt.pt2"

    if args.export:
        try:
            print("Compiling the model with Torch-TensorRT...")
            dummy_input_cuda = dummy_input.to("cuda")
            model.to("cuda")
            compiled_model = torch_tensorrt.compile(
                model, ir="dynamo", inputs=[dummy_input_cuda]
            )
            print("Model compiled successfully with Torch-TensorRT.")

            print(f"Exporting compiled TensorRT model to {exported_model_path}...")
            torch_tensorrt.save(
                compiled_model, exported_model_path, inputs=[dummy_input_cuda]
            )
            os.makedirs(os.path.dirname(exported_model_path), exist_ok=True)
            print("Compiled TensorRT model exported successfully.")
        except Exception as e:
            print(f"Error during compiled model export: {e}")
            if os.path.exists(exported_model_path):
                os.remove(exported_model_path)
            return
    else:
        if not os.path.exists(exported_model_path):
            print(
                f"Compiled model not found at {exported_model_path}. Please run with --export first."
            )
            return

        print("Running ResNet50 benchmark (original vs compiled & exported)...")
        dummy_input.to("cuda")
        model.to("cuda")
        original_model_time = benchmark(model, dummy_input)
        print(f"Original model average inference time: {original_model_time:.3f} ms")

        print(f"Loading exported compiled TensorRT model from {exported_model_path}...")
        loaded_exported_program = torch.export.load(exported_model_path).module()
        print("Compiled TensorRT model loaded successfully.")

        exported_model_time = benchmark(loaded_exported_program, dummy_input)
        print(f"Compiled model average inference time: {exported_model_time:.3f} ms")

        speedup = original_model_time / exported_model_time
        print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
