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


def export_compiled(model, input_tensor, exported_model_path):
    print(f"Compiling and export to {exported_model_path}...")
    input_tensor_cuda = input_tensor.to("cuda")
    model.to("cuda")
    compiled_model = torch_tensorrt.compile(
        model, ir="dynamo", inputs=[input_tensor_cuda]
    )
    os.makedirs(os.path.dirname(exported_model_path), exist_ok=True)
    torch_tensorrt.save(compiled_model, exported_model_path, inputs=[input_tensor_cuda])


def inference(model, input_tensor, exported_model_path):
    if not os.path.exists(exported_model_path):
        print(f"{exported_model_path} not found. Please run with --export first.")
        return

    print("Running ResNet50 benchmark (original vs compiled & exported)...")
    input_tensor.to("cuda")
    model.to("cuda")
    original_model_time = benchmark(model, input_tensor)
    print(f"Original model average inference time: {original_model_time:.3f} ms")

    print(f"Loading exported compiled TensorRT model from {exported_model_path}...")
    loaded_exported_program = torch.export.load(exported_model_path).module()
    print("Compiled TensorRT model loaded successfully.")

    exported_model_time = benchmark(loaded_exported_program, input_tensor)
    print(f"Compiled model average inference time: {exported_model_time:.3f} ms")

    speedup = original_model_time / exported_model_time
    print(f"Speedup: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch AOT Autograd Demo with ResNet50"
    )
    parser.add_argument(
        "--export", action="store_true", help="Export the model to models/trt.ep"
    )
    args = parser.parse_args()

    batch_size = 64
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()

    exported_model_path = "models/trt.ep"

    if args.export:
        try:
            export_compiled(model, input_tensor, exported_model_path)
        except Exception as e:
            print(f"Error during compiled model export: {e}")
            if os.path.exists(exported_model_path):
                os.remove(exported_model_path)
            return
    else:
        inference(model, input_tensor, exported_model_path)


if __name__ == "__main__":
    main()
