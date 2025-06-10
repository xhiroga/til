import os
import time

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch_tensorrt
import torch_tensorrt.dynamo.backend
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_or_create_input_tensor(image_path, batch_size):
    """Loads an image and preprocesses it, or creates a dummy tensor."""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if image_path and os.path.exists(image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0)
            return img_tensor.repeat(batch_size, 1, 1, 1).cuda()
        except Exception as e:
            print(f"Error loading image {image_path}: {e}. Using dummy data.")
    
    return torch.randn(batch_size, 3, 224, 224).cuda()

def build_engine(onnx_file_path, engine_file_path, precision_mode='fp32', opt_batch_size=8):
    """Builds a TensorRT engine from an ONNX file."""
    if not os.path.exists(onnx_file_path):
        print(f"ONNX file not found: {onnx_file_path}")
        return False

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, 
                      (1, 3, 224, 224), 
                      (opt_batch_size, 3, 224, 224),
                      (16, 3, 224, 224))
    config.add_optimization_profile(profile)

    if precision_mode == 'fp16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision_mode == 'fp16':
        print("FP16 not supported. Building in FP32.")
    
    print(f"Building TensorRT engine ({precision_mode})...")
    serialized_engine = builder.build_serialized_network(network, config)

    if not serialized_engine:
        print("Failed to build the TensorRT engine.")
        return False

    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Engine saved to {engine_file_path}")
    return True

class TensorRTEngineRunner:
    """Manages TensorRT engine loading and inference."""
    def __init__(self, engine_file_path):
        self.engine_file_path = engine_file_path
        self.stream = cuda.Stream()
        self.device_buffers = {}
        self.input_info = None
        self.output_infos = []

        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            mode = self.engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_info = {'name': name, 'dtype': dtype}
            else:
                self.output_infos.append({'name': name, 'dtype': dtype})

    def infer(self, input_tensor_numpy):
        """Performs inference."""
        self.context.set_input_shape(self.input_info['name'], input_tensor_numpy.shape)
        
        input_name = self.input_info['name']
        if input_name not in self.device_buffers:
            self.device_buffers[input_name] = cuda.mem_alloc(input_tensor_numpy.nbytes)
        
        cuda.memcpy_htod_async(self.device_buffers[input_name], input_tensor_numpy, self.stream)
        self.context.set_tensor_address(input_name, int(self.device_buffers[input_name]))

        outputs = {}
        for out_info in self.output_infos:
            output_name = out_info['name']
            output_shape = self.context.get_tensor_shape(output_name)
            host_buffer = np.empty(output_shape, dtype=out_info['dtype'])
            
            if output_name not in self.device_buffers:
                self.device_buffers[output_name] = cuda.mem_alloc(host_buffer.nbytes)
            
            self.context.set_tensor_address(output_name, int(self.device_buffers[output_name]))
            outputs[output_name] = host_buffer
        
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        for out_info in self.output_infos:
            output_name = out_info['name']
            cuda.memcpy_dtoh_async(outputs[output_name], self.device_buffers[output_name], self.stream)
        
        self.stream.synchronize()
        return outputs

_trt_runners = {}

def get_trt_runner(engine_path):
    if engine_path not in _trt_runners:
        _trt_runners[engine_path] = TensorRTEngineRunner(engine_path)
    return _trt_runners[engine_path]

def pytorch_inference_runner(model, input_tensor, is_warmup=False):
    model.eval()
    with torch.no_grad():
        return model(input_tensor)

def pytorch_compile_inference_runner(compiled_model, input_tensor, is_warmup=False):
    with torch.no_grad():
        return compiled_model(input_tensor)

def trt_inference_runner(engine_path, input_tensor, is_warmup=False):
    runner = get_trt_runner(engine_path)
    input_numpy = input_tensor.cpu().numpy().astype(runner.input_info['dtype'], copy=False)
    return runner.infer(input_numpy)

def benchmark_framework(name, inference_func, model_or_path, input_tensor, num_runs=100):
    """Generic benchmark function."""
    print(f"\nBenchmarking {name} ({num_runs} runs, batch_size={input_tensor.shape[0]})...")
    
    # Warm-up
    for _ in range(10):
        _ = inference_func(model_or_path, input_tensor, is_warmup=True)
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        _ = inference_func(model_or_path, input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_runs * 1000
    print(f"{name} average inference time: {avg_time_ms:.3f} ms")
    return avg_time_ms

def main():
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 8
    num_runs = 100
    image_path = None  # Use dummy data

    onnx_path = os.path.join(output_dir, "resnet50.onnx")
    fp32_engine_path = os.path.join(output_dir, f"resnet50_fp32_b{batch_size}.engine")
    fp16_engine_path = os.path.join(output_dir, f"resnet50_fp16_b{batch_size}.engine")

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    print("Loading ResNet50 model...")
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).cuda().eval()

    input_tensor = load_or_create_input_tensor(image_path, batch_size)

    # PyTorch benchmark
    pytorch_time = benchmark_framework("PyTorch", pytorch_inference_runner, 
                                      resnet50, input_tensor, num_runs)

    # PyTorch Compile benchmark
    pytorch_compile_time = float('inf')
    try:
        print("Compiling model with torch.compile...")
        compiled_model = torch_tensorrt.compile(resnet50, ir="dynamo", inputs=[input_tensor])
        pytorch_compile_time = benchmark_framework("PyTorch Compile", pytorch_compile_inference_runner,
                                                  compiled_model, input_tensor, num_runs)
    except Exception as e:
        print(f"torch.compile failed: {e}")

    # ONNX export
    if not os.path.exists(onnx_path):
        print(f"Exporting to ONNX: {onnx_path}")
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        torch.onnx.export(resnet50, dummy_input, onnx_path,
                          export_params=True, opset_version=12, do_constant_folding=True,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        print("ONNX export complete.")

    # TensorRT FP32 benchmark
    fp32_time = float('inf')
    if not os.path.exists(fp32_engine_path):
        build_engine(onnx_path, fp32_engine_path, 'fp32', batch_size)
    
    if os.path.exists(fp32_engine_path):
        fp32_time = benchmark_framework("TensorRT FP32", trt_inference_runner,
                                       fp32_engine_path, input_tensor, num_runs)

    # TensorRT FP16 benchmark
    fp16_time = float('inf')
    temp_builder = trt.Builder(TRT_LOGGER)
    if temp_builder.platform_has_fast_fp16:
        if not os.path.exists(fp16_engine_path):
            build_engine(onnx_path, fp16_engine_path, 'fp16', batch_size)
        
        if os.path.exists(fp16_engine_path):
            fp16_time = benchmark_framework("TensorRT FP16", trt_inference_runner,
                                           fp16_engine_path, input_tensor, num_runs)
    else:
        print("FP16 not supported on this platform.")

    # Summary
    print("\n--- Performance Summary ---")
    print(f"Input Batch Size: {batch_size}")
    print(f"PyTorch: {pytorch_time:.3f} ms")
    
    if pytorch_compile_time != float('inf'):
        speedup = pytorch_time / pytorch_compile_time
        print(f"PyTorch Compile: {pytorch_compile_time:.3f} ms (Speedup: {speedup:.2f}x)")

    if fp32_time != float('inf'):
        speedup = pytorch_time / fp32_time
        print(f"TensorRT FP32: {fp32_time:.3f} ms (Speedup: {speedup:.2f}x)")
    
    if fp16_time != float('inf'):
        speedup_pytorch = pytorch_time / fp16_time
        print(f"TensorRT FP16: {fp16_time:.3f} ms (Speedup: {speedup_pytorch:.2f}x)")
        if fp32_time != float('inf'):
            speedup_fp32 = fp32_time / fp16_time
            print(f"  (FP16 vs FP32: {speedup_fp32:.2f}x)")

if __name__ == "__main__":
    main() 