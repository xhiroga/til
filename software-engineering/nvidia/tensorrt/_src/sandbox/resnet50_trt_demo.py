import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # Implicitly initializes CUDA and manages context
import time
import os

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
            img_tensor = preprocess(img).unsqueeze(0) # Preprocess single image
            # Repeat the single image tensor to match the desired batch size
            return img_tensor.repeat(batch_size, 1, 1, 1).cuda()
        except Exception as e:
            print(f"Error loading image {image_path}: {e}. Using dummy data instead.")
    
    # print(f"Using dummy data for input tensor with batch size {batch_size}.") # Keep console less verbose
    return torch.randn(batch_size, 3, 224, 224).cuda()

def build_engine(onnx_file_path, engine_file_path, precision_mode='fp32', 
                 min_batch_size=1, opt_batch_size=8, max_batch_size=16):
    """Builds a TensorRT engine from an ONNX file with optimization profiles."""
    if not os.path.exists(onnx_file_path):
        print(f"ONNX file not found: {onnx_file_path}")
        return False # Indicate failure

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error_idx in range(parser.num_errors):
                print(parser.get_error(error_idx))
            return False
    # print("ONNX model parsed successfully.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, 
                      (min_batch_size, 3, 224, 224), 
                      (opt_batch_size, 3, 224, 224),
                      (max_batch_size, 3, 224, 224))
    config.add_optimization_profile(profile)
    # print(f"Optimization profile created for input '{input_name}' with min/opt/max batch: {min_batch_size}/{opt_batch_size}/{max_batch_size}")

    if precision_mode == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            # print("FP16 mode enabled.")
        else:
            print("FP16 not supported on this platform. Building in FP32 instead.")
            # Fallback to FP32 is implicit if FP16 flag isn't set.
    
    print(f"Building TensorRT engine ({precision_mode}, opt_batch={opt_batch_size})... This may take a few minutes.")
    serialized_engine = builder.build_serialized_network(network, config)

    if not serialized_engine:
        print("Failed to build the TensorRT engine.")
        return False

    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    print(f"TensorRT engine saved to {engine_file_path}")
    return True

class TensorRTEngineRunner:
    """Manages TensorRT engine loading, context, and inference."""
    def __init__(self, engine_file_path):
        self.engine_file_path = engine_file_path
        self.engine = None
        self.context = None
        self.stream = cuda.Stream()
        self.device_buffers = {} 
        self.device_buffer_sizes = {} # Store allocated byte sizes
        self.host_outputs_template = {} 
        self.input_tensor_info = None 
        self.output_tensors_info = []

        if not os.path.exists(engine_file_path):
            raise FileNotFoundError(f"Engine file not found: {engine_file_path}")
        
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if not self.engine:
            raise RuntimeError(f"Failed to deserialize CUDA engine from {engine_file_path}")
        
        self.context = self.engine.create_execution_context()
        if not self.context:
            raise RuntimeError("Failed to create TensorRT execution context.")

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            mode = self.engine.get_tensor_mode(name)
            tensor_info = {'name': name, 'dtype': dtype}
            if mode == trt.TensorIOMode.INPUT:
                if self.input_tensor_info:
                     raise ValueError("Demo supports only one input tensor.")
                self.input_tensor_info = tensor_info
            else:
                self.output_tensors_info.append(tensor_info)
        
        if not self.input_tensor_info:
            raise ValueError("No input tensor found in the engine.")

    def infer(self, input_tensor_host_numpy):
        """Performs inference using the loaded TensorRT engine."""
        actual_batch_size = input_tensor_host_numpy.shape[0]
        self.context.set_input_shape(self.input_tensor_info['name'], input_tensor_host_numpy.shape)

        input_name = self.input_tensor_info['name']
        input_nbytes = input_tensor_host_numpy.nbytes

        if input_name not in self.device_buffers or self.device_buffer_sizes.get(input_name) != input_nbytes:
            if input_name in self.device_buffers:
                self.device_buffers[input_name].free()
            self.device_buffers[input_name] = cuda.mem_alloc(input_nbytes)
            self.device_buffer_sizes[input_name] = input_nbytes # Store allocated size
        
        cuda.memcpy_htod_async(self.device_buffers[input_name], input_tensor_host_numpy, self.stream)
        self.context.set_tensor_address(input_name, int(self.device_buffers[input_name]))

        current_host_outputs = {}
        for out_info in self.output_tensors_info:
            output_name = out_info['name']
            output_shape = self.context.get_tensor_shape(output_name)
            if trt.volume(output_shape) < 0:
                 raise RuntimeError(f"Output tensor {output_name} shape {output_shape} is not fully defined.")
            
            output_dtype = out_info['dtype']
            host_buffer = np.empty(output_shape, dtype=output_dtype)
            output_nbytes = host_buffer.nbytes

            if output_name not in self.device_buffers or self.device_buffer_sizes.get(output_name) != output_nbytes:
                if output_name in self.device_buffers:
                    self.device_buffers[output_name].free()
                self.device_buffers[output_name] = cuda.mem_alloc(output_nbytes)
                self.device_buffer_sizes[output_name] = output_nbytes # Store allocated size

            self.context.set_tensor_address(output_name, int(self.device_buffers[output_name]))
            current_host_outputs[output_name] = host_buffer
        
        if not self.context.execute_async_v3(stream_handle=self.stream.handle):
            raise RuntimeError("TensorRT inference execution (execute_async_v3) failed.")
        
        for out_info in self.output_tensors_info:
            output_name = out_info['name']
            cuda.memcpy_dtoh_async(current_host_outputs[output_name], self.device_buffers[output_name], self.stream)
        
        self.stream.synchronize()
        return current_host_outputs

    def __del__(self):
        for _, mem_alloc in self.device_buffers.items():
            if mem_alloc:
                 try:
                    mem_alloc.free()
                 except cuda.LogicError:
                    pass 
        self.device_buffers.clear()
        self.device_buffer_sizes.clear()

_trt_engine_runners = {} # Cache for TensorRTEngineRunner instances

def get_trt_runner(engine_path):
    if engine_path not in _trt_engine_runners:
        print(f"Initializing TensorRT engine runner for: {os.path.basename(engine_path)}")
        _trt_engine_runners[engine_path] = TensorRTEngineRunner(engine_path)
    return _trt_engine_runners[engine_path]

def pytorch_inference_runner(model, input_tensor_pytorch, is_warmup=False):
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor_pytorch) # input_tensor_pytorch is already on cuda
    return outputs # Return PyTorch tensors

def trt_inference_runner(engine_path, input_tensor_pytorch, is_warmup=False):
    runner = get_trt_runner(engine_path)
    # Ensure input is numpy array of correct dtype for TRT engine
    input_numpy = input_tensor_pytorch.cpu().numpy().astype(runner.input_tensor_info['dtype'], copy=False)
    outputs_dict = runner.infer(input_numpy)
    # For benchmark consistency, we don't deeply inspect outputs during timing
    return outputs_dict


def benchmark_framework(name, inference_func, model_or_path, input_tensor_pytorch, num_runs=100):
    """Generic benchmark function."""
    print(f"\nBenchmarking {name} ({num_runs} runs, batch_size={input_tensor_pytorch.shape[0]})...")
    
    # Warm-up runs
    for _ in range(10): # Fixed 10 warm-up runs
        _ = inference_func(model_or_path, input_tensor_pytorch, is_warmup=True)
    torch.cuda.synchronize() # Ensure warmup is complete

    # Timed runs
    start_time = time.time()
    for _ in range(num_runs):
        _ = inference_func(model_or_path, input_tensor_pytorch)
    torch.cuda.synchronize() # Ensure all timed runs are complete
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_runs * 1000
    print(f"{name} average inference time: {avg_time_ms:.3f} ms")
    return avg_time_ms

def main():
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    # --- Configuration ---
    inference_batch_size = 8
    num_inference_runs = 100
    # For ONNX export, batch size is dynamic. For engine build, we specify a profile.
    min_trt_batch_size, opt_trt_batch_size, max_trt_batch_size = 1, inference_batch_size, 16
    
    # Optional: Path to a real image. If None, dummy data is used.
    # image_path = "path/to/your/image.jpg" 
    image_path = None 

    onnx_file_path = os.path.join(output_dir, "resnet50.onnx")
    trt_engine_fp32_path = os.path.join(output_dir, f"resnet50_fp32_b{opt_trt_batch_size}.engine")
    trt_engine_fp16_path = os.path.join(output_dir, f"resnet50_fp16_b{opt_trt_batch_size}.engine")

    # --- System and Model Setup ---
    if not torch.cuda.is_available():
        print("CUDA is not available. This demo requires a NVIDIA GPU.")
        return
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    print("Loading ResNet50 model from torchvision...")
    resnet50_pytorch = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).cuda().eval()
    print("Model loaded.")

    current_input_tensor_pytorch = load_or_create_input_tensor(image_path, inference_batch_size)

    # --- PyTorch Benchmark ---
    pytorch_time_ms = benchmark_framework("PyTorch", pytorch_inference_runner, 
                                          resnet50_pytorch, current_input_tensor_pytorch, 
                                          num_runs=num_inference_runs)

    # --- ONNX Export ---
    if not os.path.exists(onnx_file_path):
        print(f"\nExporting model to ONNX: {onnx_file_path}...")
        # Use batch size 1 for export, dynamic_axes will handle variability
        dummy_input_for_export = torch.randn(1, 3, 224, 224).cuda() 
        try:
            torch.onnx.export(resnet50_pytorch, dummy_input_for_export, onnx_file_path,
                              export_params=True, opset_version=12, do_constant_folding=True,
                              input_names=['input'], output_names=['output'], # Standard names
                              dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
            print("Model exported to ONNX successfully.")
        except Exception as e:
            print(f"Error during ONNX export: {e}. Skipping TensorRT steps.")
            return # Cannot proceed without ONNX model
    else:
        print(f"\nONNX file {onnx_file_path} already exists. Skipping export.")

    # --- TensorRT FP32 Benchmark ---
    print("\n--- TensorRT FP32 ---")
    trt_fp32_time_ms = float('inf')
    if not os.path.exists(trt_engine_fp32_path):
        build_engine(onnx_file_path, trt_engine_fp32_path, 'fp32',
                     min_trt_batch_size, opt_trt_batch_size, max_trt_batch_size)
    
    if os.path.exists(trt_engine_fp32_path):
        trt_fp32_time_ms = benchmark_framework("TensorRT FP32", trt_inference_runner,
                                               trt_engine_fp32_path, current_input_tensor_pytorch,
                                               num_runs=num_inference_runs)
    else:
        print(f"TensorRT FP32 engine not built or found. Skipping benchmark.")


    # --- TensorRT FP16 Benchmark ---
    print("\n--- TensorRT FP16 ---")
    trt_fp16_time_ms = float('inf')
    # Check FP16 support using a temporary builder
    temp_builder = trt.Builder(TRT_LOGGER)
    platform_has_fp16 = temp_builder.platform_has_fast_fp16
    del temp_builder

    if platform_has_fp16:
        if not os.path.exists(trt_engine_fp16_path):
            build_engine(onnx_file_path, trt_engine_fp16_path, 'fp16',
                         min_trt_batch_size, opt_trt_batch_size, max_trt_batch_size)
        
        if os.path.exists(trt_engine_fp16_path):
            trt_fp16_time_ms = benchmark_framework("TensorRT FP16", trt_inference_runner,
                                                   trt_engine_fp16_path, current_input_tensor_pytorch,
                                                   num_runs=num_inference_runs)
        else:
            print(f"TensorRT FP16 engine not built or found. Skipping benchmark.")
    else:
        print("FP16 not supported on this platform. Skipping FP16 benchmark.")
    
    # --- Summary ---
    print("\n--- Performance Summary ---")
    print(f"Input Batch Size: {inference_batch_size}")
    print(f"PyTorch: {pytorch_time_ms:.3f} ms")
    
    if trt_fp32_time_ms != float('inf'):
        speedup_fp32 = pytorch_time_ms / trt_fp32_time_ms if trt_fp32_time_ms > 0 else float('inf')
        print(f"TensorRT FP32: {trt_fp32_time_ms:.3f} ms (Speedup vs PyTorch: {speedup_fp32:.2f}x)")
    
    if trt_fp16_time_ms != float('inf'):
        speedup_fp16_pytorch = pytorch_time_ms / trt_fp16_time_ms if trt_fp16_time_ms > 0 else float('inf')
        print(f"TensorRT FP16: {trt_fp16_time_ms:.3f} ms (Speedup vs PyTorch: {speedup_fp16_pytorch:.2f}x)")
        if trt_fp32_time_ms != float('inf') and trt_fp32_time_ms > 0 and trt_fp16_time_ms > 0 :
            speedup_fp16_fp32 = trt_fp32_time_ms / trt_fp16_time_ms
            print(f"  (Speedup FP16 vs FP32: {speedup_fp16_fp32:.2f}x)")

if __name__ == "__main__":
    main()
    for runner in _trt_engine_runners.values():
        if hasattr(runner, '__del__'):
            runner.__del__()
    _trt_engine_runners.clear() 