import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # Necessary for CUDA context initialization
import time
import os

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Dummy input for ONNX export and inference
# Ensure this matches the expected input dimensions of the model
# For dynamic batch size, we might need a representative batch size for export
# or define dynamic axes appropriately. Let's use batch size 1 for export.
dummy_input_for_export = torch.randn(1, 3, 224, 224).cuda()

def load_image_and_preprocess(image_path, batch_size=1):
    """Loads an image and preprocesses it for ResNet50, returning a batch."""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create a dummy tensor of the specified batch size for fallback
    # This dummy input should match the model's expected input channels, height, width
    dummy_batch = torch.randn(batch_size, 3, 224, 224).cuda()

    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}. Using dummy data instead.")
        return dummy_batch
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0) # Preprocess single image
        # Repeat the single image tensor to match the desired batch size
        batched_tensor = img_tensor.repeat(batch_size, 1, 1, 1).cuda()
        return batched_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}. Using dummy data instead.")
        return dummy_batch

def build_engine(onnx_file_path, engine_file_path, precision_mode='fp32', min_batch_size=1, opt_batch_size=8, max_batch_size=32):
    """Builds a TensorRT engine from an ONNX file with optimization profiles for dynamic batch sizes."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if not os.path.exists(onnx_file_path):
        print(f"ONNX file not found: {onnx_file_path}")
        return None

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("ONNX model parsed successfully.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Create an optimization profile for the dynamic input
    profile = builder.create_optimization_profile()
    # Assuming the input tensor name in ONNX model is 'input' (as defined in export)
    # and its shape is (batch_size, 3, 224, 224)
    input_name = network.get_input(0).name # Get input name from network
    profile.set_shape(input_name, 
                      (min_batch_size, 3, 224, 224),  # Min shape
                      (opt_batch_size, 3, 224, 224),  # Opt shape
                      (max_batch_size, 3, 224, 224))  # Max shape
    config.add_optimization_profile(profile)
    print(f"Optimization profile created for input '{input_name}' with min/opt/max batch sizes: {min_batch_size}/{opt_batch_size}/{max_batch_size}")

    if precision_mode == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled.")
        else:
            print("FP16 mode requested, but not supported on this platform. Falling back to FP32.")
    elif precision_mode == 'int8':
        print("INT8 mode requested, but not implemented in this basic demo. Falling back to FP32.")
        # For a real INT8 implementation, you would set up a calibrator here:
        # config.set_flag(trt.BuilderFlag.INT8)
        # config.int8_calibrator = ... (your calibrator instance)
    
    print(f"Building TensorRT engine for {precision_mode}... This may take a few minutes.")
    # Deprecated: builder.max_workspace_size = 1 << 30
    # serialized_engine = builder.build_cuda_engine(network) # old API
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build the engine.")
        return None

    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    print(f"TensorRT engine built and saved to {engine_file_path}")
    return serialized_engine

def infer_pytorch(model, input_tensor, num_runs=100):
    """Performs inference using PyTorch and measures average time."""
    model.eval()
    # Ensure input_tensor is on the correct device (CUDA)
    input_tensor = input_tensor.cuda()
    with torch.no_grad():
        # Warm-up runs
        for _ in range(10):
            _ = model(input_tensor)
        
        torch.cuda.synchronize() # Wait for all CUDA operations to complete
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(input_tensor)
        torch.cuda.synchronize() # Wait for all CUDA operations to complete
        end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_runs * 1000
    return avg_time_ms

def infer_trt(engine_file_path, input_tensor, num_runs=100):
    """Performs inference using a TensorRT engine and measures average time."""
    if not os.path.exists(engine_file_path):
        print(f"Engine file not found: {engine_file_path}")
        return float('inf')

    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    if not engine:
        print("Failed to deserialize CUDA engine.")
        return float('inf')

    # Ensure input_tensor is on CUDA and C-contiguous
    input_tensor_cuda = input_tensor.cuda()
    if not input_tensor_cuda.is_contiguous():
        input_tensor_cuda = input_tensor_cuda.contiguous()
    
    # Determine the actual batch size from the input tensor for this inference run
    actual_batch_size = input_tensor_cuda.shape[0]

    with engine.create_execution_context() as context:
        # Set the batch size for this specific inference call if using dynamic shapes
        # The binding dimensions must be set according to the input tensor's batch size
        # This is crucial for dynamic shapes.
        # We need to find the input tensor name first for setting its shape.
        input_tensor_name = ""
        # Use engine.num_io_tensors to get the number of I/O tensors
        for i in range(engine.num_io_tensors):
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT:
                input_tensor_name = engine.get_tensor_name(i)
                break
        
        if not input_tensor_name:
            print("Could not find input tensor in the engine.")
            return float('inf')

        # Set the dynamic shape for the input tensor
        context.set_input_shape(input_tensor_name, (actual_batch_size, 3, 224, 224))

        # Allocate input/output buffers and set tensor addresses for the context
        inputs, outputs, stream = [], [], cuda.Stream() # Removed bindings list
        # Use engine.num_io_tensors for iterating through I/O tensors
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            shape = context.get_tensor_shape(tensor_name)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            
            size = trt.volume(shape)
            if size < 0: 
                try:
                    profile_idx = 0 # Assuming first profile
                    # Get min, opt, max shapes for this tensor from the profile
                    # For dynamic shapes, engine.get_tensor_shape or context.get_tensor_shape might return -1 for dynamic dims.
                    # We need a concrete shape for allocation, often max from profile is used.
                    # Note: engine.get_profile_shape gives (min, opt, max) for an *input* tensor by its index in the profile.
                    # For I/O tensors generally, we might need to be more careful.
                    # Let's assume engine.get_tensor_profile_shape(tensor_name, profile_idx) is available and gives concrete shape.
                    # Or, more robustly, use the max shape from the profile used to build the engine for this tensor.
                    # This part can be tricky. engine.get_tensor_shape(name) should provide the shape for the current profile.
                    # The issue is if it's still dynamic *after* context.set_input_shape.
                    # For this demo, we'll try to get max profile shape if current shape is dynamic.
                    # This logic for size recovery needs to be robust.
                    # For an input tensor, its shape is now fixed by set_input_shape.
                    # For output tensors, their shapes should be inferable by the context.
                    # If an output shape is still dynamic here, it usually means it depends on data, not just input shape (e.g. NonMaxSuppression)
                    # For ResNet50, output shape should be fixed once input shape is fixed.
                    current_tensor_mode = engine.get_tensor_mode(tensor_name)
                    if current_tensor_mode == trt.TensorIOMode.OUTPUT:
                        print(f"Warning: Output tensor '{tensor_name}' has a dynamic shape {shape} after input shapes set. Trying to use max profile shape.")
                        # This is a common strategy: allocate outputs based on max possible batch size from profile
                        # Assuming 'input_tensor_name' is the main input that drives batch size
                        # And its profile shapes were (min_bs, C, H, W), (opt_bs, C, H, W), (max_bs, C, H, W)
                        # We need to find the max_batch_size used in profile for 'input_tensor_name'
                        # This is getting complicated. Let's assume context.get_tensor_shape(tensor_name) is sufficient for now
                        # If it's still -1, it means the output shape is truly data-dependent or setup is wrong.
                        # For ResNet50, output should be (batch_size, num_classes)
                        # If shape is like (X, 1000) and X is actual_batch_size, volume is fine.
                        # If shape is (-1, 1000), trt.volume is negative. 
                        # The context should have resolved this after set_input_shape.
                        # If not, it's an issue. Let's add a log if size < 0 persists for an output.
                        pass # Pass for now, rely on later check
                except Exception as e_profile:
                    print(f"Error trying to get profile shape for '{tensor_name}': {e_profile}")
                    # Fallback or error
            
            if size < 0: # Still negative, implies an issue
                print(f"Error: Could not determine a valid size for tensor '{tensor_name}' with shape {shape}. This will likely fail.")
                # For ResNet50, output is (batch, num_classes). If batch is resolved, size should be positive.
                # If num_classes part is dynamic, that's a different problem.
                # Fallback to a large buffer is bad. Better to fail.
                return float('inf') 
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Set tensor address for the execution context
            context.set_tensor_address(tensor_name, int(device_mem))

            tensor_mode = engine.get_tensor_mode(tensor_name)
            if tensor_mode == trt.TensorIOMode.INPUT:
                inputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem, 'shape': shape, 'idx': i})
            elif tensor_mode == trt.TensorIOMode.OUTPUT:
                outputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem, 'shape': shape, 'idx': i})
            else: # Should not happen for typical IN/OUT tensors
                print(f"Warning: Tensor '{tensor_name}' is neither INPUT nor OUTPUT.")

        
        if not inputs:
            print("No input tensors found in the engine.")
            return float('inf')

        # Copy input data to device
        # Ensure the input_numpy matches the expected flattened size for the current batch
        input_numpy = input_tensor_cuda.cpu().numpy().ravel()

        # Ensure the host buffer and numpy array sizes match for np.copyto
        expected_elements = trt.volume(inputs[0]['shape'])
        if inputs[0]['host'].size != expected_elements or input_numpy.size != expected_elements:
            print(f"Warning: Mismatch in expected elements ({expected_elements}), host buffer size ({inputs[0]['host'].size}), or numpy array size ({input_numpy.size}).")
            # This indicates a potential issue with how shapes are handled or how buffers are allocated.
            # For robust handling, one might need to reallocate or ensure sizes always match.
            # As a fallback, try to copy if the number of bytes match, otherwise error or resize.
            # For now, we will proceed assuming the ravelled numpy array can be copied if its nbytes matches host buffer nbytes.
            if inputs[0]['host'].nbytes != input_numpy.nbytes:
                print(f"Critical Error: Host buffer nbytes {inputs[0]['host'].nbytes} and input numpy nbytes {input_numpy.nbytes} mismatch. Cannot copy.")
                # Potentially return float('inf') or raise an error
                # Fallback to direct copy, hoping for the best if types allow, but this is risky.
                # np.copyto(inputs[0]['host'], input_numpy.astype(inputs[0]['host'].dtype)) # Risky if sizes don't match element-wise
                # A safer approach for this demo if sizes don't match but should:
                if input_numpy.size == expected_elements: # if numpy is correct, but host buffer might be from max_profile
                    temp_host_view = inputs[0]['host'].reshape(input_numpy.shape)
                    np.copyto(temp_host_view, input_numpy.astype(temp_host_view.dtype, copy=False))
                else:
                    print("Cannot resolve size mismatch for np.copyto. Skipping copy.")
                    # This part needs more robust error handling or logic for mismatched sizes.
            else: # nbytes match, attempt copy with dtype conversion
                np.copyto(inputs[0]['host'].reshape(input_numpy.shape), input_numpy.astype(inputs[0]['host'].dtype, copy=False))

        else: # Sizes match
            np.copyto(inputs[0]['host'], input_numpy.astype(inputs[0]['host'].dtype, copy=False))

        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)

        # Warm-up runs
        for _ in range(10):
            # context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            if not context.execute_async_v3(stream_handle=stream.handle):
                print("Error during TensorRT warm-up inference execution (execute_async_v3).")
                return float('inf')
            stream.synchronize()

        # Inference execution and timing
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream) 
        stream.synchronize() 
        
        start_time = time.time()
        for _ in range(num_runs):
            # context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            if not context.execute_async_v3(stream_handle=stream.handle):
                print("Error during TensorRT timed inference execution (execute_async_v3).")
                # Clean up allocated memory if possible before returning
                # For simplicity in demo, just return inf
                return float('inf')
        stream.synchronize() 
        end_time = time.time()
        
        # Optionally, copy output data back to host
        # for out_buf in outputs:
        #     cuda.memcpy_dtoh_async(out_buf['host'], out_buf['device'], stream)
        # stream.synchronize()
        # output_data = {out['name']: out['host'].reshape(out['shape']) for out in outputs}

    avg_time_ms = (end_time - start_time) / num_runs * 1000
    return avg_time_ms

def main():
    # --- Configuration ---
    # For a real image, replace this with the path to your image
    # e.g., image_path = "path/to/your/image.jpg" 
    # For this demo, we'll use a dummy input for consistency in performance measurement.
    # If you want to test with a real image, download one and set:
    # input_data_source = "path_to_sample_image.jpg" 
    # And potentially adjust batch_size below.
    input_data_source = None # Will use dummy data generated by load_image_and_preprocess
    # Define a batch size for inference. This will be used for PyTorch and TensorRT.
    # The TensorRT engine will be built to support a range including this batch_size.
    inference_batch_size = 8 # Example batch size for inference runs
    
    onnx_file_path = "resnet50.onnx"
    trt_engine_fp32_path = f"resnet50_fp32_b{inference_batch_size}.engine"
    trt_engine_fp16_path = f"resnet50_fp16_b{inference_batch_size}.engine"
    num_inference_runs = 100 # Number of runs for averaging inference time

    # TensorRT engine build parameters (can be different from inference_batch_size if needed)
    min_trt_batch_size = 1
    opt_trt_batch_size = 8
    max_trt_batch_size = 16 # Max batch size the engine will support

    # --- CUDA Device Check ---
    if not torch.cuda.is_available():
        print("CUDA is not available. This demo requires a NVIDIA GPU.")
        return
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # --- Prepare Model and Input ---
    print("Loading ResNet50 model from torchvision...")
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).cuda().eval()
    print("Model loaded.")

    print(f"Preparing input tensor with batch size {inference_batch_size}...")
    if isinstance(input_data_source, str): # If a path is given
        current_input_tensor = load_image_and_preprocess(input_data_source, batch_size=inference_batch_size)
    else: # If dummy_input is used directly or path is None
        # Use a globally defined dummy_input if load_image_and_preprocess handles None by returning global dummy
        # Or generate one here. load_image_and_preprocess now creates dummy if path is invalid.
        print(f"Using dummy data for input tensor with batch size {inference_batch_size}.")
        # The dummy_input_for_export is batch size 1. We need one for inference_batch_size.
        current_input_tensor = torch.randn(inference_batch_size, 3, 224, 224).cuda()

    if current_input_tensor is None or current_input_tensor.shape[0] != inference_batch_size:
        print(f"Failed to load or prepare input tensor with batch size {inference_batch_size}. Exiting.")
        return

    # --- PyTorch Inference ---
    print(f"\nRunning inference with PyTorch ({num_inference_runs} runs)...")
    pytorch_time_ms = infer_pytorch(resnet50, current_input_tensor, num_runs=num_inference_runs)
    print(f"PyTorch average inference time: {pytorch_time_ms:.3f} ms")

    # --- ONNX Export ---
    if not os.path.exists(onnx_file_path):
        print(f"\nExporting model to ONNX: {onnx_file_path}...")
        try:
            # For ONNX export with dynamic batch size, ensure the dummy input reflects that
            # The dummy_input_for_export (batch_size 1) is fine if dynamic_axes are set correctly.
            torch.onnx.export(resnet50,
                              dummy_input_for_export, # Use BS=1 for export, dynamic_axes handles variability
                              onnx_file_path,
                              export_params=True,
                              opset_version=12, # Increased opset for better compatibility
                              do_constant_folding=True,
                              input_names=['input'],
                              output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size'},
                                            'output': {0: 'batch_size'}})
            print("Model exported to ONNX successfully.")
        except Exception as e:
            print(f"Error during ONNX export: {e}")
            print("Skipping TensorRT steps due to ONNX export failure.")
            return # Exit if ONNX export fails
    else:
        print(f"\nONNX file {onnx_file_path} already exists. Skipping export.")

    # --- TensorRT FP32 Engine ---
    print("\n--- TensorRT FP32 ---")
    if not os.path.exists(trt_engine_fp32_path):
        build_engine(onnx_file_path, trt_engine_fp32_path, precision_mode='fp32', 
                     min_batch_size=min_trt_batch_size, 
                     opt_batch_size=opt_trt_batch_size, 
                     max_batch_size=max_trt_batch_size)
    
    trt_fp32_time_ms = infer_trt(trt_engine_fp32_path, current_input_tensor, num_runs=num_inference_runs)
    if trt_fp32_time_ms != float('inf'):
        print(f"TensorRT FP32 average inference time: {trt_fp32_time_ms:.3f} ms")
    else:
        print(f"TensorRT FP32 inference failed or engine not found.")

    # --- TensorRT FP16 Engine ---
    print("\n--- TensorRT FP16 ---")
    # Check FP16 support before attempting to build or infer
    builder_fp16_check = trt.Builder(TRT_LOGGER)
    platform_has_fp16 = builder_fp16_check.platform_has_fast_fp16
    del builder_fp16_check

    if platform_has_fp16:
        if not os.path.exists(trt_engine_fp16_path):
            build_engine(onnx_file_path, trt_engine_fp16_path, precision_mode='fp16',
                         min_batch_size=min_trt_batch_size, 
                         opt_batch_size=opt_trt_batch_size, 
                         max_batch_size=max_trt_batch_size)
        
        trt_fp16_time_ms = infer_trt(trt_engine_fp16_path, current_input_tensor, num_runs=num_inference_runs)
        if trt_fp16_time_ms != float('inf'):
            print(f"TensorRT FP16 average inference time: {trt_fp16_time_ms:.3f} ms")
        else:
            print(f"TensorRT FP16 inference failed or engine not found.")
    else:
        print("FP16 not supported on this platform. Skipping FP16 engine build and inference.")
        trt_fp16_time_ms = float('inf') # Set to infinity if not supported

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"PyTorch: {pytorch_time_ms:.3f} ms")
    if trt_fp32_time_ms != float('inf') and pytorch_time_ms > 0 : # Ensure PyTorch time is valid
        speedup_fp32 = pytorch_time_ms / trt_fp32_time_ms if trt_fp32_time_ms > 0 else float('inf')
        print(f"TensorRT FP32: {trt_fp32_time_ms:.3f} ms (Speedup vs PyTorch: {speedup_fp32:.2f}x)")
    
    if trt_fp16_time_ms != float('inf') and platform_has_fp16 and pytorch_time_ms > 0: # Ensure PyTorch time is valid
        speedup_fp16_vs_pytorch = pytorch_time_ms / trt_fp16_time_ms if trt_fp16_time_ms > 0 else float('inf')
        print(f"TensorRT FP16: {trt_fp16_time_ms:.3f} ms (Speedup vs PyTorch: {speedup_fp16_vs_pytorch:.2f}x)")
        if trt_fp32_time_ms != float('inf') and trt_fp32_time_ms > 0 and trt_fp16_time_ms > 0 :
            speedup_fp16_vs_fp32 = trt_fp32_time_ms / trt_fp16_time_ms
            print(f"TensorRT FP16 Speedup vs TensorRT FP32: {speedup_fp16_vs_fp32:.2f}x")

if __name__ == "__main__":
    main() 