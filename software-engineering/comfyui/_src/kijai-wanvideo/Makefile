include .env	# COMFY_BASE_PATH, TMP

TMP ?= /tmp

.PHONY: run init

run:
	uv run ComfyUI/main.py --enable-cors-header --listen --port 11188 --extra-model-paths-config extra_model_paths.yaml --fast --highvram

init: .venv ComfyUI ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper

.venv:
	uv sync

ComfyUI:
	git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI $@

# WIP!!! I guess workflows require resizing.
ComfyUI/custom_nodes/ComfyUI_essentials:
	# ignore numba and rembg due to llvmlite==0.36.0 does not support Python 3.12
	git -C $@ pull || git clone https://github.com/cubiq/ComfyUI_essentials $@

# https://github.com/kijai/ComfyUI-WanVideoWrapper
ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper:
	git -C $@ pull || git clone https://github.com/kijai/ComfyUI-WanVideoWrapper $@

ComfyUI/custom_nodes/ComfyUI-KJNodes:
	git -C $@ pull || git clone https://github.com/kijai/ComfyUI-KJNodes $@

ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite:
	git -C $@ pull || git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite $@

ComfyUI/user/default/workflows/ComfyUI-WanVideoWrapper:
	cp -r ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/example_workflows $@

$(TMP)/Kijai/WanVideo_comfy:
	uv run huggingface-cli download Kijai/WanVideo_comfy --revision main --local-dir $@

## text_encoders
$(COMFY_BASE_PATH)/models/text_encoders/open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors: $(TMP)/Kijai/WanVideo_comfy
	mv $(TMP)/Kijai/WanVideo_comfy/open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors $@

$(COMFY_BASE_PATH)/models/text_encoders/open-clip-xlm-roberta-large-vit-huge-14_visual_fp32.safetensors: $(TMP)/Kijai/WanVideo_comfy
	mv $(TMP)/Kijai/WanVideo_comfy/open-clip-xlm-roberta-large-vit-huge-14_visual_fp32.safetensors $@

$(COMFY_BASE_PATH)/models/text_encoders/umt5-xxl-enc-bf16.safetensors: $(TMP)/Kijai/WanVideo_comfy
	mv $(TMP)/Kijai/WanVideo_comfy/umt5-xxl-enc-bf16.safetensors $@

$(COMFY_BASE_PATH)/models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors: $(TMP)/Kijai/WanVideo_comfy
	mv $(TMP)/Kijai/WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors $@

## diffusion_models/i2v
$(COMFY_BASE_PATH)/models/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors: $(TMP)/Kijai/WanVideo_comfy
	mv $(TMP)/Kijai/WanVideo_comfy/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors $@

$(COMFY_BASE_PATH)/models/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e5m2.safetensors: $(TMP)/Kijai/WanVideo_comfy
	mv $(TMP)/Kijai/WanVideo_comfy/Wan2_1-I2V-14B-480P_fp8_e5m2.safetensors $@

$(COMFY_BASE_PATH)/models/diffusion_models/Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors: $(TMP)/Kijai/WanVideo_comfy
	mv $(TMP)/Kijai/WanVideo_comfy/Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors $@

$(COMFY_BASE_PATH)/models/diffusion_models/Wan2_1-I2V-14B-720P_fp8_e5m2.safetensors: $(TMP)/Kijai/WanVideo_comfy
	mv $(TMP)/Kijai/WanVideo_comfy/Wan2_1-I2V-14B-720P_fp8_e5m2.safetensors $@

## diffusion_models/t2v
$(COMFY_BASE_PATH)/models/diffusion_models/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors: $(TMP)/Kijai/WanVideo_comfy
	mv $(TMP)/Kijai/WanVideo_comfy/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors $@

$(COMFY_BASE_PATH)/models/diffusion_models/Wan2_1-T2V-14B_fp8_e5m2.safetensors: $(TMP)/Kijai/WanVideo_comfy
	mv $(TMP)/Kijai/WanVideo_comfy/Wan2_1-T2V-14B_fp8_e5m2.safetensors $@

$(COMFY_BASE_PATH)/models/diffusion_models/Wan2_1-T2V-1_3B_bf16.safetensors: $(TMP)/Kijai/WanVideo_comfy
	mv $(TMP)/Kijai/WanVideo_comfy/Wan2_1-T2V-1_3B_bf16.safetensors $@

$(COMFY_BASE_PATH)/models/diffusion_models/Wan2_1-T2V-1_3B_fp32.safetensors: $(TMP)/Kijai/WanVideo_comfy
	mv $(TMP)/Kijai/WanVideo_comfy/Wan2_1-T2V-1_3B_fp32.safetensors $@

$(COMFY_BASE_PATH)/models/diffusion_models/Wan2_1-T2V-1_3B_fp8_e4m3fn.safetensors : $(TMP)/Kijai/WanVideo_comfy
	mv $(TMP)/Kijai/WanVideo_comfy/Wan2_1-T2V-1_3B_fp8_e4m3fn.safetensors  $@

## vae
$(COMFY_BASE_PATH)/models/vae/Wan2_1_VAE_bf16.safetensors: $(TMP)/Kijai/WanVideo_comfy
	mv $(TMP)/Kijai/WanVideo_comfy/Wan2_1_VAE_bf16.safetensors $@

$(COMFY_BASE_PATH)/models/vae/Wan2_1_VAE_fp32.safetensors: $(TMP)/Kijai/WanVideo_comfy
	mv $(TMP)/Kijai/WanVideo_comfy/Wan2_1_VAE_fp32.safetensors $@

## loras
$(COMFY_BASE_PATH)/models/loras/wan2.1-1.3b-control-lora-depth-v0.1_comfy.safetensors:
	uv run huggingface-cli download spacepxl/Wan2.1-control-loras 1.3b/depth/wan2.1-1.3b-control-lora-depth-v0.1_comfy.safetensors --revision main --local-dir $(TMP)/spacepxl/Wan2.1-control-loras
	mv $(TMP)/spacepxl/Wan2.1-control-loras/1.3b/depth/wan2.1-1.3b-control-lora-depth-v0.1_comfy.safetensors $@

$(COMFY_BASE_PATH)/models/loras/wan2.1-1.3b-control-lora-tile-v0.2_comfy.safetensors:
	uv run huggingface-cli download spacepxl/Wan2.1-control-loras 1.3b/tile/wan2.1-1.3b-control-lora-tile-v0.2_comfy.safetensors --revision main --local-dir $(TMP)/spacepxl/Wan2.1-control-loras
	mv $(TMP)/spacepxl/Wan2.1-control-loras/1.3b/tile/wan2.1-1.3b-control-lora-tile-v0.2_comfy.safetensors $@
