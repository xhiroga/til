include .env	# COMFY_BASE_PATH, TMP

# Set default TMP to /tmp if it's empty
TMP ?= /tmp

.PHONY: run init

run:
	uv run ComfyUI/main.py --enable-cors-header --listen --port 11188 --fast

init: .venv ComfyUI $(COMFY_BASE_PATH)/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors $(COMFY_BASE_PATH)/models/vae/wan_2.1_vae.safetensors $(COMFY_BASE_PATH)/models/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors $(COMFY_BASE_PATH)/models/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors $(COMFY_BASE_PATH)/models/clip_vision/clip_vision_h.safetensors

.venv:
	uv sync

ComfyUI:
	git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI $@

$(TMP):
	mkdir -p $@

# https://comfyanonymous.github.io/ComfyUI_examples/wan/

ComfyUI/user/default/workflows: ComfyUI/user/default/workflows/.installed
ComfyUI/user/default/workflows/.installed:
	ln -fsr workflows/* ComfyUI/user/default/workflows
	touch $@

## text_encoders
$(COMFY_BASE_PATH)/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors:
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors --revision main --local-dir $(TMP)/Comfy-Org/Wan_2.1_ComfyUI_repackaged
	mv $(TMP)/Comfy-Org/Wan_2.1_ComfyUI_repackaged/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors $@

## vae
$(COMFY_BASE_PATH)/models/vae/wan_2.1_vae.safetensors:
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/vae/wan_2.1_vae.safetensors --revision main --local-dir $(TMP)/Comfy-Org/Wan_2.1_ComfyUI_repackaged
	mv $(TMP)/Comfy-Org/Wan_2.1_ComfyUI_repackaged/split_files/vae/wan_2.1_vae.safetensors $@

## diffusion_models
$(COMFY_BASE_PATH)/models/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors:
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors --revision main --local-dir $(TMP)/Comfy-Org/Wan_2.1_ComfyUI_repackaged
	mv $(TMP)/Comfy-Org/Wan_2.1_ComfyUI_repackaged/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors $@

$(COMFY_BASE_PATH)/models/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors:
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors --revision main --local-dir $(TMP)/Comfy-Org/Wan_2.1_ComfyUI_repackaged
	mv $(TMP)/Comfy-Org/Wan_2.1_ComfyUI_repackaged/split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors $@

$(COMFY_BASE_PATH)/models/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors:
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors --revision main --local-dir $(TMP)/Comfy-Org/Wan_2.1_ComfyUI_repackaged
	mv $(TMP)/Comfy-Org/Wan_2.1_ComfyUI_repackaged/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors $@

## clip_vision
$(COMFY_BASE_PATH)/models/clip_vision/clip_vision_h.safetensors:
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/clip_vision/clip_vision_h.safetensors --revision main --local-dir $(TMP)/Comfy-Org/Wan_2.1_ComfyUI_repackaged
	mv $(TMP)/Comfy-Org/Wan_2.1_ComfyUI_repackaged/split_files/clip_vision/clip_vision_h.safetensors $@

## diffusion_models (control)
$(COMFY_BASE_PATH)/models/diffusion_models/Wan2.1-Fun-1.3B-Control.safetensors:
	uv run huggingface-cli download alibaba-pai/Wan2.1-Fun-1.3B-Control diffusion_pytorch_model.safetensors --revision main --local-dir $(TMP)/alibaba-pai/Wan2.1-Fun-1.3B-Control
	mv $(TMP)/alibaba-pai/Wan2.1-Fun-1.3B-Control/diffusion_pytorch_model.safetensors $@
