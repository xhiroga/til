include .env	# MODEL_PATH, TMP

.PHONY: run init

run:
	uv run ComfyUI/main.py --enable-cors-header --listen --port 11188 --extra-model-paths-config extra_model_paths.yaml --fast

.venv:
	uv sync

ComfyUI:
	(git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI $@) && git -C $@ checkout v0.3.34

models = \
	$(MODEL_PATH)/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
	$(MODEL_PATH)/vae/wan_2.1_vae.safetensors \
	$(MODEL_PATH)/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors \
	$(MODEL_PATH)/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors \
	$(MODEL_PATH)/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors \
	$(MODEL_PATH)/clip_vision/clip_vision_h.safetensors \
	$(MODEL_PATH)/diffusion_models/Wan2.1-Fun-1.3B-Control.safetensors

$(MODEL_PATH)/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors: REPO = Comfy-Org/Wan_2.1_ComfyUI_repackaged
$(MODEL_PATH)/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors: FILE = split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors

$(MODEL_PATH)/vae/wan_2.1_vae.safetensors: REPO = Comfy-Org/Wan_2.1_ComfyUI_repackaged
$(MODEL_PATH)/vae/wan_2.1_vae.safetensors: FILE = split_files/vae/wan_2.1_vae.safetensors

$(MODEL_PATH)/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors: REPO = Comfy-Org/Wan_2.1_ComfyUI_repackaged
$(MODEL_PATH)/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors: FILE = split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors

$(MODEL_PATH)/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors: REPO = Comfy-Org/Wan_2.1_ComfyUI_repackaged
$(MODEL_PATH)/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors: FILE = split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors

$(MODEL_PATH)/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors: REPO = Comfy-Org/Wan_2.1_ComfyUI_repackaged
$(MODEL_PATH)/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors: FILE = split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors

$(MODEL_PATH)/clip_vision/clip_vision_h.safetensors: REPO = Comfy-Org/Wan_2.1_ComfyUI_repackaged
$(MODEL_PATH)/clip_vision/clip_vision_h.safetensors: FILE = split_files/clip_vision/clip_vision_h.safetensors

$(MODEL_PATH)/diffusion_models/Wan2.1-Fun-1.3B-Control.safetensors: REPO = alibaba-pai/Wan2.1-Fun-1.3B-Control
$(MODEL_PATH)/diffusion_models/Wan2.1-Fun-1.3B-Control.safetensors: FILE = diffusion_pytorch_model.safetensors

init: .venv ComfyUI $(models) ComfyUI/user/default/workflows/.installed

$(models):
	uv run huggingface-cli download $(REPO) $(FILE) --local-dir $(TMP)/$(REPO)
	mv $(TMP)/$(REPO)/$(FILE) $@

# https://comfyanonymous.github.io/ComfyUI_examples/wan/
ComfyUI/user/default/workflows: ComfyUI/user/default/workflows/.installed
ComfyUI/user/default/workflows/.installed: ComfyUI
	mkdir -p $(dir $@)
	cp workflows/* $(dir $@)
