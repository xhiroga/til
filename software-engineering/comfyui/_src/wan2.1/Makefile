include .env	# COMFY_BASE_PATH, TEMP_PARENT_DIR

# Set default TEMP_PARENT_DIR to /tmp if it's empty
TEMP_PARENT_DIR ?= /tmp


.PHONY: run init

run:
	uv run ComfyUI/main.py --enable-cors-header --listen --port 11188 --fast --highvram

init: .venv ComfyUI $(COMFY_BASE_PATH)/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors $(COMFY_BASE_PATH)/vae/wan_2.1_vae.safetensors $(COMFY_BASE_PATH)/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors $(COMFY_BASE_PATH)/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors $(COMFY_BASE_PATH)/clip_vision/clip_vision_h.safetensors

.venv:
	uv sync

ComfyUI:
	git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI $@

$(TEMP_PARENT_DIR):
	mkdir -p $@

# https://comfyanonymous.github.io/ComfyUI_examples/wan/
$(COMFY_BASE_PATH)/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors: .venv $(TEMP_PARENT_DIR)
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d -p $(TEMP_PARENT_DIR)) && \
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors $@

$(COMFY_BASE_PATH)/vae/wan_2.1_vae.safetensors: .venv $(TEMP_PARENT_DIR)
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d -p $(TEMP_PARENT_DIR)) && \
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/vae/wan_2.1_vae.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/vae/wan_2.1_vae.safetensors $@

$(COMFY_BASE_PATH)/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors: .venv $(TEMP_PARENT_DIR)
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d -p $(TEMP_PARENT_DIR)) && \
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors $@

$(COMFY_BASE_PATH)/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors: .venv $(TEMP_PARENT_DIR)
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d -p $(TEMP_PARENT_DIR)) && \
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors $@

$(COMFY_BASE_PATH)/clip_vision/clip_vision_h.safetensors: .venv $(TEMP_PARENT_DIR)
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d -p $(TEMP_PARENT_DIR)) && \
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/clip_vision/clip_vision_h.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/clip_vision/clip_vision_h.safetensors $@
