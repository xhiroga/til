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
# .venv や $(TMP) を依存ファイルとして指定しない。
# 環境がおかしい時にカジュアルに.venv を初期化していると、そのたびに重みのダウンロードが走る書き方になってしまう。
$(COMFY_BASE_PATH)/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors:
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d -p $(TMP)) && \
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors $@

$(COMFY_BASE_PATH)/models/vae/wan_2.1_vae.safetensors:
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d -p $(TMP)) && \
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/vae/wan_2.1_vae.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/vae/wan_2.1_vae.safetensors $@

$(COMFY_BASE_PATH)/models/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors:
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d -p $(TMP)) && \
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors $@

$(COMFY_BASE_PATH)/models/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors:
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors --revision main --local-dir $(TMP)/Comfy-Org/Wan_2.1_ComfyUI_repackaged
	mv $(TMP)/Comfy-Org/Wan_2.1_ComfyUI_repackaged/split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors $@

$(COMFY_BASE_PATH)/models/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors:
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d -p $(TMP)) && \
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors $@

$(COMFY_BASE_PATH)/models/clip_vision/clip_vision_h.safetensors:
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d -p $(TMP)) && \
	uv run huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/clip_vision/clip_vision_h.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/clip_vision/clip_vision_h.safetensors $@
