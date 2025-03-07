include .env	# COMFY_BASE_PATH

.PHONY: run download

run: ComfyUI extra_model_paths.yaml download
	uv run ComfyUI/main.py --enable-cors-header --listen --port 11188 --extra-model-paths-config extra_model_paths.yaml

download: $(COMFY_BASE_PATH)/clip_vision/llava_llama3_vision.safetensors $(COMFY_BASE_PATH)/text_encoders/clip_l.safetensors $(COMFY_BASE_PATH)/text_encoders/llava_llama3_fp16.safetensors $(COMFY_BASE_PATH)/text_encoders/llava_llama3_fp8_scaled.safetensors $(COMFY_BASE_PATH)/vae/hunyuan_video_vae_bf16.safetensors $(COMFY_BASE_PATH)/diffusion_models/hunyuan_video_image_to_video_720p_bf16.safetensors

.venv:
	uv sync

ComfyUI:
	git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI $@

extra_model_paths.yaml:
	touch extra_model_paths.yaml

$(COMFY_BASE_PATH)/clip_vision/llava_llama3_vision.safetensors: .venv
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d) && \
	uv run huggingface-cli download Comfy-Org/HunyuanVideo_repackaged split_files/clip_vision/llava_llama3_vision.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/clip_vision/llava_llama3_vision.safetensors $@

$(COMFY_BASE_PATH)/text_encoders/clip_l.safetensors: .venv
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d) && \
	uv run huggingface-cli download Comfy-Org/HunyuanVideo_repackaged split_files/text_encoders/clip_l.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/text_encoders/clip_l.safetensors $@

$(COMFY_BASE_PATH)/text_encoders/llava_llama3_fp16.safetensors: .venv
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d) && \
	uv run huggingface-cli download Comfy-Org/HunyuanVideo_repackaged split_files/text_encoders/llava_llama3_fp16.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/text_encoders/llava_llama3_fp16.safetensors $@

$(COMFY_BASE_PATH)/text_encoders/llava_llama3_fp8_scaled.safetensors: .venv
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d) && \
	uv run huggingface-cli download Comfy-Org/HunyuanVideo_repackaged split_files/text_encoders/llava_llama3_fp8_scaled.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors $@

$(COMFY_BASE_PATH)/vae/hunyuan_video_vae_bf16.safetensors: .venv
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d) && \
	uv run huggingface-cli download Comfy-Org/HunyuanVideo_repackaged split_files/vae/hunyuan_video_vae_bf16.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/vae/hunyuan_video_vae_bf16.safetensors $@

$(COMFY_BASE_PATH)/vae/hunyuan_video_vae_bf16.safetensors: .venv
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d) && \
	uv run huggingface-cli download Comfy-Org/HunyuanVideo_repackaged split_files/vae/hunyuan_video_vae_bf16.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/vae/hunyuan_video_vae_bf16.safetensors $@

$(COMFY_BASE_PATH)/diffusion_models/hunyuan_video_image_to_video_720p_bf16.safetensors: .venv
	mkdir -p $(dir $@)
	TEMP_DIR=$$(mktemp -d) && \
	uv run huggingface-cli download Comfy-Org/HunyuanVideo_repackaged split_files/diffusion_models/hunyuan_video_image_to_video_720p_bf16.safetensors --revision main --local-dir $$TEMP_DIR && \
	mv $$TEMP_DIR/split_files/diffusion_models/hunyuan_video_image_to_video_720p_bf16.safetensors $@
