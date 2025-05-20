include .env	# MODEL_PATH, TMP

.PHONY: run init

run:
	uv run --extra $(EXTRA) ComfyUI/main.py --enable-cors-header --listen --port 11188 --extra-model-paths-config extra_model_paths.yaml --fast

.venv:
	uv sync --extra $(EXTRA)

ComfyUI:
	(git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI $@) && git -C $@ checkout v0.3.34

custom_nodes = \
	ComfyUI/custom_nodes/ComfyUI-FramePackWrapper_Plus \
	ComfyUI/custom_nodes/comfyui-get-meta \
	ComfyUI/custom_nodes/ComfyUI-KJNodes \
	ComfyUI/custom_nodes/ComfyUI-LogicUtils \
	ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite \
	ComfyUI/custom_nodes/ComfyUI_essentials
models = \
	$(MODEL_PATH)/diffusion_models/FramePackI2V_HY_fp8_e4m3fn.safetensors \
	$(MODEL_PATH)/diffusion_models/FramePackI2V_HY_bf16.safetensors \
	$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model.safetensors.index.json \
	$(MODEL_PATH)/clip_vision/sigclip_vision_patch14_384.safetensors \
	$(MODEL_PATH)/text_encoders/clip_l.safetensors \
	$(MODEL_PATH)/text_encoders/llava_llama3_fp16.safetensors \
	$(MODEL_PATH)/text_encoders/llava_llama3_fp8_scaled.safetensors \
	$(MODEL_PATH)/vae/hunyuan_video_vae_bf16.safetensors
f1_models = \
	$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model-00001-of-00003.safetensors \
	$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model-00002-of-00003.safetensors \
	$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model-00003-of-00003.safetensors \
	$(MODEL_PATH)/diffusion_models/FramePack_F1_I2V_HY_20250503.safetensors

init: .venv ComfyUI $(custom_nodes) $(models) ComfyUI/user/default/workflows/FramePackWrapper_Plus

# Custom Nodes
ComfyUI/custom_nodes/ComfyUI-FramePackWrapper_Plus: REPO = ShmuelRonen/ComfyUI-FramePackWrapper_Plus
ComfyUI/custom_nodes/ComfyUI-FramePackWrapper_Plus:COMMIT = 9a635c8caf9ecd301ade4ad09d75b1049ff6ba59
ComfyUI/custom_nodes/comfyui-get-meta: REPO = shinich39/comfyui-get-meta
ComfyUI/custom_nodes/comfyui-get-meta: COMMIT = b1af205fed09a3a4e2257f208fee2c53ad27a96e
ComfyUI/custom_nodes/ComfyUI-KJNodes: REPO = kijai/ComfyUI-KJNodes
ComfyUI/custom_nodes/ComfyUI-KJNodes: COMMIT = 5dcda71011870278c35d92ff77a677ed2e538f2d
ComfyUI/custom_nodes/ComfyUI-LogicUtils: REPO = aria1th/ComfyUI-LogicUtils
ComfyUI/custom_nodes/ComfyUI-LogicUtils: COMMIT = 60f8f1187c66ee544e09a85303e4140cf0bd0ff2
ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite: REPO = Kosinkadink/ComfyUI-VideoHelperSuite
ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite: COMMIT = a7ce59e381934733bfae03b1be029756d6ce936d
ComfyUI/custom_nodes/ComfyUI_essentials: REPO = cubiq/ComfyUI_essentials
ComfyUI/custom_nodes/ComfyUI_essentials: COMMIT = 9d9f4bedfc9f0321c19faf71855e228c93bd0dc9

$(custom_nodes): ComfyUI
	git -C $@ rev-parse HEAD || git clone https://github.com/$(REPO) $@
	git -C $@ checkout $(COMMIT)

# Models
## diffusion_models
$(MODEL_PATH)/diffusion_models/FramePackI2V_HY_fp8_e4m3fn.safetensors: REPO = Kijai/HunyuanVideo_comfy
$(MODEL_PATH)/diffusion_models/FramePackI2V_HY_fp8_e4m3fn.safetensors: FILE = FramePackI2V_HY_fp8_e4m3fn.safetensors
$(MODEL_PATH)/diffusion_models/FramePackI2V_HY_bf16.safetensors: REPO = Kijai/HunyuanVideo_comfy
$(MODEL_PATH)/diffusion_models/FramePackI2V_HY_bf16.safetensors: FILE = FramePackI2V_HY_bf16.safetensors
$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model-00001-of-00003.safetensors: REPO = lllyasviel/FramePackI2V_HY
$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model-00001-of-00003.safetensors: FILE = diffusion_pytorch_model-00001-of-00003.safetensors
$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model-00002-of-00003.safetensors: REPO = lllyasviel/FramePackI2V_HY
$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model-00002-of-00003.safetensors: FILE = diffusion_pytorch_model-00002-of-00003.safetensors
$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model-00003-of-00003.safetensors: REPO = lllyasviel/FramePackI2V_HY
$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model-00003-of-00003.safetensors: FILE = diffusion_pytorch_model-00003-of-00003.safetensors
$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model.safetensors.index.json: REPO = lllyasviel/FramePackI2V_HY
$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model.safetensors.index.json: FILE = diffusion_pytorch_model.safetensors.index.json

# clip_vision
$(MODEL_PATH)/clip_vision/sigclip_vision_patch14_384.safetensors: REPO = Comfy-Org/sigclip_vision_384
$(MODEL_PATH)/clip_vision/sigclip_vision_patch14_384.safetensors: FILE = sigclip_vision_patch14_384.safetensors

## text_encoders
$(MODEL_PATH)/text_encoders/clip_l.safetensors: REPO = Comfy-Org/HunyuanVideo_repackaged
$(MODEL_PATH)/text_encoders/clip_l.safetensors: FILE = split_files/text_encoders/clip_l.safetensors
$(MODEL_PATH)/text_encoders/llava_llama3_fp16.safetensors: REPO = Comfy-Org/HunyuanVideo_repackaged
$(MODEL_PATH)/text_encoders/llava_llama3_fp16.safetensors: FILE = split_files/text_encoders/llava_llama3_fp16.safetensors
$(MODEL_PATH)/text_encoders/llava_llama3_fp8_scaled.safetensors: REPO = Comfy-Org/HunyuanVideo_repackaged
$(MODEL_PATH)/text_encoders/llava_llama3_fp8_scaled.safetensors: FILE = split_files/text_encoders/llava_llama3_fp8_scaled.safetensors

## vae
$(MODEL_PATH)/vae/hunyuan_video_vae_bf16.safetensors: REPO = Comfy-Org/HunyuanVideo_repackaged
$(MODEL_PATH)/vae/hunyuan_video_vae_bf16.safetensors: FILE = split_files/vae/hunyuan_video_vae_bf16.safetensors

$(models):
	uv run huggingface-cli download $(REPO) $(FILE) --local-dir $(TMP)/$(REPO)
	mv $(TMP)/$(REPO)/$(FILE) $@

$(MODEL_PATH)/diffusion_models/FramePack_F1_I2V_HY_20250503.safetensors: $(MODEL_PATH)/diffusion_models/diffusion_pytorch_model-00001-of-00003.safetensors $(MODEL_PATH)/diffusion_models/diffusion_pytorch_model-00002-of-00003.safetensors $(MODEL_PATH)/diffusion_models/diffusion_pytorch_model-00003-of-00003.safetensors
	uv run merge.py --input_files \
		$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model-00001-of-00003.safetensors \
		$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model-00002-of-00003.safetensors \
		$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model-00003-of-00003.safetensors \
	--output $@

# Workflows
ComfyUI/user/default/workflows/FramePackWrapper_Plus: ComfyUI ComfyUI/custom_nodes/ComfyUI-FramePackWrapper_Plus
	mkdir -p $@
	cp ComfyUI/custom_nodes/ComfyUI-FramePackWrapper_Plus/example_workflows/* $@
