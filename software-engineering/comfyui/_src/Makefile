include .env	# MODEL_PATH, TMP

extra_model_paths.yaml:
	uvx --from jinja2-cli jinja2 extra_model_paths.yaml.j2 -D model_path=$$(MODEL_PATH) > $@

models = \
	$(MODEL_PATH)/checkpoints/animagine-xl-4.0-opt.safetensors \
	$(MODEL_PATH)/checkpoints/animagine-xl-4.0.safetensors \
	$(MODEL_PATH)/checkpoints/Illustrious-XL-v1.0.safetensors \
	$(MODEL_PATH)/checkpoints/Illustrious-XL-v1.1.safetensors \
	$(MODEL_PATH)/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors \
	$(MODEL_PATH)/loras/pixel-art-xl.safetensors \
	$(MODEL_PATH)/controlnet/sai_xl_depth_256lora.safetensors

# License: openrail++
$(MODEL_PATH)/checkpoints/animagine-xl-4.0-opt.safetensors: REPO = cagliostrolab/animagine-xl-4.0
$(MODEL_PATH)/checkpoints/animagine-xl-4.0-opt.safetensors: FILE = animagine-xl-4.0-opt.safetensors

# License: openrail++
$(MODEL_PATH)/checkpoints/animagine-xl-4.0.safetensors: REPO = cagliostrolab/animagine-xl-4.0
$(MODEL_PATH)/checkpoints/animagine-xl-4.0.safetensors: FILE = animagine-xl-4.0.safetensors

# License: sdxl-license
$(MODEL_PATH)/checkpoints/Illustrious-XL-v1.0.safetensors: REPO = OnomaAIResearch/Illustrious-XL-v1.0
$(MODEL_PATH)/checkpoints/Illustrious-XL-v1.0.safetensors: FILE = Illustrious-XL-v1.0.safetensors

# License: sdxl-license
# https://web.archive.org/web/20250321210743/https://huggingface.co/OnomaAIResearch/Illustrious-XL-v1.1/tree/main
$(MODEL_PATH)/checkpoints/Illustrious-XL-v1.1.safetensors: REPO = OnomaAIResearch/Illustrious-XL-v1.1
$(MODEL_PATH)/checkpoints/Illustrious-XL-v1.1.safetensors: FILE = Illustrious-XL-v1.1.safetensors

# License: private
$(MODEL_PATH)/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors: REPO = xhiroga/private-models
$(MODEL_PATH)/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors: FILE = RunDiffusion/juggernautXL_juggXIByRundiffusion.safetensors

# License: creativeml-openrail-m
$(MODEL_PATH)/loras/pixel-art-xl.safetensors: REPO = nerijs/pixel-art-xl
$(MODEL_PATH)/loras/pixel-art-xl.safetensors: FILE = pixel-art-xl.safetensors

# Controlnet
$(MODEL_PATH)/controlnet/sai_xl_depth_256lora.safetensors: REPO = lllyasviel/sd_control_collection
$(MODEL_PATH)/controlnet/sai_xl_depth_256lora.safetensors: FILE = sai_xl_depth_256lora.safetensors

$(models):
	uvx --from huggingface-hub huggingface-cli download $(REPO) $(FILE) --local-dir $(TMP)/$(REPO)
	mkdir -p $(dir $@)
	mv $(TMP)/$(REPO)/$(FILE) $@
