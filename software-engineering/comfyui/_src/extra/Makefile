include .env	# TMP

TMP ?= /tmp

.PNOHY: init

init: .venv ComfyUI ComfyUI/custom_nodes/ComfyUI-layerdiffuse extra_model_paths.yaml $(COMFY_BASE_PATH)/checkpoints/animagine-xl-4.0.safetensors $(COMFY_BASE_PATH)/checkpoints/animagine-xl-4.0-opt.safetensors

.venv:
	uv sync

ComfyUI:
	git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI $@

extra_model_paths.yaml:
	uv run jinja2 extra_model_paths.yaml.j2 -D pwd=$$(pwd) > $@

$(TMP)/cagliostrolab/animagine-xl-4.0:
	uv run huggingface-cli download cagliostrolab/animagine-xl-4.0 --revision main --local-dir $@

# License: openrail++
ComfyUI/models/checkpoints/animagine-xl-4.0-opt.safetensors: $(TMP)/cagliostrolab/animagine-xl-4.0
	mv $(TMP)/cagliostrolab/animagine-xl-4.0/animagine-xl-4.0-opt.safetensors $@

# License: openrail++
ComfyUI/models/checkpoints/animagine-xl-4.0.safetensors: $(TMP)/cagliostrolab/animagine-xl-4.0
	mv $(TMP)/cagliostrolab/animagine-xl-4.0/animagine-xl-4.0.safetensors $@

# License: sdxl-license
ComfyUI/models/checkpoints/Illustrious-XL-v1.0.safetensors:
	uv run huggingface-cli download OnomaAIResearch/Illustrious-XL-v1.0 --revision main --local-dir $(TMP)/OnomaAIResearch/Illustrious-XL-v1.0
	mv $(TMP)/OnomaAIResearch/Illustrious-XL-v1.0/Illustrious-XL-v1.0.safetensors $@

# License: sdxl-license
# https://web.archive.org/web/20250321210743/https://huggingface.co/OnomaAIResearch/Illustrious-XL-v1.1/tree/main
ComfyUI/models/checkpoints/Illustrious-XL-v1.1.safetensors:
	uv run huggingface-cli download OnomaAIResearch/Illustrious-XL-v1.1 --revision main --local-dir $(TMP)/OnomaAIResearch/Illustrious-XL-v1.1
	mv $(TMP)/OnomaAIResearch/Illustrious-XL-v1.1/Illustrious-XL-v1.1.safetensors $@
