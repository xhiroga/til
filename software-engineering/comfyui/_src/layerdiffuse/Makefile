include .env	# COMFY_BASE_PATH, TEMP_PARENT_DIR

# Set default TEMP_PARENT_DIR to /tmp if it's empty
TEMP_PARENT_DIR ?= /tmp

.PHONY: run init

run:
	uv run ComfyUI/main.py --enable-cors-header --listen --port 11188 --extra-model-paths-config extra_model_paths.yaml --fast --highvram

init: .venv ComfyUI ComfyUI/custom_nodes/ComfyUI-layerdiffuse

.venv:
	uv sync

ComfyUI:
	git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI $@

ComfyUI/custom_nodes/ComfyUI-layerdiffuse: ComfyUI
	git -C $@ pull || git clone https://github.com/huchenlei/ComfyUI-layerdiffuse.git $@
