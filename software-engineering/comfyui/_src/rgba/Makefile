include .env	# COMFY_BASE_PATH

.PHONY: run

run: .venv ComfyUI extra_model_paths.yaml ComfyUI/custom_nodes/comfyui-manager ComfyUI/custom_nodes/ComfyUI-layerdiffuse
	uv run ComfyUI/main.py --enable-cors-header --listen --port 11188 --extra-model-paths-config extra_model_paths.yaml

.venv:
	uv sync

extra_model_paths.yaml:
	touch extra_model_paths.yaml

ComfyUI:
	git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI $@

ComfyUI/custom_nodes/comfyui-manager: ComfyUI
	git -C $@ pull || git clone https://github.com/ltdrdata/ComfyUI-Manager $@

ComfyUI/custom_nodes/ComfyUI-layerdiffuse: ComfyUI
	git -C $@ pull || git clone https://github.com/huchenlei/ComfyUI-layerdiffuse.git $@
