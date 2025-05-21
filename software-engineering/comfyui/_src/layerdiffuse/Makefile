.PHONY: run init

run:
	uv run ComfyUI/main.py --enable-cors-header --listen --port 11188 --extra-model-paths-config extra_model_paths.yaml --fast --highvram

init: .venv ComfyUI ComfyUI/custom_nodes/ComfyUI-layerdiffuse ComfyUI/custom_nodes/ComfyUI-WD14-Tagger

.venv:
	uv sync

ComfyUI:
	git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI $@

ComfyUI/custom_nodes/ComfyUI-layerdiffuse: ComfyUI
	git -C $@ pull || git clone https://github.com/huchenlei/ComfyUI-layerdiffuse.git $@

ComfyUI/custom_nodes/ComfyUI-WD14-Tagger: ComfyUI
	git -C $@ pull || git clone https://github.com/pythongosssss/ComfyUI-WD14-Tagger.git $@

ComfyUI/user/default/workflows/layerdiffuse:
	cp -r ComfyUI/custom_nodes/ComfyUI-layerdiffuse/example_workflows $@
