.PHONY: run

run:
	ComfyUI/.venv/bin/python ComfyUI/main.py --listen --port 11188 --fast

ComfyUI:
	git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI.git

ComfyUI/.venv: ComfyUI
	test -d $@ || python3 -m venv $@
	ComfyUI/.venv/bin/pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
	ComfyUI/.venv/bin/pip install -r ComfyUI/requirements.txt

ComfyUI/custom_nodes/ComfyUI-Manager: ComfyUI/.venv
	git -C $@ pull || git clone https://github.com/Comfy-Org/ComfyUI-Manager $@
	ComfyUI/.venv/bin/pip install -r $@/requirements.txt

init: ComfyUI ComfyUI/.venv ComfyUI/custom_nodes/ComfyUI-Manager
