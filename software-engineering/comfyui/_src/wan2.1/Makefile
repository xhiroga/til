.PNOHY:
	run

run:
	uv run ComfyUI/main.py --enable-cors-header --listen --port 11188 --fast --highvram

.venv:
	uv sync

ComfyUI:
	git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI $@
