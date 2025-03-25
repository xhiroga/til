ComfyUI:
	git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI $@

extra_model_paths.yaml:
	uv run jinja2 extra_model_paths.yaml.j2 -D pwd=$$(pwd) > $@
