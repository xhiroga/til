.PHONY: run upload

run:
	uv run OpenManus/main.py

.venv:
	uv sync

OpenManus:
	git -C $@ pull || git clone https://github.com/mannaandpoem/OpenManus $@

OpenManus/config/config.yaml: OpenManus
	cp config.yaml $@	
	echo "Edit $@"
