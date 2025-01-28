app_januspro: install
	cd ./Janus && .venv/bin/python demo/app_januspro.py

install: Janus/.venv
	cd Janus && .venv/bin/pip install -e .[gradio]

Janus:
	git -C $@ pull || git clone https://github.com/deepseek-ai/Janus $@

Janus/.venv: Janus
	# Python >= 3.8
	cd Janus && python3.12 -m venv .venv
