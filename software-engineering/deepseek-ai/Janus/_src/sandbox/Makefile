app_januspro-1B: 1B app_januspro;

app_januspro: install
	cd ./Janus && .venv/bin/python demo/app_januspro.py

1B: Janus/demo/app_januspro.py
	sed -i 's/deepseek-ai\/Janus-Pro-7B/deepseek-ai\/Janus-Pro-1B/g' Janus/demo/app_januspro.py

7B: Janus/demo/app_januspro.py
	sed -i 's/deepseek-ai\/Janus-Pro-1B/deepseek-ai\/Janus-Pro-7B/g' Janus/demo/app_januspro.py

install: Janus/.venv
	cd Janus && .venv/bin/pip install -e .[gradio]

Janus:
	git -C $@ pull || git clone https://github.com/deepseek-ai/Janus $@

Janus/.venv: Janus
	# Python >= 3.8
	cd Janus && python3.12 -m venv .venv

Janus/demo/app_januspro.py: Janus
