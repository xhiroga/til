include .env	# TMP

TMP ?= /tmp

run: FramePack FramePack-LoRAReady;
	uv run FramePack-LoRAReady/demo_gradio.py

FramePack:
	git -C $@ pull || git clone https://github.com/lllyasviel/FramePack $@

FramePack-LoRAReady:
	git -C $@ pull || git clone https://github.com/kohya-ss/FramePack-LoRAReady $@
