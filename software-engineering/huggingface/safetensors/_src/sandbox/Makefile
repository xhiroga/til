.PHONY: diy

diy:
	uv run python diy.py models/pixel-art-medium-128-v0.1.safetensors

models/pixel-art-medium-128-v0.1.safetensors:
	uvx --from huggingface-hub huggingface-cli download nerijs/pixel-art-medium-128-v0.1 pixel-art-medium-128-v0.1.safetensors --local-dir $(dir $@)
