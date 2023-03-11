.PHONY: all clean;

all: "stable-diffusion-webui/webui-user.bat";

"stable-diffusion-webui/webui-user.bat":
	ln -f webui-user.bat ./stable-diffusion-webui/webui-user.bat

stable-diffusion-webui/models/Stable-Diffusion/v2-1_768-ema-pruned.ckpt:
	curl -o $@ https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt

stable-diffusion-webui/models/Stable-Diffusion/v2-1_768-ema-pruned.safetensors:
	curl -o $@ https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors

stable-diffusion-webui/models/Stable-Diffusion/v2-1_768-ema-pruned.yaml:
	curl -o $@ https://huggingface.co/webui/stable-diffusion-2-1/raw/main/v2-1_768-ema-pruned.yaml

stable-diffusion-webui/models/Stable-Diffusion/cool-japan-diffusion-v2-1-2.safetensors:
	curl -o $@ https://huggingface.co/aipicasso/cool-japan-diffusion-2-1-2/resolve/main/v2-1-2.safetensors

stable-diffusion-webui/models/Stable-Diffusion/cool-japan-diffusion-v2-1-2.yaml:
	curl -o $@ https://huggingface.co/aipicasso/cool-japan-diffusion-2-1-2/raw/main/v2-1-2.yaml
