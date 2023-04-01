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

stable-diffusion-webui/models/Stable-Diffusion/mitsua-diffusion-one.ckpt:
	curl -o $@ https://huggingface.co/Mitsua/mitsua-diffusion-one/resolve/main/mitsua-diffusion-one.ckpt

stable-diffusion-webui/models/Stable-Diffusion/mitsua-diffusion-one.yaml:
	curl -o $@ https://huggingface.co/Mitsua/mitsua-diffusion-one/raw/main/mitsua-diffusion-one.yaml

stable-diffusion-webui/models/Stable-Diffusion/trinart2_step95000.ckpt:
	curl -o $@ https://huggingface.co/naclbit/trinart_stable_diffusion_v2/resolve/main/trinart2_step95000.ckpt

stable-diffusion-webui/models/Stable-Diffusion/dreamlike-anime-1.0.safetensors:
	curl -o $@ https://huggingface.co/dreamlike-art/dreamlike-anime-1.0/blob/main/dreamlike-anime-1.0.safetensors

# 真夜ヰなちのLoRAファイル
stable-diffusion-webui/models/Lora/mayoi_nachi_v1.0.safetensors:
	curl -o $@.zip https://booth.pm/downloadables/3181017
	unzip $@.zip

stable-diffusion-webui/models/Lora/mayoi_nachi_v1.0.jpg:
	curl -o $@ https://s2.booth.pm/1ede2542-c07f-4cb8-b0f9-02953f79fa14/i/4622045/e9d716bc-3bd5-4244-b3da-35b420348cfc_base_resized.jpg
