include .env	# DATASET, TMP

DATASET ?= https://huggingface.co/datasets/******/******
TMP ?= /tmp	# クラウドの場合はマウントしているボリューム配下のパスにすること

.PHONY: run cache huggingface-cli-login models

run: musubi-tuner/.venv musubi-tuner/dataset.toml musubi-tuner/metadata.jsonl cache models wandb-login
# --image_encoder は required=True なので、--config_file とは別に必要
	uv --directory musubi-tuner --project musubi-tuner run \
		accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
			fpack_train_network.py \
				--image_encoder ../models/image_encoder/model.safetensors \
				--config_file ../config.toml


cache: musubi-tuner/.venv musubi-tuner/dataset.toml musubi-tuner/metadata.jsonl models
	uv --directory musubi-tuner --project musubi-tuner run fpack_cache_latents.py \
		--dataset_config dataset.toml \
		--vae ../models/framepack/vae/diffusion_pytorch_model.safetensors \
		--image_encoder ../models/image_encoder/model.safetensors \
		--vae_chunk_size 32 \
		--vae_spatial_tile_sample_min_size 256 \
		--f1

	uv --directory musubi-tuner --project musubi-tuner run fpack_cache_text_encoder_outputs.py \
		--dataset_config dataset.toml \
		--text_encoder1 ../models/text_encoder/model-00001-of-00004.safetensors \
		--text_encoder2 ../models/text_encoder_2/model.safetensors \
		--batch_size 16

# 慣れるまでは VAE の tiling 設定は deeppwiki に従う方針。
# https://deepwiki.com/search/vaetiling-vaespatialtilesample_d53d814c-27e9-405f-a43f-111b316047a3

.venv:
	uv sync

musubi-tuner:
	git -C $@ pull || git clone https://github.com/kohya-ss/musubi-tuner $@

musubi-tuner/.venv: musubi-tuner
	uv --project musubi-tuner sync

huggingface-cli-login: .venv
	if uv run huggingface-cli whoami | grep -q 'Not logged in'; then uv run huggingface-cli login; fi

wandb-login: .venv
# musubi-tuner 自身の pyproject.toml に手を入れたくなかったが、uv pip install だと uv sync が走るたびに wandb が消える。addしかない。
	uv --project musubi-tuner add wandb
	uv --project musubi-tuner run wandb login

datasets/musubi-tuner-fp.toml: dataset
dataset: huggingface-cli-login
	uv run huggingface-cli download $(DATASET) --repo-type dataset --local-dir $@

musubi-tuner/dataset.toml: .venv musubi-tuner dataset
# LoRA学習なので単一のデータセットと toml ファイルをまとめて管理したほうが効率が良いと判断
# ただし、toml 中の jsonl ファイルのパスが相対パスの場合、musubi-tuner が基準となる
	cp dataset/dataset.toml $@

musubi-tuner/metadata.jsonl: .venv musubi-tuner dataset
	uv run jinja2 -D dir=$$(realpath dataset) dataset/metadata.jsonl.j2 > $@

models: models/framepack/dit/diffusion_pytorch_model-00001-of-00003.safetensors models/framepack/dit/diffusion_pytorch_model-00002-of-00003.safetensors models/framepack/dit/diffusion_pytorch_model-00003-of-00003.safetensors models/framepack/vae/diffusion_pytorch_model.safetensors models/text_encoder/model-00001-of-00004.safetensors models/text_encoder/model-00002-of-00004.safetensors models/text_encoder/model-00003-of-00004.safetensors models/text_encoder/model-00004-of-00004.safetensors models/text_encoder_2/model.safetensors models/image_encoder/model.safetensors

models/framepack/dit/diffusion_pytorch_model-00001-of-00003.safetensors: models/framepack/dit
models/framepack/dit/diffusion_pytorch_model-00002-of-00003.safetensors: models/framepack/dit
models/framepack/dit/diffusion_pytorch_model-00003-of-00003.safetensors: models/framepack/dit
models/framepack/dit: huggingface-cli-login
	uv run huggingface-cli download lllyasviel/FramePack_F1_I2V_HY_20250503 --local-dir $@

models/framepack/vae/diffusion_pytorch_model.safetensors: huggingface-cli-login
# 毎回忘れるが、huggingface-cli の download は 階層構造ごとダウンロードしてくれる点に注意。この場合は local-dir 内部に vae フォルダが作られる
	uv run huggingface-cli download hunyuanvideo-community/HunyuanVideo vae/diffusion_pytorch_model.safetensors --local-dir models/framepack

models/text_encoder/model-00001-of-00004.safetensors: models/text_encoder
models/text_encoder/model-00002-of-00004.safetensors: models/text_encoder
models/text_encoder/model-00003-of-00004.safetensors: models/text_encoder
models/text_encoder/model-00004-of-00004.safetensors: models/text_encoder
models/text_encoder: huggingface-cli-login
	uv run huggingface-cli download hunyuanvideo-community/HunyuanVideo text_encoder/model-00001-of-00004.safetensors text_encoder/model-00002-of-00004.safetensors text_encoder/model-00003-of-00004.safetensors text_encoder/model-00004-of-00004.safetensors --local-dir $(@D)

models/text_encoder_2/model.safetensors: huggingface-cli-login
	uv run huggingface-cli download hunyuanvideo-community/HunyuanVideo text_encoder_2/model.safetensors --local-dir models

models/image_encoder/model.safetensors: huggingface-cli-login
	uv run huggingface-cli download lllyasviel/flux_redux_bfl image_encoder/model.safetensors --local-dir models
