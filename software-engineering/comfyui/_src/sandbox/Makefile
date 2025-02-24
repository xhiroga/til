-include .env

CUDA_PATH_V12_4 := /usr/local/cuda-12.4
REMOTE_ENV := /til/software-engineering/comfyui/_src/sandbox/.env
REMOTE_EXTRA_MODEL_PATHS := /til/software-engineering/comfyui/_src/sandbox/extra_model_paths.yaml

.PNOHY: push-env push-extra_model_paths.yaml install run

# ComfyUI-3D-Pack requires CUDA v12.4
install: .env extra_model_paths.yaml ComfyUI ComfyUI/venv ComfyUI/custom_nodes/comfyui-manager ComfyUI/custom_nodes/ComfyUI-3D-Pack ComfyUI/user/default/workflows/ComfyUI-3D-Pack
	export PATH="$(CUDA_PATH_V12_4)/bin:$(PATH)" && export LIBGL_ALWAYS_SOFTWARE=1 && . ComfyUI/venv/bin/activate && cd ComfyUI && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
	export PATH="$(CUDA_PATH_V12_4)/bin:$(PATH)" && export LIBGL_ALWAYS_SOFTWARE=1 && . ComfyUI/venv/bin/activate && cd ComfyUI && pip install -r requirements.txt
	export PATH="$(CUDA_PATH_V12_4)/bin:$(PATH)" && export LIBGL_ALWAYS_SOFTWARE=1 && . ComfyUI/venv/bin/activate && cd ComfyUI && pip install "huggingface_hub[cli]"	
	export PATH="$(CUDA_PATH_V12_4)/bin:$(PATH)" && export LIBGL_ALWAYS_SOFTWARE=1 && . ComfyUI/venv/bin/activate && cd ComfyUI/custom_nodes/ComfyUI-3D-Pack && pip install -r requirements.txt
	# Takes 10 ~ 20 minutes
	export PATH="$(CUDA_PATH_V12_4)/bin:$(PATH)" && export LIBGL_ALWAYS_SOFTWARE=1 && . ComfyUI/venv/bin/activate && cd ComfyUI/custom_nodes/ComfyUI-3D-Pack && python install.py

run:
	PATH="$(CUDA_PATH_V12_4)/bin:$(PATH)" LIBGL_ALWAYS_SOFTWARE=1 ComfyUI/venv/bin/python ComfyUI/main.py --enable-cors-header --listen 0.0.0.0 --port 11188 --extra-model-paths-config extra_model_paths.yaml

push-env:
	aws ssm put-parameter \
		--name $(REMOTE_ENV) \
		--tier "Advanced" \
		--type SecureString \
		--value file://.env \
		--overwrite \
		--region ap-northeast-1

.env:
	aws ssm get-parameter \
		--name $(REMOTE_ENV) \
		--output text \
		--with-decryption \
		--region ap-northeast-1 \
		--query 'Parameter.Value' > .env

push-extra_model_paths.yaml:
	aws ssm put-parameter \
		--name $(REMOTE_EXTRA_MODEL_PATHS) \
		--tier "Advanced" \
		--type SecureString \
		--value file://extra_model_paths.yaml \
		--overwrite \
		--region ap-northeast-1

extra_model_paths.yaml:
	aws ssm get-parameter \
		--name $(REMOTE_EXTRA_MODEL_PATHS) \
		--output text \
		--with-decryption \
		--region ap-northeast-1 \
		--query 'Parameter.Value' > extra_model_paths.yaml

ComfyUI:
	git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI $@

ComfyUI/venv:
	cd ComfyUI && python3.12 -m venv venv

ComfyUI/custom_nodes/comfyui-manager: .env
	git -C $@ pull || git clone https://github.com/ltdrdata/ComfyUI-Manager $@

# v0.1.4 fails to install from ComfyUI-Manager
# Reqruires OpenGL 4.4 or higher
# Requires modification written in https://github.com/MrForExample/ComfyUI-3D-Pack/issues/425#issuecomment-2679837846
ComfyUI/custom_nodes/ComfyUI-3D-Pack: .env
	git -C $@ pull || git clone https://github.com/MrForExample/ComfyUI-3D-Pack.git $@

ComfyUI/user/default/workflows/ComfyUI-3D-Pack: .env
	git -C $@ pull || git clone --filter=blob:none --sparse https://github.com/MrForExample/ComfyUI-3D-Pack.git $@
	git -C $@ sparse-checkout set _Example_Workflows
