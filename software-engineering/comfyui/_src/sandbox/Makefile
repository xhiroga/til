CURRENT_DIR := $(realpath .)
CUDA_PATH_V12_4 := /usr/local/cuda-12.4
REMOTE_EXTRA_MODEL_PATHS := /til/software-engineering/comfyui/_src/sandbox/extra_model_paths.yaml

.PNOHY: push-env push-extra_model_paths.yaml install run

# ComfyUI-3D-Pack requires CUDA v12.4
install: .env extra_model_paths.yaml ComfyUI ComfyUI/venv ComfyUI/custom_nodes/comfyui-manager ComfyUI/custom_nodes/ComfyUI-3D-Pack ComfyUI/custom_nodes/ComfyUI-layerdiffuse ComfyUI/user/default/workflows/ComfyUI-3D-Pack
	export PATH="$(CUDA_PATH_V12_4)/bin:$(PATH)" && export LIBGL_ALWAYS_SOFTWARE=1 && . ComfyUI/venv/bin/activate && cd ComfyUI && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
	export PATH="$(CUDA_PATH_V12_4)/bin:$(PATH)" && export LIBGL_ALWAYS_SOFTWARE=1 && . ComfyUI/venv/bin/activate && cd ComfyUI && pip install -r requirements.txt
	export PATH="$(CUDA_PATH_V12_4)/bin:$(PATH)" && export LIBGL_ALWAYS_SOFTWARE=1 && . ComfyUI/venv/bin/activate && cd ComfyUI && pip install "huggingface_hub[cli]"	
	export PATH="$(CUDA_PATH_V12_4)/bin:$(PATH)" && export LIBGL_ALWAYS_SOFTWARE=1 && . ComfyUI/venv/bin/activate && cd ComfyUI/custom_nodes/ComfyUI-3D-Pack && pip install -r requirements.txt
	# Takes 10 ~ 20 minutes
	export PATH="$(CUDA_PATH_V12_4)/bin:$(PATH)" && export LIBGL_ALWAYS_SOFTWARE=1 && . ComfyUI/venv/bin/activate && cd ComfyUI/custom_nodes/ComfyUI-3D-Pack && python install.py
	export PATH="$(CUDA_PATH_V12_4)/bin:$(PATH)" && export LIBGL_ALWAYS_SOFTWARE=1 && . ComfyUI/venv/bin/activate && cd ComfyUI/custom_nodes/ComfyUI-layerdiffuse && pip install -r requirements.txt

run:
	PATH="$(CUDA_PATH_V12_4)/bin:$(PATH)" LIBGL_ALWAYS_SOFTWARE=1 ComfyUI/venv/bin/python ComfyUI/main.py --enable-cors-header --listen 0.0.0.0 --port 11188 --extra-model-paths-config extra_model_paths.yaml

push-extra_model_paths.yaml:
	aws ssm put-parameter \
		--name $(REMOTE_EXTRA_MODEL_PATHS) \
		--tier "Advanced" \
		--type SecureString \
		--value file://extra_model_paths.yaml \
		--overwreite \
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

ComfyUI/custom_nodes/comfyui-manager: ComfyUI
	git -C $@ pull || git clone https://github.com/ltdrdata/ComfyUI-Manager $@

# v0.1.4 fails to install from ComfyUI-Manager
# Reqruires OpenGL 4.4 or higher
# Requires modification written in https://github.com/MrForExample/ComfyUI-3D-Pack/issues/425#issuecomment-2679837846
ComfyUI/custom_nodes/ComfyUI-3D-Pack: ComfyUI
	git -C $@ pull || git clone https://github.com/MrForExample/ComfyUI-3D-Pack.git $@

ComfyUI/custom_nodes/ComfyUI-layerdiffuse: ComfyUI
	git -C $@ pull || git clone https://github.com/huchenlei/ComfyUI-layerdiffuse.git $@

ComfyUI/user/default/workflows/ComfyUI-3D-Pack: ComfyUI
	ln -sf $(CURRENT_DIR)/ComfyUI/custom_nodes/ComfyUI-3D-Pack/_Example_Workflows $@

ComfyUI/user/default/workflows/ComfyUI-layerdiffuse: ComfyUI ComfyUI/custom_nodes/ComfyUI-layerdiffuse
	ln -sf $(CURRENT_DIR)/ComfyUI/custom_nodes/ComfyUI-layerdiffuse/example_workflows $@
