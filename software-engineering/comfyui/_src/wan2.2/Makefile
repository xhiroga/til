.PHONY: download clean status run init

# Model files to download
models = \
	ComfyUI/models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors \
	ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
	ComfyUI/models/vae/wan2.2_vae.safetensors

# Main download target
download: $(models)

# Download individual model files
ComfyUI/models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors:
	@mkdir -p $(dir $@)
	@echo "Downloading wan2.2_ti2v_5B_fp16.safetensors..."
	uvx --from "huggingface_hub[cli]" huggingface-cli download \
		Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
		split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors \
		--repo-type model \
		--local-dir .tmp_download
	@mv .tmp_download/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors $@
	@rm -rf .tmp_download

ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors:
	@mkdir -p $(dir $@)
	@echo "Downloading umt5_xxl_fp8_e4m3fn_scaled.safetensors..."
	uvx --from "huggingface_hub[cli]" huggingface-cli download \
		Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
		split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
		--repo-type model \
		--local-dir .tmp_download
	@mv .tmp_download/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors $@
	@rm -rf .tmp_download

ComfyUI/models/vae/wan2.2_vae.safetensors:
	@mkdir -p $(dir $@)
	@echo "Downloading wan2.2_vae.safetensors..."
	uvx --from "huggingface_hub[cli]" huggingface-cli download \
		Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
		split_files/vae/wan2.2_vae.safetensors \
		--repo-type model \
		--local-dir .tmp_download
	@mv .tmp_download/split_files/vae/wan2.2_vae.safetensors $@
	@rm -rf .tmp_download

# Clean downloaded models
clean:
	rm -rf $(models)
	@echo "Cleaned downloaded models."

# Show download status
status:
	@echo "Model download status:"
	@for model in $(models); do \
		if [ -f $$model ]; then \
			echo "  ✓ $$model"; \
		else \
			echo "  ✗ $$model"; \
		fi \
	done

# Run ComfyUI
run:
	uv run ComfyUI/main.py --enable-cors-header --listen --port 11188 --extra-model-paths-config extra_model_paths.yaml --fast

# Initialize project
init: .venv ComfyUI download

.venv:
	uv sync

ComfyUI:
	(git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI $@) && git -C $@ checkout v0.3.46
