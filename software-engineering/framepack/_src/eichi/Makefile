# copied from FramePack-eichi/run_oneframe_ichi.bat
endframe_ichi: FramePack FramePack-eichi FramePack/eichi_utils FramePack/locales FramePack/lora_utils FramePack/endframe_ichi.py FramePack/endframe_ichi_f1.py FramePack/oneframe_ichi.py FramePack/submodules FramePack/webui FramePack/lora/lineart_lora.safetensors override_diffusers_helper_bucket_tools_py override_diffusers_helper_memory_py
	export PYTHONPATH=$$PYTHONPATH:$$(pwd)/FramePack && uv run --directory FramePack endframe_ichi.py --server 127.0.0.1 --inbrowser

endframe_ichi_f1: FramePack FramePack-eichi FramePack/eichi_utils FramePack/locales FramePack/lora_utils FramePack/endframe_ichi.py FramePack/endframe_ichi_f1.py FramePack/oneframe_ichi.py FramePack/submodules FramePack/webui FramePack/lora/lineart_lora.safetensors override_diffusers_helper_bucket_tools_py override_diffusers_helper_memory_py
	export PYTHONPATH=$$PYTHONPATH:$$(pwd)/FramePack && uv run --directory FramePack endframe_ichi_f1.py --server 127.0.0.1 --inbrowser

FramePack:
	git -C $@ pull || git clone https://github.com/lllyasviel/FramePack $@

FramePack-eichi:
	git -C $@ pull || git clone https://github.com/git-ai-code/FramePack-eichi $@

FramePack/eichi_utils: FramePack FramePack-eichi
	rsync -av --delete FramePack-eichi/webui/eichi_utils/ FramePack/eichi_utils/

FramePack/locales: FramePack FramePack-eichi
	rsync -av --delete FramePack-eichi/webui/locales/ FramePack/locales/

FramePack/lora_utils: FramePack FramePack-eichi
	rsync -av --delete FramePack-eichi/webui/lora_utils/ FramePack/lora_utils/

FramePack/endframe_ichi.py: FramePack FramePack-eichi
	cp -fr FramePack-eichi/webui/endframe_ichi.py FramePack/endframe_ichi.py

FramePack/endframe_ichi_f1.py: FramePack FramePack-eichi
	cp -fr FramePack-eichi/webui/endframe_ichi_f1.py FramePack/endframe_ichi_f1.py

FramePack/oneframe_ichi.py: FramePack FramePack-eichi
	cp -fr FramePack-eichi/webui/oneframe_ichi.py FramePack/oneframe_ichi.py

FramePack/submodules: FramePack FramePack-eichi
	rsync -av --delete FramePack-eichi/webui/submodules/ FramePack/submodules/

FramePack/webui: FramePack FramePack-eichi
	rsync -av --delete FramePack-eichi/webui/ FramePack/webui/

override_diffusers_helper_bucket_tools_py: FramePack FramePack-eichi
	cp -fr FramePack-eichi/webui/diffusers_helper/bucket_tools.py FramePack/diffusers_helper/bucket_tools.py

override_diffusers_helper_memory_py: FramePack FramePack-eichi
	cp -fr FramePack-eichi/webui/diffusers_helper/memory.py FramePack/diffusers_helper/memory.py

FramePack/lora: FramePack
	mkdir -p $@

FramePack/lora/lineart_lora.safetensors: FramePack/lora
	if uv run huggingface-cli whoami | grep -q 'Not logged in'; then uv run huggingface-cli login; fi
	uv run huggingface-cli download mattyamonaca/framepack_oneframe_extract_lineart_lora lineart_lora.safetensors --local-dir FramePack/lora
