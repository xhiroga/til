# copied from FramePack-eichi/run_oneframe_ichi.bat
run: FramePack FramePack-eichi FramePack/eichi_utils FramePack/endframe_ichi.py FramePack/endframe_ichi_f1.py FramePack/oneframe_ichi.py FramePack/submodules FramePack/webui
	export PYTHONPATH=$$PYTHONPATH:$$(pwd)/FramePack && uv run --directory FramePack oneframe_ichi.py --server 127.0.0.1 --inbrowser

FramePack:
	git -C $@ pull || git clone https://github.com/lllyasviel/FramePack $@

FramePack-eichi:
	git -C $@ pull || git clone https://github.com/git-ai-code/FramePack-eichi $@

# Thanks to https://note.com/catap_art3d/n/nca05195eecc2
FramePack/eichi_utils: FramePack FramePack-eichi
	ln -fsr FramePack-eichi/webui/eichi_utils FramePack/eichi_utils

FramePack/endframe_ichi.py: FramePack FramePack-eichi
	ln -fsr FramePack-eichi/webui/endframe_ichi.py FramePack/endframe_ichi.py

FramePack/endframe_ichi_f1.py: FramePack FramePack-eichi
	ln -fsr FramePack-eichi/webui/endframe_ichi_f1.py FramePack/endframe_ichi_f1.py

FramePack/oneframe_ichi.py: FramePack FramePack-eichi
	ln -fsr FramePack-eichi/webui/oneframe_ichi.py FramePack/oneframe_ichi.py

FramePack/submodules: FramePack FramePack-eichi
	ln -fsr FramePack-eichi/webui/submodules FramePack/submodules

FramePack/webui: FramePack FramePack-eichi
	ln -fsr FramePack-eichi/webui FramePack/webui
