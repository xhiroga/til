all: FramePack FramePack-LoRAReady;

FramePack:
	git -C $@ pull || git clone https://github.com/lllyasviel/FramePack $@

FramePack-LoRAReady:
	git -C $@ pull || git clone https://github.com/kohya-ss/FramePack-LoRAReady $@
