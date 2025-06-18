LOGDIR=/mnt/d/output/til/software-engineering/python/pytorch/_src/sandbox

.PHONY: run

run: models/trt.ep
	uv run python demo_aot.py

models/trt.ep: demo_aot.py
	mkdir -p models
	uv run demo_aot.py --export

profile:
	uv run python demo_profile.py --logdir $(LOGDIR) --repeat 10

nsight:
	@# WSLにGUIがない場合はホストのWindowsで表示する。
	nsys profile -o $(LOGDIR)/profile_report_$(shell date +%Y%m%d_%H%M%S) \
		--cudabacktrace=true --cuda-memory-usage=true uv run python demo_profile.py --logdir $(LOGDIR) --repeat 10
