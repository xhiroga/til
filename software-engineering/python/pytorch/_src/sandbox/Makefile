LOGDIR=/mnt/d/output/til/software-engineering/python/pytorch/_src/sandbox

.PHONY: run profile nsight

run: models/trt.pt2
	uv run python demo_aot.py

models/trt.pt2: demo_aot.py
	mkdir -p models
	uv run demo_aot.py --export
	@# 最新PyTorchで導入された`as_custom_obj`に対応していないらしく、2025-06-18時点ではエラーになる。
	# uvx netron models/trt.pt2

profile:
	uv run python demo_profile.py --logdir $(LOGDIR) --repeat 10

nsight:
	@# WSLにGUIがない場合はホストのWindowsで表示する。
	nsys profile -o $(LOGDIR)/profile_report_$(shell date +%Y%m%d_%H%M%S) \
		--cudabacktrace=true --cuda-memory-usage=true uv run python demo_profile.py --logdir $(LOGDIR) --repeat 10
