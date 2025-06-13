.PHONY: run

run: models/exported_trt.pt2
	uv run python demo_aot.py

models/exported_trt.pt2: demo_aot.py
	mkdir -p models
	uv run demo_aot.py --export
