.PHONY: run

run: models/trt.ep
	uv run python demo_aot.py

models/trt.ep: demo_aot.py
	mkdir -p models
	uv run demo_aot.py --export
