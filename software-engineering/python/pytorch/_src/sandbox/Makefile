.PHONY: run

run: models/exported.pt2
	mkdir -p models
	uv run python demo_aot.py

models/exported.pt2: demo_aot.py
	uv run demo_aot.py --export
