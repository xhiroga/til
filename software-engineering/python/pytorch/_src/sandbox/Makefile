.PHONY: run

run: models/compiled.so
	mkdir -p models
	uv run python demo_compile.py

models/compiled.so: demo_compile.py
	uv run demo_compile.py --compile
