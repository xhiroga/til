.PHONY: run

run: models/yolo11m.pt
	uv run main.py --model models/yolo11m.pt

models/yolo11m.pt:
	mkdir -p models
	wget -O $@ https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
