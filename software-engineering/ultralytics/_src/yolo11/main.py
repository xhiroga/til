from pathlib import Path
import argparse

import gradio as gr
from ultralytics import YOLO


def load_model(path_text, state):
    text = (path_text or "").strip()
    if not text:
        return state, "モデルパスを入力してください"
    path = Path(text).expanduser()
    if not path.exists():
        return state, f"見つかりません: {path}"
    try:
        state = {"model": YOLO(str(path)), "path": str(path)}
    except Exception as err:
        return {}, f"読み込み失敗: {err}"
    return state, f"読み込み成功: {path}"


def run(image, conf, iou, state):
    if image is None:
        raise gr.Error("画像をアップロードしてください")
    model = state.get("model") if state else None
    if model is None:
        raise gr.Error("先にモデルを読み込んでください")
    result = model.predict(image, conf=conf, iou=iou, verbose=False)[0]
    boxes = getattr(result, "boxes", None)
    names = result.names
    rows = []
    if boxes is not None:
        for box in boxes:
            rows.append([names.get(int(box.cls), "?"), float(box.conf)])
    return result.plot(), rows


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv9 Gradio demo")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO weights file to load on startup",
    )
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    default_model_path = (args.model or "").strip()
    if not default_model_path:
        raise SystemExit("--model でモデルパスを指定してください")

    initial_state, status_msg = load_model(default_model_path, {})
    if not initial_state:
        raise SystemExit(status_msg)

    with gr.Blocks(title="YOLOv9 Demo") as demo:
        state = gr.State(initial_state)
        gr.Markdown(f"読み込んだモデル: {initial_state.get('path', '不明')}")

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="numpy", label="入力画像")
                conf = gr.Slider(0.01, 1.0, value=0.25, step=0.01, label="Confidence")
                iou = gr.Slider(0.01, 1.0, value=0.45, step=0.01, label="IoU")
                run_btn = gr.Button("推論実行")
            with gr.Column(scale=1):
                image_out = gr.Image(type="numpy", label="推論結果")
                table = gr.Dataframe(
                    headers=["label", "confidence"],
                    datatype=["str", "number"],
                    interactive=False,
                )

        run_btn.click(run, [image_in, conf, iou, state], [image_out, table])

    demo.launch()


if __name__ == "__main__":
    main()
