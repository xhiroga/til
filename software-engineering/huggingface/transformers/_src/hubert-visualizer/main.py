from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
import umap
from transformers import AutoProcessor, HubertModel


MODEL_ID = "facebook/hubert-large-ls960-ft"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
MODEL = HubertModel.from_pretrained(MODEL_ID)
MODEL.eval()

PREPROCESSED_DIR = Path(__file__).resolve().parent / "preprocessed"
UMAP_MODEL_PATH = PREPROCESSED_DIR / "umap_model.pkl"
METADATA_PATH = PREPROCESSED_DIR / "metadata.json"

FRAME_LIMIT_DEFAULT = 200


def _load_preprocessed() -> tuple[Optional[umap.UMAP], int, Optional[int]]:
    reducer: Optional[umap.UMAP] = None
    frame_limit = FRAME_LIMIT_DEFAULT
    reference_points = None

    if UMAP_MODEL_PATH.exists():
        with UMAP_MODEL_PATH.open("rb") as fp:
            reducer = pickle.load(fp)

    if METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text())
            frame_limit = int(metadata.get("frames_per_file_limit", frame_limit))
            if "total_reference_points" in metadata:
                reference_points = int(metadata["total_reference_points"])
        except Exception:  # noqa: BLE001
            pass

    return reducer, frame_limit, reference_points


REFERENCE_REDUCER, FRAME_LIMIT, REFERENCE_POINTS_COUNT = _load_preprocessed()


def _prepare_waveform(audio: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray]:
    if audio is None:
        raise gr.Error("音声ファイルをアップロードしてください。")

    sample_rate, data = audio
    if data.size == 0:
        raise gr.Error("空の音声ファイルです。別のファイルをお試しください。")

    if data.ndim == 2:
        data = data.mean(axis=1)

    data = data.astype(np.float32)
    peak = float(np.max(np.abs(data)))
    if peak > 1.0 and not math.isclose(peak, 0.0):
        data = data / peak

    return sample_rate, data


def _select_frame_indices(frame_count: int) -> np.ndarray:
    if frame_count <= FRAME_LIMIT:
        return np.arange(frame_count, dtype=np.int32)
    return np.linspace(0, frame_count - 1, FRAME_LIMIT, dtype=np.int32)


def _reduce_embeddings(hidden_states: torch.Tensor, duration: float) -> tuple[go.Figure, int]:
    embeddings = hidden_states.cpu().numpy()
    frame_count = embeddings.shape[0]

    if frame_count < 4:
        raise gr.Error("音声が短すぎて可視化できません。もう少し長い音声でお試しください。")

    indices = np.arange(frame_count, dtype=np.int32)
    times = np.linspace(0.0, duration, frame_count)
    frame_indices = np.arange(frame_count)
    neighbours = max(2, min(15, frame_count - 1))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbours,
        random_state=42,
        metric="cosine",
        min_dist=0.1,
    )
    umap_points = reducer.fit_transform(embeddings)

    scatter = go.Scatter3d(
        x=frame_indices,
        y=umap_points[:, 0],
        z=umap_points[:, 1],
        mode="markers",
        marker=dict(
            size=4,
            color=times,
            colorscale="Viridis",
            colorbar=dict(title="Time (s)"),
            opacity=0.85,
        ),
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        title="HuBERT Last Hidden States (UMAP + Time)",
        scene=dict(
            xaxis_title="Frame Index",
            yaxis_title="UMAP-1",
            zaxis_title="UMAP-2",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    return fig, len(indices)


def _project_on_reference(hidden_states: torch.Tensor, duration: float) -> tuple[go.Figure, int]:
    if REFERENCE_REDUCER is None:
        raise ValueError("Reference reducer is not available")

    embeddings = hidden_states.cpu().numpy()
    frame_count = embeddings.shape[0]

    if frame_count < 4:
        raise gr.Error("音声が短すぎて可視化できません。もう少し長い音声でお試しください。")

    indices = _select_frame_indices(frame_count)
    selected_embeddings = embeddings[indices]
    times = np.linspace(0.0, duration, frame_count, dtype=np.float32)[indices]

    umap_points = REFERENCE_REDUCER.transform(selected_embeddings)

    scatter = go.Scatter3d(
        x=indices,
        y=umap_points[:, 0],
        z=umap_points[:, 1],
        mode="markers",
        marker=dict(
            size=5,
            color=times,
            colorscale="Inferno",
            colorbar=dict(title="Time (s)"),
            opacity=0.9,
        ),
        name="Uploaded audio",
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        title="HuBERT UMAP (Reference Projection)",
        scene=dict(
            xaxis_title="Frame Index",
            yaxis_title="UMAP-1",
            zaxis_title="UMAP-2",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    return fig, len(indices)


def embed_audio(audio: Tuple[int, np.ndarray]):
    sample_rate, waveform = _prepare_waveform(audio)
    duration = waveform.shape[0] / sample_rate

    inputs = PROCESSOR(waveform, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        outputs = MODEL(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1].squeeze(0)

    frames = hidden_states.shape[0]
    if REFERENCE_REDUCER is not None:
        figure, projected_frames = _project_on_reference(hidden_states, duration)
        projection_mode = "Reference UMAP"
        reference_points = REFERENCE_POINTS_COUNT
    else:
        figure, projected_frames = _reduce_embeddings(hidden_states, duration)
        projection_mode = "On-the-fly UMAP"
        reference_points = None

    summary = (
        f"**Sample rate:** {sample_rate} Hz\n"
        f"**Duration:** {duration:.2f} s\n"
        f"**Frames:** {frames}\n"
        f"**Projected frames:** {projected_frames}\n"
        f"**Projection:** {projection_mode}\n"
    )
    if reference_points is not None:
        summary += f"**Reference points:** {reference_points}\n"

    return figure, summary


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="HuBERT Embedding Visualizer") as demo:
        gr.Markdown(
            """
            ## HuBERT UMAP 可視化
            左欄に音声ファイルをドロップまたは録音すると、HuBERT の最終層埋め込みを UMAP で 2 次元に射影し、
            時間方向を z 軸とした 3D 散布図で表示します。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="numpy",
                    label="音声ファイル",
                )
                run_button = gr.Button("埋め込みを計算する", variant="primary")
            with gr.Column(scale=1):
                plot_output = gr.Plot(label="HuBERT UMAP + Time")
                summary_output = gr.Markdown(label="メタ情報")

        run_button.click(embed_audio, inputs=audio_input, outputs=[plot_output, summary_output])

    return demo


def main() -> None:
    demo = build_interface()
    demo.queue().launch()


if __name__ == "__main__":
    main()
