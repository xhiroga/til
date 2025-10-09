from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
import torchaudio
import umap


# ---------------------------------------------------------------------------
# Locate AV-HuBERT implementation
# ---------------------------------------------------------------------------


def _resolve_repo_path() -> Path:
    env_override = os.environ.get("AV_HUBERT_S2S_PATH")
    if env_override:
        return Path(env_override).expanduser().resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "AV-HuBERT-S2S"
        if candidate.is_dir():
            return candidate
    raise RuntimeError(
        "Couldn't locate AV-HuBERT-S2S repository. Set AV_HUBERT_S2S_PATH environment variable."
    )


_REPO_PATH = _resolve_repo_path()
_SRC_PATH = _REPO_PATH / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from av_hubert_s2s.dataset.utils import load_video  # noqa: E402
from av_hubert_s2s.model.avhubert2text import AV2TextForConditionalGeneration  # noqa: E402


# ---------------------------------------------------------------------------
# Model bootstrap helpers
# ---------------------------------------------------------------------------


def _resolve_model_path() -> Path:
    env_override = os.environ.get("AV_HUBERT_MODEL_PATH")
    if env_override:
        path = Path(env_override).expanduser().resolve()
        if path.is_dir():
            return path
        raise RuntimeError(f"AV_HUBERT_MODEL_PATH={path} is not a directory")

    cache_root = _REPO_PATH / "model-bin"
    candidates = list(cache_root.glob("models--nguyenvulebinh--AV-HuBERT-*/snapshots/*"))
    if not candidates:
        raise RuntimeError(
            "Hugging Face snapshot for AV-HuBERT not found. Download the model into model-bin/"
        )
    return sorted(candidates)[0]


_MODEL_DIR = _resolve_model_path()
_MODEL_ID = _MODEL_DIR.name

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUDIO_SAMPLE_RATE = 16_000


def _load_encoder() -> torch.nn.Module:
    model = AV2TextForConditionalGeneration.from_pretrained(
        str(_MODEL_DIR),
        local_files_only=True,
    )
    return model.model.encoder.to(_DEVICE).eval()


_ENCODER = _load_encoder()


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


STACK_ORDER_AUDIO = 4
FBANK_MEL_BINS = 26


def _stack_frames(feats: np.ndarray, stack_order: int) -> np.ndarray:
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        remainder = stack_order - len(feats) % stack_order
        padding = np.zeros((remainder, feat_dim), dtype=feats.dtype)
        feats = np.concatenate([feats, padding], axis=0)
    feats = feats.reshape(-1, stack_order, feat_dim).reshape(-1, stack_order * feat_dim)
    return feats


def _compute_logfbank(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    tensor = torch.from_numpy(waveform).unsqueeze(0)
    fbank = torchaudio.compliance.kaldi.fbank(
        tensor,
        sample_frequency=sample_rate,
        num_mel_bins=FBANK_MEL_BINS,
        use_energy=False,
        dither=0.0,
        snip_edges=False,
    )
    return fbank.numpy()


def _prepare_audio(audio: Optional[Tuple[int, np.ndarray]], target_rate: int) -> Tuple[torch.Tensor, float]:
    if audio is None:
        raise gr.Error("音声をアップロードしてください。")

    sample_rate, waveform = audio
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if waveform.size == 0:
        raise gr.Error("空の音声です。別のファイルをお試しください。")

    waveform = waveform.astype(np.float32)
    waveform = np.clip(waveform, -1.0, 1.0)

    if sample_rate != target_rate:
        tensor = torch.from_numpy(waveform).unsqueeze(0)
        resampled = torchaudio.functional.resample(tensor, sample_rate, target_rate)
        waveform = resampled.squeeze(0).numpy()
        sample_rate = target_rate

    min_required = int(target_rate * 0.025)  # 25 ms window in Kaldi FBANK
    if waveform.shape[0] < max(min_required, 2):
        pad = max(min_required, 2) - waveform.shape[0]
        waveform = np.pad(waveform, (0, pad), mode="constant")

    duration = waveform.shape[0] / float(sample_rate)

    feats = _compute_logfbank(waveform, sample_rate=sample_rate).astype(np.float32)
    feats = _stack_frames(feats, STACK_ORDER_AUDIO)

    feats_tensor = torch.from_numpy(feats)
    feats_tensor = torch.nn.functional.layer_norm(feats_tensor, feats_tensor.shape[1:])
    feats_tensor = feats_tensor.permute(1, 0).unsqueeze(0)
    return feats_tensor, duration


VideoInput = Union[str, Path, Tuple[Union[str, Path], Optional[Union[str, Path]]]]


def _prepare_video(video_value: Optional[VideoInput]) -> torch.Tensor:
    if video_value is None:
        raise gr.Error("動画をアップロードしてください。")
    if isinstance(video_value, tuple):
        video_path = Path(video_value[0])
    else:
        video_path = Path(video_value)
    if not video_path.exists():
        raise gr.Error("動画ファイルを読み込めませんでした。")

    frames = load_video(str(video_path))
    if frames.size == 0:
        raise gr.Error("動画にフレームが含まれていません。")
    image_crop_size = 88
    image_mean = 0.421
    image_std = 0.165
    t, h, w = frames.shape
    top = max((h - image_crop_size) // 2, 0)
    left = max((w - image_crop_size) // 2, 0)
    frames = frames[:, top : top + image_crop_size, left : left + image_crop_size]
    frames = (frames - image_mean * 255.0) / (image_std * 255.0)
    frames = frames[..., None].astype(np.float32)
    video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
    return video_tensor


def _align_modalities(audio_feats: torch.Tensor, video_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    audio_steps = audio_feats.size(-1)
    video_steps = video_feats.size(2)

    if audio_steps == video_steps:
        return audio_feats, video_feats

    if audio_steps > video_steps:
        audio_feats = audio_feats[..., :video_steps]
    else:
        pad = video_steps - audio_steps
        pad_tensor = torch.zeros(audio_feats.size(0), audio_feats.size(1), pad, dtype=audio_feats.dtype)
        audio_feats = torch.cat([audio_feats, pad_tensor], dim=-1)
    return audio_feats, video_feats


# ---------------------------------------------------------------------------
# Embedding + projection
# ---------------------------------------------------------------------------


def _reduce_embeddings(hidden_states: torch.Tensor, duration: float) -> go.Figure:
    embeddings = hidden_states.cpu().numpy()
    frame_count = embeddings.shape[0]

    if frame_count < 4:
        raise gr.Error("シーケンスが短すぎて可視化できません。もう少し長い入力をお試しください。")

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
        title="AV-HuBERT Last Hidden States (UMAP + Time)",
        scene=dict(
            xaxis_title="Frame Index",
            yaxis_title="UMAP-1",
            zaxis_title="UMAP-2",
        ),
        height=720,
        margin=dict(l=0, r=0, b=0, t=30),
    )
    return fig


def embed_modalities(audio: Optional[Tuple[int, np.ndarray]], video: Optional[VideoInput]) -> Tuple[go.Figure, str]:
    audio_feats, duration = _prepare_audio(audio, target_rate=AUDIO_SAMPLE_RATE)
    video_feats = _prepare_video(video)
    audio_feats, video_feats = _align_modalities(audio_feats, video_feats)

    attention_mask = torch.zeros(audio_feats.size(0), audio_feats.size(-1), dtype=torch.bool)

    audio_feats = audio_feats.to(_DEVICE)
    video_feats = video_feats.to(_DEVICE)
    attention_mask = attention_mask.to(_DEVICE)

    with torch.no_grad():
        outputs = _ENCODER(
            input_features=audio_feats,
            attention_mask=attention_mask,
            video=video_feats,
        )
        hidden_states = outputs.last_hidden_state.squeeze(0)

    figure = _reduce_embeddings(hidden_states, duration=duration)

    summary = (
        f"**Model snapshot:** {_MODEL_ID}\n"
        f"**Device:** {_DEVICE.type}\n"
        f"**Duration:** {duration:.2f} s\n"
        f"**Frames:** {hidden_states.shape[0]}\n"
        f"**Embedding dim:** {hidden_states.shape[1]}"
    )

    return figure, summary


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def build_interface() -> gr.Blocks:
    description = (
        "## AV-HuBERT UMAP 可視化\n"
        "動画（口元）と対応する 16 kHz モノラル音声を入力すると、"
        "AV-HuBERT エンコーダの最終層埋め込みを UMAP で 2 次元に射影し、"
        "時間軸を含む 3D 散布図として表示します。"
    )

    with gr.Blocks(title="AV-HuBERT Embedding Visualizer") as demo:
        gr.Markdown(description)
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="映像 (口元クリップ)",
                    sources=["upload"],
                    format="mp4",
                )
                audio_input = gr.Audio(label="音声 (16 kHz WAV)", sources=["upload", "microphone"], type="numpy")
                run_button = gr.Button("埋め込みを計算する", variant="primary")
            with gr.Column(scale=1):
                plot_output = gr.Plot(label="AV-HuBERT UMAP + Time")
                summary_output = gr.Markdown(label="メタ情報")

        run_button.click(embed_modalities, inputs=[audio_input, video_input], outputs=[plot_output, summary_output])

    return demo


def main() -> None:
    demo = build_interface()
    demo.queue().launch()


if __name__ == "__main__":
    main()
