from __future__ import annotations

import argparse
import json
import math
import pickle
import wave
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import umap
from tqdm.auto import tqdm
from transformers import AutoProcessor, HubertModel

MODEL_ID = "facebook/hubert-large-ls960-ft"
FRAMES_PER_FILE_LIMIT = 200
MIN_FRAMES = 4
OUTPUT_DIRNAME = "preprocessed"


def load_waveform_from_wav(path: Path) -> Tuple[int, np.ndarray]:
    with wave.open(path.as_posix(), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_count = wf.getnframes()
        audio_bytes = wf.readframes(frame_count)

    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    if sample_width not in dtype_map:
        raise ValueError(f"Unsupported sample width {sample_width} bytes in {path}")

    data = np.frombuffer(audio_bytes, dtype=dtype_map[sample_width])

    if sample_width == 1:
        # 8-bit PCM is unsigned
        data = data.astype(np.float32)
        data = (data - 128.0) / 128.0
    else:
        max_val = float(2 ** (8 * sample_width - 1))
        data = data.astype(np.float32) / max_val

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return sample_rate, data.astype(np.float32)


def prepare_waveform(audio: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray]:
    sample_rate, data = audio
    if data.size == 0:
        raise ValueError("Empty audio file")

    peak = float(np.max(np.abs(data)))
    if peak > 1.0 and not math.isclose(peak, 0.0):
        data = data / peak

    return sample_rate, data


def iter_wav_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.wav"):
        if path.is_file():
            yield path


def select_frame_indices(frame_count: int) -> np.ndarray:
    if frame_count <= FRAMES_PER_FILE_LIMIT:
        return np.arange(frame_count, dtype=np.int32)
    return np.linspace(0, frame_count - 1, FRAMES_PER_FILE_LIMIT, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute HuBERT embeddings and UMAP projection")
    parser.add_argument("audio_dir", type=Path, help="Root directory containing .wav files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to store preprocessed artifacts")
    parser.add_argument("--device", type=str, default=None, help="Torch device override (e.g. cuda, cpu)")
    args = parser.parse_args()

    audio_root = args.audio_dir.expanduser().resolve()
    if not audio_root.exists():
        raise SystemExit(f"Audio directory not found: {audio_root}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(OUTPUT_DIRNAME)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print(f"Loading model {MODEL_ID} on {device}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = HubertModel.from_pretrained(MODEL_ID)
    model.eval()
    model.to(device)

    audio_files = sorted(iter_wav_files(audio_root))
    if not audio_files:
        raise SystemExit(f"No .wav files found under {audio_root}")

    print(f"Found {len(audio_files)} audio files. Processing...")

    collected_embeddings: List[np.ndarray] = []
    collected_times: List[np.ndarray] = []
    collected_clip_ids: List[str] = []
    collected_frame_idx: List[np.ndarray] = []
    collected_durations: List[np.ndarray] = []
    processed_files = 0
    skipped_files = 0

    for wav_path in tqdm(audio_files, desc="Embedding", unit="file"):
        relative_id = wav_path.relative_to(audio_root).as_posix()
        try:
            sample_rate, waveform = load_waveform_from_wav(wav_path)
            sample_rate, waveform = prepare_waveform((sample_rate, waveform))
        except Exception as exc:  # noqa: BLE001
            skipped_files += 1
            print(f"[warn] Skipping {relative_id}: {exc}")
            continue

        duration = waveform.shape[0] / sample_rate
        inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1].squeeze(0).cpu()

        frame_count = hidden_states.shape[0]
        if frame_count < MIN_FRAMES:
            skipped_files += 1
            print(f"[warn] Skipping {relative_id}: only {frame_count} frames")
            continue

        indices = select_frame_indices(frame_count)
        embeddings = hidden_states[indices].numpy()
        times = np.linspace(0.0, duration, frame_count, dtype=np.float32)[indices]

        collected_embeddings.append(embeddings)
        collected_times.append(times)
        collected_clip_ids.extend([relative_id] * len(indices))
        collected_frame_idx.append(indices.astype(np.int32))
        collected_durations.append(np.full(len(indices), duration, dtype=np.float32))
        processed_files += 1

    if not collected_embeddings:
        raise SystemExit("No audio files yielded valid embeddings.")

    feature_matrix = np.vstack(collected_embeddings)
    print(f"Fitting UMAP on {feature_matrix.shape[0]} frames, dimensionality {feature_matrix.shape[1]}...")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        random_state=42,
        metric="cosine",
        min_dist=0.1,
    )
    reducer.fit(feature_matrix)

    umap_points = reducer.embedding_.astype(np.float32)
    times = np.concatenate(collected_times).astype(np.float32)
    frame_indices = np.concatenate(collected_frame_idx).astype(np.int32)
    durations = np.concatenate(collected_durations).astype(np.float32)
    clip_ids = np.array(collected_clip_ids)

    reference_path = output_dir / "reference_points.npz"
    model_path = output_dir / "umap_model.pkl"
    metadata_path = output_dir / "metadata.json"

    np.savez_compressed(
        reference_path,
        umap=umap_points,
        time=times,
        frame=frame_indices,
        duration=durations,
        clip=clip_ids,
    )
    print(f"Saved reference points to {reference_path}")

    with model_path.open("wb") as fp:
        pickle.dump(reducer, fp)
    print(f"Saved UMAP model to {model_path}")

    metadata = {
        "model_id": MODEL_ID,
        "audio_root": str(audio_root),
        "processed_files": processed_files,
        "skipped_files": skipped_files,
        "total_reference_points": int(umap_points.shape[0]),
        "frames_per_file_limit": FRAMES_PER_FILE_LIMIT,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
