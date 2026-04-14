"""VQ codebook 训练所需的数据采样工具。"""

from __future__ import annotations

import json
import os
import random
import typing as tp

import torch
from tqdm import tqdm

from .muq_feature_cache import (
    DEFAULT_MUQ_MIN_CHUNK_SECONDS,
    align_embeddings_to_target_fps,
    build_chunk_spans,
    get_or_compute_muq_embedding,
)


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")



def _short_audio_label(audio_path: str, limit: int = 32) -> str:
    label = os.path.basename(audio_path) or audio_path
    if len(label) <= limit:
        return label
    return "..." + label[-(limit - 3):]



def load_manifest_audio_paths(path: str) -> list[str]:
    audio_paths: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            audio_path = row.get("audio_path")
            if audio_path:
                audio_paths.append(str(audio_path))
    return audio_paths



def list_audio_files(audio_dir: str, recursive: bool = True) -> list[str]:
    results: list[str] = []
    for root, _dirs, files in os.walk(audio_dir):
        for name in files:
            if name.lower().endswith(AUDIO_EXTENSIONS):
                results.append(os.path.join(root, name))
        if not recursive:
            break
    return sorted(results)



def split_audio_paths(
    audio_paths: tp.Sequence[str],
    heldout_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    ordered = sorted(str(path) for path in audio_paths)
    if not ordered:
        return [], []

    rng = random.Random(seed)
    rng.shuffle(ordered)

    if len(ordered) == 1 or heldout_ratio <= 0:
        return ordered, []

    heldout_count = max(1, int(len(ordered) * heldout_ratio))
    heldout_count = min(heldout_count, len(ordered) - 1)

    heldout = sorted(ordered[:heldout_count])
    train = sorted(ordered[heldout_count:])
    return train, heldout



def sample_frames_from_embedding(
    embedding: torch.Tensor,
    max_frames: int,
    rng: random.Random,
) -> torch.Tensor:
    frames = embedding.squeeze(0).detach().cpu()
    num_frames = frames.shape[0]
    if num_frames <= max_frames:
        return frames.float()
    indices = sorted(rng.sample(range(num_frames), max_frames))
    return frames[indices].float()



def collect_embedding_samples(
    audio_paths: tp.Sequence[str],
    muq_model,
    sample_rate: int,
    target_fps: int,
    frames_per_audio: int,
    max_total_frames: int,
    seed: int,
    muq_model_name: str = "",
    cache_dir: str = "",
    muq_chunk_seconds: float = 0.0,
    progress_desc: str = "Collecting MuQ frames",
) -> torch.Tensor:
    rng = random.Random(seed)
    collected: list[torch.Tensor] = []
    total_frames = 0
    songs_used = 0
    use_full_ceiling = max_total_frames <= 0
    frame_budget_label = "full" if use_full_ceiling else str(max_total_frames)

    with tqdm(total=None if use_full_ceiling else max_total_frames, desc=progress_desc, unit="frame") as progress:
        for audio_path in audio_paths:
            audio_label = _short_audio_label(audio_path)
            progress.set_postfix(songs=songs_used, audio=audio_label)

            embedding = get_or_compute_muq_embedding(
                audio_path=audio_path,
                muq_model=muq_model,
                muq_model_name=muq_model_name or type(muq_model).__name__,
                sample_rate=sample_rate,
                target_fps=target_fps,
                cache_dir=cache_dir,
                muq_chunk_seconds=muq_chunk_seconds,
                progress_postfix=lambda payload, audio_label=audio_label, songs_used=songs_used: progress.set_postfix(
                    songs=songs_used,
                    audio=audio_label,
                    **payload,
                ),
            )

            sampled = sample_frames_from_embedding(embedding, frames_per_audio, rng)
            if sampled.numel() == 0:
                continue

            if not use_full_ceiling:
                remaining = max_total_frames - total_frames
                if remaining <= 0:
                    break
                if sampled.shape[0] > remaining:
                    sampled = sampled[:remaining]

            collected.append(sampled)
            songs_used += 1
            total_frames += sampled.shape[0]
            progress.update(sampled.shape[0])
            progress.set_postfix(
                songs=songs_used,
                audio=audio_label,
                frames=f"{total_frames}/{frame_budget_label}",
            )

            if not use_full_ceiling and total_frames >= max_total_frames:
                break

    if not collected:
        raise RuntimeError("No MuQ embedding frames collected; check audio input and MuQ setup.")
    return torch.cat(collected, dim=0)



def resolve_audio_paths(
    *,
    audio_dir: str = "",
    input_jsonl: str = "",
    recursive: bool = True,
    max_audios: int = 0,
) -> list[str]:
    if audio_dir:
        audio_paths = list_audio_files(audio_dir, recursive=recursive)
    else:
        audio_paths = load_manifest_audio_paths(input_jsonl)

    if max_audios > 0:
        audio_paths = audio_paths[:max_audios]
    return audio_paths
