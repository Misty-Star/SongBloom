"""VQ codebook 训练所需的数据采样工具。"""

from __future__ import annotations

import json
import os
import random
import typing as tp

import torch
import torch.nn.functional as F
from tqdm import tqdm


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
DEFAULT_MUQ_MIN_CHUNK_SECONDS = 10.0


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


def align_embeddings_to_target_fps(
    embedding: torch.Tensor,
    audio_num_samples: int,
    sample_rate: int,
    target_fps: int,
) -> torch.Tensor:
    target_frames = int(audio_num_samples / sample_rate * target_fps)
    if target_frames <= 0 or embedding.shape[1] == target_frames:
        return embedding
    return F.interpolate(
        embedding.transpose(1, 2),
        size=target_frames,
        mode="nearest",
    ).transpose(1, 2)


def build_chunk_spans(
    total_num_samples: int,
    chunk_num_samples: int,
    min_chunk_num_samples: int,
) -> list[tuple[int, int]]:
    if total_num_samples <= 0:
        return []
    if chunk_num_samples <= 0 or total_num_samples <= chunk_num_samples:
        return [(0, total_num_samples)]

    spans: list[tuple[int, int]] = []
    for start in range(0, total_num_samples, chunk_num_samples):
        end = min(start + chunk_num_samples, total_num_samples)
        if spans and end == total_num_samples and (end - start) < min_chunk_num_samples:
            prev_start, _prev_end = spans[-1]
            spans[-1] = (prev_start, end)
        else:
            spans.append((start, end))
    return spans


def collect_embedding_samples(
    audio_paths: tp.Sequence[str],
    muq_model,
    sample_rate: int,
    target_fps: int,
    frames_per_audio: int,
    max_total_frames: int,
    seed: int,
    muq_chunk_seconds: float = 0.0,
    progress_desc: str = "Collecting MuQ frames",
) -> torch.Tensor:
    import torchaudio

    from .extract_sketch import extract_muq_embeddings

    rng = random.Random(seed)
    collected: list[torch.Tensor] = []
    total_frames = 0
    songs_used = 0
    min_chunk_num_samples = max(1, int(DEFAULT_MUQ_MIN_CHUNK_SECONDS * sample_rate))
    chunk_num_samples = max(0, int(float(muq_chunk_seconds) * sample_rate))
    if chunk_num_samples > 0:
        chunk_num_samples = max(chunk_num_samples, min_chunk_num_samples)

    with tqdm(total=max_total_frames, desc=progress_desc, unit="frame") as progress:
        for audio_path in audio_paths:
            audio_label = _short_audio_label(audio_path)
            progress.set_postfix(songs=songs_used, audio=audio_label)

            wav, sr = torchaudio.load(audio_path)
            if sr != sample_rate:
                wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)

            if chunk_num_samples > 0 and wav.shape[-1] > chunk_num_samples:
                spans = build_chunk_spans(
                    total_num_samples=wav.shape[-1],
                    chunk_num_samples=chunk_num_samples,
                    min_chunk_num_samples=min_chunk_num_samples,
                )
                embedding_chunks: list[torch.Tensor] = []
                total_chunks = len(spans)
                for chunk_index, (start, end) in enumerate(spans, start=1):
                    progress.set_postfix(
                        songs=songs_used,
                        audio=audio_label,
                        chunk=f"{chunk_index}/{total_chunks}",
                    )
                    chunk = wav[:, start:end]
                    if chunk.shape[-1] == 0:
                        continue
                    embedding_chunks.append(extract_muq_embeddings(muq_model, chunk, sample_rate).cpu())
                if not embedding_chunks:
                    continue
                embedding = torch.cat(embedding_chunks, dim=1)
            else:
                embedding = extract_muq_embeddings(muq_model, wav, sample_rate).cpu()
            embedding = align_embeddings_to_target_fps(
                embedding=embedding,
                audio_num_samples=wav.shape[-1],
                sample_rate=sample_rate,
                target_fps=target_fps,
            )

            sampled = sample_frames_from_embedding(embedding, frames_per_audio, rng)
            if sampled.numel() == 0:
                continue

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
                frames=f"{total_frames}/{max_total_frames}",
            )

            if total_frames >= max_total_frames:
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
