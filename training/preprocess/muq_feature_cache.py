"""MuQ 对齐 embedding 的提取与磁盘缓存工具。"""

from __future__ import annotations

import hashlib
import json
import os
import typing as tp

import torch
import torch.nn.functional as F


DEFAULT_MUQ_MIN_CHUNK_SECONDS = 10.0
MUQ_CACHE_VERSION = 1


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



def _audio_source_fingerprint(audio_path: str) -> dict:
    normalized_path = os.path.abspath(audio_path)
    try:
        stat = os.stat(normalized_path)
        return {
            "audio_path": normalized_path,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except FileNotFoundError:
        return {
            "audio_path": normalized_path,
            "missing": True,
        }



def _muq_cache_key(
    *,
    audio_path: str,
    muq_model_name: str,
    sample_rate: int,
    target_fps: int,
) -> str:
    payload = {
        "version": MUQ_CACHE_VERSION,
        "source": _audio_source_fingerprint(audio_path),
        "muq_model": muq_model_name,
        "sample_rate": int(sample_rate),
        "target_fps": int(target_fps),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()



def _resolve_cache_path(
    *,
    cache_dir: str,
    audio_path: str,
    muq_model_name: str,
    sample_rate: int,
    target_fps: int,
) -> str:
    cache_key = _muq_cache_key(
        audio_path=audio_path,
        muq_model_name=muq_model_name,
        sample_rate=sample_rate,
        target_fps=target_fps,
    )
    return os.path.join(os.path.abspath(cache_dir), cache_key[:2], f"{cache_key}.pt")



def _load_cached_embedding(cache_path: str) -> torch.Tensor | None:
    if not os.path.exists(cache_path):
        return None
    state = torch.load(cache_path, map_location="cpu")
    if isinstance(state, dict) and "embedding" in state:
        return state["embedding"].float().contiguous().cpu()
    if isinstance(state, torch.Tensor):
        return state.float().contiguous().cpu()
    raise TypeError(f"Unsupported MuQ cache payload type: {type(state)!r}")



def _save_cached_embedding(
    *,
    cache_path: str,
    embedding: torch.Tensor,
    metadata: dict,
) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    temp_path = cache_path + ".tmp"
    torch.save(
        {
            "embedding": embedding.detach().cpu().float().contiguous(),
            "metadata": metadata,
        },
        temp_path,
    )
    os.replace(temp_path, cache_path)



def _compute_aligned_embedding(
    *,
    audio_path: str,
    muq_model,
    sample_rate: int,
    target_fps: int,
    muq_chunk_seconds: float,
    wav: torch.Tensor | None,
    wav_sample_rate: int | None,
    progress_postfix: tp.Optional[tp.Callable[[dict], None]] = None,
) -> torch.Tensor:
    from .extract_sketch import extract_muq_embeddings

    if wav is None:
        import torchaudio

        wav, sr = torchaudio.load(audio_path)
    else:
        sr = wav_sample_rate or sample_rate
    if sr != sample_rate:
        import torchaudio

        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)

    min_chunk_num_samples = max(1, int(DEFAULT_MUQ_MIN_CHUNK_SECONDS * sample_rate))
    chunk_num_samples = max(0, int(float(muq_chunk_seconds) * sample_rate))
    if chunk_num_samples > 0:
        chunk_num_samples = max(chunk_num_samples, min_chunk_num_samples)

    if chunk_num_samples > 0 and wav.shape[-1] > chunk_num_samples:
        spans = build_chunk_spans(
            total_num_samples=wav.shape[-1],
            chunk_num_samples=chunk_num_samples,
            min_chunk_num_samples=min_chunk_num_samples,
        )
        embedding_chunks: list[torch.Tensor] = []
        total_chunks = len(spans)
        for chunk_index, (start, end) in enumerate(spans, start=1):
            if progress_postfix is not None:
                progress_postfix({"chunk": f"{chunk_index}/{total_chunks}"})
            chunk = wav[:, start:end]
            if chunk.shape[-1] == 0:
                continue
            embedding_chunks.append(extract_muq_embeddings(muq_model, chunk, sample_rate).cpu())
        if not embedding_chunks:
            raise RuntimeError(f"MuQ produced no chunks for audio: {audio_path}")
        embedding = torch.cat(embedding_chunks, dim=1)
    else:
        embedding = extract_muq_embeddings(muq_model, wav, sample_rate).cpu()

    return align_embeddings_to_target_fps(
        embedding=embedding,
        audio_num_samples=wav.shape[-1],
        sample_rate=sample_rate,
        target_fps=target_fps,
    ).float().contiguous().cpu()



def get_or_compute_muq_embedding(
    *,
    audio_path: str,
    muq_model,
    muq_model_name: str,
    sample_rate: int,
    target_fps: int,
    cache_dir: str = "",
    muq_chunk_seconds: float = 0.0,
    wav: torch.Tensor | None = None,
    wav_sample_rate: int | None = None,
    progress_postfix: tp.Optional[tp.Callable[[dict], None]] = None,
) -> torch.Tensor:
    cache_path = ""
    if cache_dir:
        cache_path = _resolve_cache_path(
            cache_dir=cache_dir,
            audio_path=audio_path,
            muq_model_name=muq_model_name,
            sample_rate=sample_rate,
            target_fps=target_fps,
        )
        cached = _load_cached_embedding(cache_path)
        if cached is not None:
            return cached

    embedding = _compute_aligned_embedding(
        audio_path=audio_path,
        muq_model=muq_model,
        sample_rate=sample_rate,
        target_fps=target_fps,
        muq_chunk_seconds=muq_chunk_seconds,
        wav=wav,
        wav_sample_rate=wav_sample_rate,
        progress_postfix=progress_postfix,
    )

    if cache_path:
        _save_cached_embedding(
            cache_path=cache_path,
            embedding=embedding,
            metadata={
                "version": MUQ_CACHE_VERSION,
                "audio": _audio_source_fingerprint(audio_path),
                "muq_model": muq_model_name,
                "sample_rate": int(sample_rate),
                "target_fps": int(target_fps),
                "muq_chunk_seconds": float(muq_chunk_seconds),
                "num_frames": int(embedding.shape[1]),
                "embed_dim": int(embedding.shape[2]),
            },
        )
    return embedding
