"""用 K-Means 拟合 SongBloom 预处理所需的单层 VQ codebook。

这个脚本是一个工程化替代方案，只负责生成与当前
`training/preprocess/extract_sketch.py` 兼容的 `vq_codebook.pt`：

    {"codebook.weight": Tensor[num_codes, embed_dim]}

为避免与未来更正式的 VQ / RVQ 训练方案混淆，脚本名显式带有 `kmeans`。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import typing as tp

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from .extract_sketch import extract_muq_embeddings, load_muq_model


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


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


def sample_frames_from_embedding(
    embedding: torch.Tensor,
    max_frames: int,
    rng: random.Random,
) -> torch.Tensor:
    # embedding: (1, T, D)
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
    device: str,
    seed: int,
) -> torch.Tensor:
    rng = random.Random(seed)
    collected: list[torch.Tensor] = []
    total_frames = 0

    for audio_path in tqdm(audio_paths, desc="Collecting MuQ frames"):
        wav, sr = torchaudio.load(audio_path)
        if sr != sample_rate:
            wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)

        embedding = extract_muq_embeddings(muq_model, wav, sample_rate)
        target_frames = int(wav.shape[-1] / sample_rate * target_fps)
        if target_frames > 0 and embedding.shape[1] != target_frames:
            embedding = F.interpolate(
                embedding.transpose(1, 2),
                size=target_frames,
                mode="nearest",
            ).transpose(1, 2)

        sampled = sample_frames_from_embedding(embedding, frames_per_audio, rng)
        if sampled.numel() == 0:
            continue

        remaining = max_total_frames - total_frames
        if remaining <= 0:
            break
        if sampled.shape[0] > remaining:
            sampled = sampled[:remaining]

        collected.append(sampled)
        total_frames += sampled.shape[0]

        if total_frames >= max_total_frames:
            break

    if not collected:
        raise RuntimeError("No MuQ embedding frames collected; check audio input and MuQ setup.")
    return torch.cat(collected, dim=0)


def minibatch_kmeans(
    samples: torch.Tensor,
    num_clusters: int,
    batch_size: int,
    num_steps: int,
    device: str,
    seed: int,
    final_refine_pass: bool = True,
) -> torch.Tensor:
    if samples.ndim != 2:
        raise ValueError(f"`samples` must be 2D, got {tuple(samples.shape)}")
    if samples.shape[0] < num_clusters:
        raise ValueError(
            f"Need at least {num_clusters} sampled frames, but only got {samples.shape[0]}."
        )

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    init_indices = torch.randperm(samples.shape[0], generator=rng)[:num_clusters]

    working_samples = samples.to(device=device, dtype=torch.float32)
    centers = working_samples[init_indices.to(working_samples.device)].clone()
    counts = torch.ones(num_clusters, device=working_samples.device, dtype=torch.float32)

    for _step in tqdm(range(num_steps), desc="MiniBatch K-Means"):
        batch_indices = torch.randint(
            low=0,
            high=working_samples.shape[0],
            size=(batch_size,),
            generator=rng,
        )
        batch = working_samples[batch_indices.to(working_samples.device)]
        distances = torch.cdist(batch, centers)
        assignments = distances.argmin(dim=-1)

        unique_ids = assignments.unique()
        for cluster_id in unique_ids.tolist():
            mask = assignments == cluster_id
            points = batch[mask]
            if points.numel() == 0:
                continue
            counts[cluster_id] += points.shape[0]
            eta = points.shape[0] / counts[cluster_id]
            centers[cluster_id] = (1.0 - eta) * centers[cluster_id] + eta * points.mean(dim=0)

    if final_refine_pass:
        sums = torch.zeros_like(centers)
        new_counts = torch.zeros(num_clusters, device=working_samples.device, dtype=torch.float32)
        chunk = max(1, batch_size)
        for start in tqdm(range(0, working_samples.shape[0], chunk), desc="Final refine pass"):
            batch = working_samples[start:start + chunk]
            assignments = torch.cdist(batch, centers).argmin(dim=-1)
            unique_ids = assignments.unique()
            for cluster_id in unique_ids.tolist():
                mask = assignments == cluster_id
                points = batch[mask]
                if points.numel() == 0:
                    continue
                sums[cluster_id] += points.sum(dim=0)
                new_counts[cluster_id] += points.shape[0]

        empty = new_counts == 0
        if empty.any():
            refill = torch.randperm(working_samples.shape[0], device=working_samples.device)[: int(empty.sum().item())]
            centers[empty] = working_samples[refill]
            new_counts[empty] = 1.0

        centers = sums / new_counts.unsqueeze(-1)

    return centers.detach().cpu()


def save_codebook(output_path: str, centers: torch.Tensor, metadata: dict) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({"codebook.weight": centers.float().contiguous()}, output_path)

    meta_path = os.path.splitext(output_path)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-dir", type=str, default="", help="原始音频目录")
    source.add_argument("--input-jsonl", type=str, default="", help="包含 `audio_path` 的 manifest")

    parser.add_argument("--output-path", type=str, required=True, help="输出 `vq_codebook.pt` 路径")
    parser.add_argument("--recursive", action="store_true", help="递归扫描 audio-dir")
    parser.add_argument("--max-audios", type=int, default=0, help="最多使用多少首歌，0 表示不限制")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--target-fps", type=int, default=25)
    parser.add_argument("--muq-model", type=str, default="OpenMuQ/MuQ-large-msd-iter")
    parser.add_argument("--device", type=str, default="cuda:0", help="MuQ 提取与 K-Means 运行设备")
    parser.add_argument("--frames-per-audio", type=int, default=512, help="每首歌最多采样多少帧")
    parser.add_argument("--max-total-frames", type=int, default=50000, help="全局最多采样多少帧")
    parser.add_argument("--num-codes", type=int, default=16384)
    parser.add_argument("--kmeans-batch-size", type=int, default=1024)
    parser.add_argument("--kmeans-steps", type=int, default=2000)
    parser.add_argument("--no-final-refine-pass", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.audio_dir:
        audio_paths = list_audio_files(args.audio_dir, recursive=args.recursive)
    else:
        audio_paths = load_manifest_audio_paths(args.input_jsonl)

    if args.max_audios > 0:
        audio_paths = audio_paths[:args.max_audios]
    if not audio_paths:
        raise RuntimeError("No audio files found for K-Means codebook fitting.")

    muq_model = load_muq_model(args.muq_model, args.device)
    samples = collect_embedding_samples(
        audio_paths=audio_paths,
        muq_model=muq_model,
        sample_rate=args.sample_rate,
        target_fps=args.target_fps,
        frames_per_audio=args.frames_per_audio,
        max_total_frames=args.max_total_frames,
        device=args.device,
        seed=args.seed,
    )
    centers = minibatch_kmeans(
        samples=samples,
        num_clusters=args.num_codes,
        batch_size=args.kmeans_batch_size,
        num_steps=args.kmeans_steps,
        device=args.device,
        seed=args.seed,
        final_refine_pass=not args.no_final_refine_pass,
    )
    save_codebook(
        output_path=args.output_path,
        centers=centers,
        metadata={
            "scheme": "kmeans",
            "num_codes": int(centers.shape[0]),
            "embed_dim": int(centers.shape[1]),
            "num_audio_files": len(audio_paths),
            "num_sampled_frames": int(samples.shape[0]),
            "muq_model": args.muq_model,
            "sample_rate": args.sample_rate,
            "target_fps": args.target_fps,
            "frames_per_audio": args.frames_per_audio,
            "max_total_frames": args.max_total_frames,
            "kmeans_batch_size": args.kmeans_batch_size,
            "kmeans_steps": args.kmeans_steps,
            "final_refine_pass": not args.no_final_refine_pass,
            "seed": args.seed,
        },
    )

    print(
        json.dumps(
            {
                "output_path": args.output_path,
                "num_codes": int(centers.shape[0]),
                "embed_dim": int(centers.shape[1]),
                "num_audio_files": len(audio_paths),
                "num_sampled_frames": int(samples.shape[0]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
