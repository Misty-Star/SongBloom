"""效果优先的单层 VQ codebook 训练入口。"""

from __future__ import annotations

import argparse
import json
import os

import torch

from .evaluate_vq_codebook import evaluate_codebook_frames, save_report
from .vq_codebook_dataset import (
    collect_embedding_samples,
    resolve_audio_paths,
    split_audio_paths,
)
from .vq_codebook_trainer import StreamingKMeans


def load_muq_model(model_name: str, device: str):
    from .extract_sketch import load_muq_model as _load_muq_model

    return _load_muq_model(model_name, device)


def save_codebook(output_path: str, centers: torch.Tensor, metadata: dict) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save({"codebook.weight": centers.float().contiguous()}, output_path)
    with open(os.path.splitext(output_path)[0] + ".meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def train_codebook(
    *,
    audio_paths: list[str],
    output_path: str,
    muq_model_name: str,
    device: str,
    sample_rate: int,
    target_fps: int,
    frames_per_audio: int,
    max_total_train_frames: int,
    max_total_heldout_frames: int,
    num_codes: int,
    heldout_ratio: float,
    batch_size: int,
    train_steps: int,
    refresh_every: int,
    seed: int,
    muq_chunk_seconds: float,
    distance_chunk_size: int,
) -> dict:
    if not audio_paths:
        raise RuntimeError("No audio files found for streaming VQ codebook fitting.")

    train_audio_paths, heldout_audio_paths = split_audio_paths(audio_paths, heldout_ratio, seed)
    muq_model = load_muq_model(muq_model_name, device)

    train_frames = collect_embedding_samples(
        audio_paths=train_audio_paths,
        muq_model=muq_model,
        sample_rate=sample_rate,
        target_fps=target_fps,
        frames_per_audio=frames_per_audio,
        max_total_frames=max_total_train_frames,
        seed=seed,
        muq_chunk_seconds=muq_chunk_seconds,
        progress_desc="Collecting train MuQ frames",
    )
    if train_frames.shape[0] < num_codes:
        raise ValueError(f"Need at least {num_codes} sampled frames, but only got {train_frames.shape[0]}.")

    heldout_frames = None
    if heldout_audio_paths:
        heldout_frames = collect_embedding_samples(
            audio_paths=heldout_audio_paths,
            muq_model=muq_model,
            sample_rate=sample_rate,
            target_fps=target_fps,
            frames_per_audio=frames_per_audio,
            max_total_frames=max_total_heldout_frames,
            seed=seed + 1,
            muq_chunk_seconds=muq_chunk_seconds,
            progress_desc="Collecting heldout MuQ frames",
        )

    del muq_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    trainer = StreamingKMeans(
        num_codes=num_codes,
        embed_dim=int(train_frames.shape[1]),
        device=device,
        seed=seed,
    )
    trainer.fit(
        samples=train_frames,
        batch_size=batch_size,
        num_steps=train_steps,
        refresh_every=refresh_every,
    )
    centers = trainer.refine_full(train_frames, batch_size=batch_size)

    metadata = {
        "scheme": "streaming_kmeans",
        "num_codes": int(centers.shape[0]),
        "embed_dim": int(centers.shape[1]),
        "num_audio_files": len(audio_paths),
        "num_train_audio_files": len(train_audio_paths),
        "num_heldout_audio_files": len(heldout_audio_paths),
        "num_train_frames": int(train_frames.shape[0]),
        "num_heldout_frames": int(heldout_frames.shape[0]) if heldout_frames is not None else 0,
        "muq_model": muq_model_name,
        "sample_rate": sample_rate,
        "target_fps": target_fps,
        "frames_per_audio": frames_per_audio,
        "max_total_train_frames": max_total_train_frames,
        "max_total_heldout_frames": max_total_heldout_frames,
        "batch_size": batch_size,
        "train_steps": train_steps,
        "refresh_every": refresh_every,
        "heldout_ratio": heldout_ratio,
        "seed": seed,
    }
    save_codebook(output_path, centers=centers, metadata=metadata)

    train_metrics = evaluate_codebook_frames(
        train_frames,
        centers,
        distance_chunk_size=distance_chunk_size,
    )
    heldout_metrics = (
        evaluate_codebook_frames(
            heldout_frames,
            centers,
            distance_chunk_size=distance_chunk_size,
        )
        if heldout_frames is not None and heldout_frames.numel() > 0
        else {}
    )
    report = {
        "codebook_path": output_path,
        "metadata": metadata,
        "train_metrics": train_metrics,
        "heldout_metrics": heldout_metrics,
    }
    save_report(os.path.splitext(output_path)[0] + ".report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-dir", type=str, default="", help="原始音频目录")
    source.add_argument("--input-jsonl", type=str, default="", help="包含 `audio_path` 的 manifest")

    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-audios", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--target-fps", type=int, default=25)
    parser.add_argument("--muq-model", type=str, default="OpenMuQ/MuQ-large-msd-iter")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frames-per-audio", type=int, default=512)
    parser.add_argument("--max-total-train-frames", type=int, default=200000)
    parser.add_argument("--max-total-heldout-frames", type=int, default=50000)
    parser.add_argument("--num-codes", type=int, default=16384)
    parser.add_argument("--heldout-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--train-steps", type=int, default=4000)
    parser.add_argument("--refresh-every", type=int, default=200)
    parser.add_argument("--muq-chunk-seconds", type=float, default=20.0)
    parser.add_argument("--distance-chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_paths = resolve_audio_paths(
        audio_dir=args.audio_dir,
        input_jsonl=args.input_jsonl,
        recursive=args.recursive,
        max_audios=args.max_audios,
    )
    report = train_codebook(
        audio_paths=audio_paths,
        output_path=args.output_path,
        muq_model_name=args.muq_model,
        device=args.device,
        sample_rate=args.sample_rate,
        target_fps=args.target_fps,
        frames_per_audio=args.frames_per_audio,
        max_total_train_frames=args.max_total_train_frames,
        max_total_heldout_frames=args.max_total_heldout_frames,
        num_codes=args.num_codes,
        heldout_ratio=args.heldout_ratio,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        refresh_every=args.refresh_every,
        seed=args.seed,
        muq_chunk_seconds=args.muq_chunk_seconds,
        distance_chunk_size=args.distance_chunk_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
