"""评估单层 VQ codebook 的量化质量。"""

from __future__ import annotations

import argparse
import json
import math
import os
import typing as tp

import torch

from .vq_codebook_dataset import (
    collect_embedding_samples,
    resolve_audio_paths,
)


def load_muq_model(model_name: str, device: str):
    from .extract_sketch import load_muq_model as _load_muq_model

    return _load_muq_model(model_name, device)


def load_codebook(codebook_path: str) -> torch.Tensor:
    state = torch.load(codebook_path, map_location="cpu")
    return state["codebook.weight"].float().contiguous()


def iter_sample_chunks(samples: torch.Tensor, distance_chunk_size: int) -> tp.Iterable[torch.Tensor]:
    if distance_chunk_size <= 0 or samples.shape[0] <= distance_chunk_size:
        yield samples
        return
    for start in range(0, samples.shape[0], distance_chunk_size):
        yield samples[start:start + distance_chunk_size]


def assign_codebook(
    samples: torch.Tensor,
    codebook: torch.Tensor,
    distance_chunk_size: int = 0,
) -> torch.Tensor:
    codebook = codebook.float()
    assignments: list[torch.Tensor] = []
    for chunk in iter_sample_chunks(samples, distance_chunk_size):
        distances = torch.cdist(chunk.float(), codebook)
        assignments.append(distances.argmin(dim=-1).cpu())
    return torch.cat(assignments, dim=0) if assignments else torch.empty(0, dtype=torch.long)


def compute_quantization_mse(
    samples: torch.Tensor,
    codebook: torch.Tensor,
    assignments: torch.Tensor | None = None,
    distance_chunk_size: int = 0,
) -> float:
    if assignments is None:
        assignments = assign_codebook(samples, codebook, distance_chunk_size=distance_chunk_size)
    reconstructed = codebook[assignments]
    return float(torch.mean((samples.float() - reconstructed.float()) ** 2).item())


def summarize_assignments(assignments: torch.Tensor, num_codes: int) -> dict:
    counts = torch.bincount(assignments.detach().cpu(), minlength=num_codes).float()
    return summarize_counts(counts)


def summarize_counts(counts: torch.Tensor) -> dict:
    counts = counts.detach().cpu().float()
    total = counts.sum().clamp_min(1.0)
    probs = counts / total
    nonzero = probs[probs > 0]
    entropy = float((-(nonzero * nonzero.log())).sum().item()) if nonzero.numel() else 0.0
    return {
        "dead_code_ratio": float((counts == 0).float().mean().item()),
        "top_1_usage_share": float(probs.max().item()) if probs.numel() else 0.0,
        "usage_entropy": entropy,
        "perplexity": float(math.exp(entropy)) if entropy > 0 else 1.0,
    }


def evaluate_codebook_frames(
    samples: torch.Tensor,
    codebook: torch.Tensor,
    distance_chunk_size: int = 0,
) -> dict:
    codebook = codebook.float()
    if distance_chunk_size <= 0:
        assignments = assign_codebook(samples, codebook)
        report = summarize_assignments(assignments, num_codes=codebook.shape[0])
        report["quantization_mse"] = compute_quantization_mse(samples, codebook, assignments=assignments)
    else:
        counts = torch.zeros(codebook.shape[0], dtype=torch.float32)
        squared_error_sum = 0.0
        total_values = 0
        for chunk in iter_sample_chunks(samples, distance_chunk_size):
            chunk = chunk.float()
            assignments = assign_codebook(chunk, codebook, distance_chunk_size=0)
            counts += torch.bincount(assignments, minlength=codebook.shape[0]).float()
            reconstructed = codebook[assignments]
            squared_error_sum += float(torch.sum((chunk - reconstructed.float()) ** 2).item())
            total_values += int(chunk.numel())
        report = summarize_counts(counts)
        report["quantization_mse"] = squared_error_sum / max(total_values, 1)
    report["num_frames"] = int(samples.shape[0])
    report["embed_dim"] = int(samples.shape[1])
    return report


def save_report(report_path: str, report: dict) -> None:
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-dir", type=str, default="", help="原始音频目录")
    source.add_argument("--input-jsonl", type=str, default="", help="包含 `audio_path` 的 manifest")

    parser.add_argument("--codebook-path", type=str, required=True)
    parser.add_argument("--report-path", type=str, default="")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-audios", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--target-fps", type=int, default=25)
    parser.add_argument("--muq-model", type=str, default="OpenMuQ/MuQ-large-msd-iter")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frames-per-audio", type=int, default=512)
    parser.add_argument("--max-total-frames", type=int, default=50000)
    parser.add_argument("--num-codes", type=int, default=16384)
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
    if not audio_paths:
        raise RuntimeError("No audio files found for VQ codebook evaluation.")

    muq_model = load_muq_model(args.muq_model, args.device)
    samples = collect_embedding_samples(
        audio_paths=audio_paths,
        muq_model=muq_model,
        sample_rate=args.sample_rate,
        target_fps=args.target_fps,
        frames_per_audio=args.frames_per_audio,
        max_total_frames=args.max_total_frames,
        seed=args.seed,
        muq_chunk_seconds=args.muq_chunk_seconds,
        progress_desc="Collecting evaluation MuQ frames",
    )
    codebook = load_codebook(args.codebook_path)
    if args.num_codes and codebook.shape[0] != args.num_codes:
        raise ValueError(
            f"Codebook size mismatch: expected {args.num_codes}, got {codebook.shape[0]}."
        )

    report = evaluate_codebook_frames(
        samples,
        codebook,
        distance_chunk_size=args.distance_chunk_size,
    )
    report["codebook_path"] = args.codebook_path
    report_path = args.report_path or (os.path.splitext(args.codebook_path)[0] + ".evaluation.report.json")
    save_report(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
