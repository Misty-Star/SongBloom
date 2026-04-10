"""批量训练并筛选多个 VQ codebook 候选。"""

from __future__ import annotations

import argparse
import json
import os

from .fit_vq_codebook_streaming import train_codebook
from .vq_codebook_dataset import resolve_audio_paths


def rank_candidates(
    rows: list[dict],
    max_dead_code_ratio: float,
    max_top1_share: float,
) -> list[dict]:
    valid = [
        row for row in rows
        if row["dead_code_ratio"] <= max_dead_code_ratio
        and row["top_1_usage_share"] <= max_top1_share
    ]
    return sorted(valid, key=lambda row: row["quantization_mse"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-dir", type=str, default="", help="原始音频目录")
    source.add_argument("--input-jsonl", type=str, default="", help="包含 `audio_path` 的 manifest")

    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-audios", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--target-fps", type=int, default=25)
    parser.add_argument("--muq-model", type=str, default="OpenMuQ/MuQ-large-msd-iter")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frames-per-audio", type=int, default=512)
    parser.add_argument("--heldout-ratio", type=float, default=0.1)
    parser.add_argument("--num-codes", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--train-steps", type=int, default=4000)
    parser.add_argument("--refresh-every", type=int, default=200)
    parser.add_argument("--muq-chunk-seconds", type=float, default=20.0)
    parser.add_argument("--distance-chunk-size", type=int, default=2048)
    parser.add_argument("--max-dead-code-ratio", type=float, default=0.05)
    parser.add_argument("--max-top1-share", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--frame-budgets", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    audio_paths = resolve_audio_paths(
        audio_dir=args.audio_dir,
        input_jsonl=args.input_jsonl,
        recursive=args.recursive,
        max_audios=args.max_audios,
    )
    if not audio_paths:
        raise RuntimeError("No audio files found for VQ codebook candidate search.")

    candidates: list[dict] = []
    for seed in args.seeds:
        for frame_budget in args.frame_budgets:
            candidate_name = f"seed{seed}_frames{frame_budget}"
            output_path = os.path.join(args.output_dir, f"{candidate_name}.pt")
            report = train_codebook(
                audio_paths=audio_paths,
                output_path=output_path,
                muq_model_name=args.muq_model,
                device=args.device,
                sample_rate=args.sample_rate,
                target_fps=args.target_fps,
                frames_per_audio=args.frames_per_audio,
                max_total_train_frames=frame_budget,
                max_total_heldout_frames=max(1, frame_budget // 4),
                num_codes=args.num_codes,
                heldout_ratio=args.heldout_ratio,
                batch_size=args.batch_size,
                train_steps=args.train_steps,
                refresh_every=args.refresh_every,
                seed=seed,
                muq_chunk_seconds=args.muq_chunk_seconds,
                distance_chunk_size=args.distance_chunk_size,
            )

            heldout_metrics = report.get("heldout_metrics") or report.get("train_metrics") or {}
            metadata = report.get("metadata", {})
            candidates.append(
                {
                    "name": candidate_name,
                    "seed": seed,
                    "frame_budget": frame_budget,
                    "codebook_path": report["codebook_path"],
                    "dead_code_ratio": heldout_metrics["dead_code_ratio"],
                    "top_1_usage_share": heldout_metrics["top_1_usage_share"],
                    "usage_entropy": heldout_metrics.get("usage_entropy", 0.0),
                    "quantization_mse": heldout_metrics["quantization_mse"],
                    "num_train_frames": metadata.get("num_train_frames", 0),
                    "num_heldout_frames": metadata.get("num_heldout_frames", 0),
                }
            )

    ranked = rank_candidates(
        candidates,
        max_dead_code_ratio=args.max_dead_code_ratio,
        max_top1_share=args.max_top1_share,
    )
    summary = {
        "candidates": candidates,
        "ranked_candidates": ranked,
        "best_candidate": ranked[0] if ranked else None,
    }

    summary_path = os.path.join(args.output_dir, "candidate_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
