"""串联 prepare_assets 和 build_dataset 的轻量入口。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def build_prepare_assets_cmd(args) -> list[str]:
    prepare_cmd = [
        sys.executable,
        "-m",
        "training.preprocess.prepare_assets",
        "--input-jsonl",
        args.input_jsonl,
        "--output-manifest",
        args.prepared_manifest,
        "--assets-dir",
        args.assets_dir,
        "--separator-cmd",
        args.separator_cmd,
        "--separator-python",
        args.separator_python,
        "--separator-model",
        args.separator_model,
        "--separator-model-flag",
        args.separator_model_flag,
        "--separator-output-format",
        args.separator_output_format,
        "--separator-timeout-sec",
        str(args.separator_timeout_sec),
        "--separator-max-files-per-batch",
        str(args.separator_max_files_per_batch),
        "--whisperx-cmd",
        args.whisperx_cmd,
        "--whisperx-model",
        args.whisperx_model,
        "--whisperx-device",
        args.whisperx_device,
        "--whisperx-compute-type",
        args.whisperx_compute_type,
        "--whisperx-batch-size",
        str(args.whisperx_batch_size),
        "--whisperx-timeout-sec",
        str(args.whisperx_timeout_sec),
        "--whisperx-max-files-per-batch",
        str(args.whisperx_max_files_per_batch),
        "--songformer-root",
        args.songformer_root,
        "--songformer-python",
        args.songformer_python,
        "--songformer-gpu-num",
        str(args.songformer_gpu_num),
        "--songformer-threads",
        str(args.songformer_threads),
        "--songformer-model",
        args.songformer_model,
        "--songformer-checkpoint",
        args.songformer_checkpoint,
        "--songformer-config",
        args.songformer_config,
        "--songformer-timeout-sec",
        str(args.songformer_timeout_sec),
    ]
    if args.separator_model_file_dir:
        prepare_cmd += ["--separator-model-file-dir", args.separator_model_file_dir]
    if args.separator_numba_cache_dir:
        prepare_cmd += ["--separator-numba-cache-dir", args.separator_numba_cache_dir]
    if args.language:
        prepare_cmd += ["--language", args.language]
    if args.workspace_dir:
        prepare_cmd += ["--workspace-dir", os.path.join(args.workspace_dir, "prepare_assets")]
    if args.skip_existing:
        prepare_cmd.append("--skip-existing")
    if args.keep_intermediate:
        prepare_cmd.append("--keep-intermediate")
    if args.skip_separation:
        prepare_cmd.append("--skip-separation")
    if args.skip_whisperx:
        prepare_cmd.append("--skip-whisperx")
    if args.skip_structure:
        prepare_cmd.append("--skip-structure")
    if args.songformer_no_rule_post:
        prepare_cmd.append("--songformer-no-rule-post")
    return prepare_cmd


def build_build_dataset_cmd(args, build_dataset_extra_args) -> list[str]:
    build_cmd = [
        sys.executable,
        "-m",
        "training.preprocess.build_dataset",
        "--input-jsonl",
        args.prepared_manifest,
        "--output-dir",
        args.output_dir,
        "--vq-ckpt",
        args.vq_ckpt,
    ]
    if args.workspace_dir:
        build_cmd += ["--workspace-dir", os.path.join(args.workspace_dir, "build_dataset")]
    if args.skip_existing:
        build_cmd.append("--skip-existing")
    if args.keep_intermediate:
        build_cmd.append("--keep-intermediate")
    if args.songformer_python:
        build_cmd += ["--songformer-python", args.songformer_python]
    build_cmd += build_dataset_extra_args
    return build_cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--prepared-manifest", type=str, required=True)
    parser.add_argument("--assets-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--workspace-dir", type=str, default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")

    parser.add_argument("--skip-separation", action="store_true")
    parser.add_argument("--separator-cmd", type=str, default="conda run -n audiosep-py310 audio-separator")
    parser.add_argument("--separator-python", type=str, default="")
    parser.add_argument("--separator-model", type=str, default="BS-Roformer-Viperx-1297")
    parser.add_argument("--separator-model-flag", type=str, default="auto", choices=["auto", "model_filename", "model_name"])
    parser.add_argument("--separator-output-format", type=str, default="FLAC")
    parser.add_argument("--separator-model-file-dir", type=str, default="")
    parser.add_argument("--separator-numba-cache-dir", type=str, default="")
    parser.add_argument("--separator-timeout-sec", type=float, default=0.0)
    parser.add_argument("--separator-max-files-per-batch", type=int, default=32)

    parser.add_argument("--skip-whisperx", action="store_true")
    parser.add_argument("--whisperx-cmd", type=str, default="conda run -n whisperx whisperx")
    parser.add_argument("--whisperx-model", type=str, default="large-v3")
    parser.add_argument("--whisperx-device", type=str, default="cuda")
    parser.add_argument("--whisperx-compute-type", type=str, default="float16")
    parser.add_argument("--whisperx-batch-size", type=int, default=8)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--whisperx-timeout-sec", type=float, default=0.0)
    parser.add_argument("--whisperx-max-files-per-batch", type=int, default=32)
    parser.add_argument("--skip-structure", action="store_true")
    parser.add_argument("--songformer-python", type=str, default="")
    parser.add_argument("--songformer-root", type=str, default="third_party/SongFormer")
    parser.add_argument("--songformer-gpu-num", type=int, default=1)
    parser.add_argument("--songformer-threads", type=int, default=1)
    parser.add_argument("--songformer-model", type=str, default="SongFormer")
    parser.add_argument("--songformer-checkpoint", type=str, default="SongFormer.safetensors")
    parser.add_argument("--songformer-config", type=str, default="SongFormer.yaml")
    parser.add_argument("--songformer-no-rule-post", action="store_true")
    parser.add_argument("--songformer-timeout-sec", type=float, default=0.0)
    args, build_dataset_extra_args = parser.parse_known_args()

    prepare_cmd = build_prepare_assets_cmd(args)
    subprocess.run(prepare_cmd, check=True)

    build_cmd = build_build_dataset_cmd(args, build_dataset_extra_args)
    subprocess.run(build_cmd, check=True)


if __name__ == "__main__":
    main()
