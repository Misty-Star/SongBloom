"""串联 prepare_assets 和 build_dataset 的轻量入口。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


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
    parser.add_argument("--separator-model", type=str, default="BS-Roformer-Viperx-1297")
    parser.add_argument("--separator-model-flag", type=str, default="auto", choices=["auto", "model_filename", "model_name"])
    parser.add_argument("--separator-output-format", type=str, default="FLAC")
    parser.add_argument("--separator-model-file-dir", type=str, default="")

    parser.add_argument("--skip-whisperx", action="store_true")
    parser.add_argument("--whisperx-cmd", type=str, default="conda run -n whisperx whisperx")
    parser.add_argument("--whisperx-model", type=str, default="large-v3")
    parser.add_argument("--whisperx-device", type=str, default="cuda")
    parser.add_argument("--whisperx-compute-type", type=str, default="float16")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--songformer-python", type=str, default="")
    args, build_dataset_extra_args = parser.parse_known_args()

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
        "--separator-model",
        args.separator_model,
        "--separator-model-flag",
        args.separator_model_flag,
        "--separator-output-format",
        args.separator_output_format,
        "--whisperx-cmd",
        args.whisperx_cmd,
        "--whisperx-model",
        args.whisperx_model,
        "--whisperx-device",
        args.whisperx_device,
        "--whisperx-compute-type",
        args.whisperx_compute_type,
    ]
    if args.separator_model_file_dir:
        prepare_cmd += ["--separator-model-file-dir", args.separator_model_file_dir]
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
    subprocess.run(prepare_cmd, check=True)

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
    subprocess.run(build_cmd, check=True)


if __name__ == "__main__":
    main()
