"""为 SongBloom 训练预处理准备 stems 和 WhisperX 资产。

该脚本面向“独立工具环境 + 派生 manifest”工作流：
1. 用 audio-separator 生成 `vocals_path` / `no_vocals_path`
2. 用 WhisperX 生成 `whisperx_json`
3. 将新增字段写入派生 manifest，供 `build_dataset.py` 继续消费
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback
import typing as tp

import torchaudio

from .align_lyrics import run_whisperx_cli
from .common import (
    AUDIO_EXTENSIONS,
    command_exists,
    ensure_dir,
    find_file_with_stem,
    format_seconds,
    get_audio_path,
    get_item_id,
    load_jsonl,
    log_progress,
    normalize_whitespace,
    write_jsonl,
)
from .extract_structure import prepare_structure_assets
from .structure_assets import compute_prepare_assets_status, merge_structure_preparation_results


SEPARATOR_MODEL_ALIASES = {
    "BS-Roformer-Viperx-1297": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
}


def convert_audio(
    input_path: str,
    output_path: str,
    sample_rate: int,
    mono: bool = False,
) -> tp.Tuple[str, float]:
    wav, sr = torchaudio.load(input_path)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
    if mono and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif not mono and wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    duration = wav.shape[-1] / sample_rate
    ensure_dir(os.path.dirname(output_path))
    torchaudio.save(output_path, wav, sample_rate)
    return output_path, duration


def resolve_command_name(command: str) -> str:
    argv = shlex.split(command)
    if not argv:
        raise ValueError("command must not be empty")
    return argv[0]


def separator_suffix(output_format: str) -> str:
    return "." + output_format.strip().lower().lstrip(".")


def expected_separator_outputs(output_dir: str, output_format: str) -> tp.Tuple[str, str]:
    suffix = separator_suffix(output_format)
    return (
        os.path.join(output_dir, f"vocals{suffix}"),
        os.path.join(output_dir, f"no_vocals{suffix}"),
    )


def resolve_separator_model(model: str) -> str:
    normalized = normalize_whitespace(model)
    if not normalized:
        raise ValueError("`separator_model` is empty.")
    return SEPARATOR_MODEL_ALIASES.get(normalized, model)


def default_separator_model_dir() -> str:
    return os.environ.get("AUDIO_SEPARATOR_MODEL_DIR", "/tmp/audio-separator-models")


def get_separator_model_dir(args) -> str:
    return os.path.abspath(args.separator_model_file_dir or default_separator_model_dir())


def get_separator_numba_cache_dir(args) -> str:
    return os.path.abspath(args.separator_numba_cache_dir or os.path.join(args.assets_dir, "_numba_cache"))


def companion_separator_assets(model_filename: str) -> tp.List[str]:
    base, ext = os.path.splitext(model_filename)
    if ext.lower() == ".ckpt":
        return [f"{base}.yaml"]
    return []


def host_resolves(hostname: str) -> bool:
    try:
        socket.getaddrinfo(hostname, 443)
    except socket.gaierror:
        return False
    return True


def describe_exception(exc: Exception) -> str:
    if isinstance(exc, subprocess.TimeoutExpired):
        stderr = normalize_whitespace((exc.stderr or "") if isinstance(exc.stderr, str) else "")
        stdout = normalize_whitespace((exc.stdout or "") if isinstance(exc.stdout, str) else "")
        detail = stderr or stdout
        message = f"timed out after {exc.timeout}s"
        return f"{message}. {detail}" if detail else message
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = normalize_whitespace(exc.stderr or "")
        stdout = normalize_whitespace(exc.stdout or "")
        details = stderr or stdout
        if details:
            return f"{exc}. {details}"
    return str(exc)


def find_separator_outputs(output_dir: str, output_format: str) -> tp.Tuple[str, str]:
    expected_vocals, expected_no_vocals = expected_separator_outputs(output_dir, output_format)
    if os.path.exists(expected_vocals) and os.path.exists(expected_no_vocals):
        return os.path.abspath(expected_vocals), os.path.abspath(expected_no_vocals)

    suffixes = [separator_suffix(output_format)] + [suffix for suffix in AUDIO_EXTENSIONS if suffix != separator_suffix(output_format)]
    vocals_path = find_file_with_stem(output_dir, "vocals", suffixes)
    accompaniment_path = (
        find_file_with_stem(output_dir, "no_vocals", suffixes)
        or find_file_with_stem(output_dir, "instrumental", suffixes)
    )

    if vocals_path and accompaniment_path:
        return os.path.abspath(vocals_path), os.path.abspath(accompaniment_path)

    for root, _dirs, files in os.walk(output_dir):
        for name in files:
            stem = os.path.splitext(name)[0].lower()
            path = os.path.abspath(os.path.join(root, name))
            if stem == "vocals":
                vocals_path = vocals_path or path
            elif stem in {"no_vocals", "instrumental"}:
                accompaniment_path = accompaniment_path or path
    if not vocals_path or not accompaniment_path:
        raise FileNotFoundError(
            f"Could not resolve audio-separator outputs under {output_dir}. "
            "Expected stems named vocals/no_vocals or vocals/instrumental."
        )
    return os.path.abspath(vocals_path), os.path.abspath(accompaniment_path)


def run_audio_separator(audio_path: str, output_dir: str, args) -> tp.Tuple[str, str]:
    command = shlex.split(args.separator_cmd)
    if not command:
        raise ValueError("`separator_cmd` is empty.")
    model_flag = detect_separator_model_flag(args.separator_cmd, args.separator_model_flag)
    model_value = args.separator_model if model_flag == "--model_name" else resolve_separator_model(args.separator_model)

    argv = list(command) + [
        audio_path,
        model_flag,
        model_value,
        "--output_format",
        args.separator_output_format,
        "--output_dir",
        output_dir,
        "--custom_output_names",
        json.dumps({"Vocals": "vocals", "Instrumental": "no_vocals"}),
    ]
    if args.separator_model_file_dir:
        argv += ["--model_file_dir", args.separator_model_file_dir]

    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", get_separator_numba_cache_dir(args))
    ensure_dir(env["NUMBA_CACHE_DIR"])

    subprocess.run(
        argv,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=None if not args.separator_timeout_sec or args.separator_timeout_sec <= 0 else args.separator_timeout_sec,
    )
    return find_separator_outputs(output_dir, args.separator_output_format)


def expected_whisperx_output(audio_path: str, output_dir: str) -> str:
    output_name = os.path.splitext(os.path.basename(audio_path))[0] + ".json"
    return os.path.abspath(os.path.join(output_dir, output_name))


def expected_structure_output(sample_id: str, assets_dir: str) -> str:
    return os.path.abspath(os.path.join(assets_dir, sample_id, "structure", sample_id + ".json"))


def detect_separator_model_flag(separator_cmd: str, preferred_flag: str) -> str:
    if preferred_flag != "auto":
        return f"--{preferred_flag.lstrip('-')}"

    command = shlex.split(separator_cmd)
    if not command:
        raise ValueError("`separator_cmd` is empty.")
    try:
        completed = subprocess.run(
            list(command) + ["--help"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        help_text = f"{completed.stdout}\n{completed.stderr}"
    except Exception:
        return "--model_filename"

    if "--model_filename" in help_text:
        return "--model_filename"
    if "--model_name" in help_text:
        return "--model_name"
    return "--model_filename"


def collect_preflight_issues(items: tp.Sequence[dict], args) -> tp.List[str]:
    issues: tp.List[str] = []
    model_flag = detect_separator_model_flag(args.separator_cmd, args.separator_model_flag)
    resolved_model = resolve_separator_model(args.separator_model)
    model_dir = get_separator_model_dir(args)

    needs_separator = False
    needs_whisperx = False
    needs_structure = False
    for item in items:
        sample_id = get_item_id(item)
        cached_vocals_path = None
        if not args.skip_separation and not (item.get("vocals_path") and item.get("no_vocals_path")):
            cached_separator_dir = os.path.join(args.assets_dir, sample_id, "separator")
            try:
                cached_vocals_path, _cached_no_vocals_path = find_separator_outputs(
                    cached_separator_dir,
                    args.separator_output_format,
                )
            except FileNotFoundError:
                needs_separator = True

        if not args.skip_whisperx and not item.get("whisperx_json"):
            whisperx_dir = os.path.join(args.assets_dir, sample_id, "whisperx")
            whisperx_inputs = [
                str(item.get("vocals_path")) if item.get("vocals_path") else "",
                cached_vocals_path or "",
                get_audio_path(item),
            ]
            if not any(
                candidate and os.path.exists(expected_whisperx_output(candidate, whisperx_dir))
                for candidate in whisperx_inputs
            ):
                needs_whisperx = True

        if (
            not args.skip_structure
            and not item.get("structure_json")
            and not os.path.exists(expected_structure_output(sample_id, args.assets_dir))
        ):
            needs_structure = True

    if needs_separator:
        separator_exec = resolve_command_name(args.separator_cmd)
        if not command_exists(separator_exec):
            issues.append(
                f"audio-separator launcher {separator_exec!r} is not in PATH. "
                "Install it in a dedicated environment or pass --separator-cmd."
            )
        elif model_flag == "--model_filename":
            missing_assets = [
                name for name in [resolved_model] + companion_separator_assets(resolved_model)
                if not os.path.exists(os.path.join(model_dir, name))
            ]
            if missing_assets and not host_resolves("github.com"):
                issues.append(
                    "audio-separator model assets are incomplete and github.com cannot be resolved. "
                    f"Missing under {model_dir}: {', '.join(missing_assets)}. "
                    "For BS-Roformer-Viperx-1297 you need both the .ckpt and companion .yaml files."
                )

    if needs_whisperx:
        whisperx_exec = resolve_command_name(args.whisperx_cmd)
        if not command_exists(whisperx_exec):
            issues.append(
                f"WhisperX launcher {whisperx_exec!r} is not in PATH. "
                "Install it in a dedicated environment or pass --whisperx-cmd."
            )

    if needs_structure:
        songformer_exec = resolve_command_name(args.songformer_python)
        infer_path = os.path.join(args.songformer_root, "src", "SongFormer", "infer", "infer.py")
        if not command_exists(songformer_exec):
            issues.append(
                f"SongFormer python launcher {songformer_exec!r} is not in PATH. "
                "Install SongFormer in a dedicated environment or pass --songformer-python."
            )
        if not os.path.exists(infer_path):
            issues.append(
                f"SongFormer infer script is missing: {infer_path}. "
                "Check --songformer-root or initialize third_party/SongFormer."
            )

    return issues


def normalize_record_paths(record: dict) -> dict:
    normalized = dict(record)
    for key in ("audio_path", "vocals_path", "no_vocals_path", "whisperx_json"):
        if normalized.get(key):
            normalized[key] = os.path.abspath(str(normalized[key]))
    return normalized


def prepare_item(item: dict, args) -> tp.Tuple[dict, dict]:
    sample_id = get_item_id(item)
    record = normalize_record_paths(item)
    asset_sample_dir = ensure_dir(os.path.join(args.assets_dir, sample_id))
    workspace_sample_dir = ensure_dir(os.path.join(args.workspace_dir, sample_id))

    separator_status = "provided" if record.get("vocals_path") and record.get("no_vocals_path") else "skipped"
    whisperx_status = "provided" if record.get("whisperx_json") else "skipped"
    errors: tp.List[dict] = []

    if not args.skip_separation and not (record.get("vocals_path") and record.get("no_vocals_path")):
        separator_dir = ensure_dir(os.path.join(asset_sample_dir, "separator"))
        try:
            reused_outputs = False
            if args.skip_existing:
                try:
                    vocals_path, no_vocals_path = find_separator_outputs(separator_dir, args.separator_output_format)
                    reused_outputs = True
                    separator_status = "skipped_existing"
                    print(f"[prepare_assets] {sample_id} reuse separator outputs", flush=True)
                except FileNotFoundError:
                    reused_outputs = False
            if not reused_outputs:
                print(f"[prepare_assets] {sample_id} run audio-separator", flush=True)
                standardized_audio, _duration = convert_audio(
                    input_path=get_audio_path(record),
                    output_path=os.path.join(workspace_sample_dir, "separator_input.flac"),
                    sample_rate=args.asset_sample_rate,
                    mono=False,
                )
                vocals_path, no_vocals_path = run_audio_separator(standardized_audio, separator_dir, args)
                separator_status = "ok"
            record["vocals_path"] = vocals_path
            record["no_vocals_path"] = no_vocals_path
        except Exception as exc:
            separator_status = "error"
            errors.append(
                {
                    "stage": "separator",
                    "error": describe_exception(exc),
                    "traceback": traceback.format_exc(limit=3),
                }
            )

    if not args.skip_whisperx and not record.get("whisperx_json"):
        whisperx_dir = ensure_dir(os.path.join(asset_sample_dir, "whisperx"))
        asr_input = str(record.get("vocals_path") or get_audio_path(record))
        try:
            existing_json = expected_whisperx_output(asr_input, whisperx_dir)
            if args.skip_existing and os.path.exists(existing_json):
                whisperx_json = existing_json
                whisperx_status = "skipped_existing"
                print(f"[prepare_assets] {sample_id} reuse whisperx output", flush=True)
            else:
                print(f"[prepare_assets] {sample_id} run whisperx", flush=True)
                whisperx_json = run_whisperx_cli(
                    audio_path=asr_input,
                    output_dir=whisperx_dir,
                    whisperx_cmd=args.whisperx_cmd,
                    language=record.get("language", args.language),
                    model=args.whisperx_model,
                    device=args.whisperx_device,
                    compute_type=args.whisperx_compute_type,
                    timeout_sec=args.whisperx_timeout_sec,
                )
                whisperx_status = "ok"
            record["whisperx_json"] = os.path.abspath(whisperx_json)
        except Exception as exc:
            whisperx_status = "error"
            errors.append(
                {
                    "stage": "whisperx",
                    "error": describe_exception(exc),
                    "traceback": traceback.format_exc(limit=3),
                }
            )

    if not args.keep_intermediate:
        shutil.rmtree(workspace_sample_dir, ignore_errors=True)

    report = {
        "id": sample_id,
        "separator_status": separator_status,
        "whisperx_status": whisperx_status,
    }
    if record.get("vocals_path"):
        report["vocals_path"] = record["vocals_path"]
    if record.get("no_vocals_path"):
        report["no_vocals_path"] = record["no_vocals_path"]
    if record.get("whisperx_json"):
        report["whisperx_json"] = record["whisperx_json"]
    if errors:
        report["errors"] = errors
    report["status"] = compute_prepare_assets_status(record=record, report=report)
    return record, report


def build_default_report_path(output_manifest: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(output_manifest)), "prepare_assets_report.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--assets-dir", type=str, required=True)
    parser.add_argument("--workspace-dir", type=str, default="")
    parser.add_argument("--report-path", type=str, default="")
    parser.add_argument("--asset-sample-rate", type=int, default=44100)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")

    parser.add_argument("--skip-separation", action="store_true")
    parser.add_argument("--separator-cmd", type=str, default="conda run -n audiosep-py310 audio-separator")
    parser.add_argument("--separator-model", type=str, default="BS-Roformer-Viperx-1297")
    parser.add_argument("--separator-model-flag", type=str, default="auto", choices=["auto", "model_filename", "model_name"])
    parser.add_argument("--separator-output-format", type=str, default="FLAC")
    parser.add_argument("--separator-model-file-dir", type=str, default="")
    parser.add_argument("--separator-numba-cache-dir", type=str, default="")
    parser.add_argument("--separator-timeout-sec", type=float, default=0.0)

    parser.add_argument("--skip-whisperx", action="store_true")
    parser.add_argument("--whisperx-cmd", type=str, default="conda run -n whisperx whisperx")
    parser.add_argument("--whisperx-model", type=str, default="large-v3")
    parser.add_argument("--whisperx-device", type=str, default="cuda")
    parser.add_argument("--whisperx-compute-type", type=str, default="float16")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--whisperx-timeout-sec", type=float, default=0.0)

    parser.add_argument("--skip-structure", action="store_true")
    parser.add_argument("--songformer-root", type=str, default="third_party/SongFormer")
    parser.add_argument("--songformer-python", type=str, default=sys.executable)
    parser.add_argument("--songformer-gpu-num", type=int, default=1)
    parser.add_argument("--songformer-threads", type=int, default=1)
    parser.add_argument("--songformer-model", type=str, default="SongFormer")
    parser.add_argument("--songformer-checkpoint", type=str, default="SongFormer.safetensors")
    parser.add_argument("--songformer-config", type=str, default="SongFormer.yaml")
    parser.add_argument("--songformer-no-rule-post", action="store_true")
    parser.add_argument("--songformer-timeout-sec", type=float, default=0.0)
    args = parser.parse_args()

    args.assets_dir = ensure_dir(args.assets_dir)
    args.workspace_dir = ensure_dir(args.workspace_dir or os.path.join(args.assets_dir, "_workspace"))
    args.report_path = args.report_path or build_default_report_path(args.output_manifest)

    items = load_jsonl(args.input_jsonl)
    preflight_issues = collect_preflight_issues(items, args)
    if preflight_issues:
        error_message = "Preflight failed: " + " ".join(preflight_issues)
        write_jsonl(
            args.report_path,
            [{"id": get_item_id(item), "status": "error", "error": error_message} for item in items],
        )
        raise SystemExit(error_message)

    output_rows = []
    report_rows = []
    total_items = len(items)
    for index, item in enumerate(items, start=1):
        sample_id = get_item_id(item)
        sample_tag = f"[{index}/{total_items}] {sample_id}"
        sample_started_at = time.time()
        log_progress(f"{sample_tag} prepare_assets start")
        record, report = prepare_item(item, args)
        output_rows.append(record)
        report_rows.append(report)
        write_jsonl(args.output_manifest, output_rows)
        write_jsonl(args.report_path, report_rows)
        log_progress(
            f"{sample_tag} prepare_assets {report['status']} in {format_seconds(time.time() - sample_started_at)}"
        )

    if not args.skip_structure and output_rows:
        log_progress("[prepare_assets] structure batch start")
        structure_items, structure_reports = prepare_structure_assets(
            items=output_rows,
            assets_dir=args.assets_dir,
            workspace_dir=args.workspace_dir,
            songformer_root=args.songformer_root,
            python_exec=args.songformer_python,
            gpu_num=args.songformer_gpu_num,
            num_thread_per_gpu=args.songformer_threads,
            model=args.songformer_model,
            checkpoint=args.songformer_checkpoint,
            config_path=args.songformer_config,
            no_rule_post_processing=args.songformer_no_rule_post,
            skip_existing=args.skip_existing,
            timeout_sec=args.songformer_timeout_sec,
        )
        merge_structure_preparation_results(
            output_rows=output_rows,
            report_rows=report_rows,
            structure_items=structure_items,
            structure_reports=structure_reports,
        )
        write_jsonl(args.output_manifest, output_rows)
        write_jsonl(args.report_path, report_rows)
        log_progress("[prepare_assets] structure batch finished")


if __name__ == "__main__":
    main()
