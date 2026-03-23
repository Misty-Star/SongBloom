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
import tempfile
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
    load_json,
    load_jsonl,
    log_progress,
    normalize_whitespace,
    save_json,
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


def resolve_python_launcher(explicit_command: str, tool_command: str, tool_name: str) -> str:
    if explicit_command:
        return explicit_command

    argv = shlex.split(tool_command)
    if not argv:
        raise ValueError(f"`{tool_name}` launcher is empty.")

    if os.path.basename(argv[0]).startswith("python"):
        return tool_command

    if os.path.basename(argv[-1]) == tool_name:
        argv[-1] = "python"
        return shlex.join(argv)

    return "python"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _extend_pythonpath(env: dict) -> dict:
    updated = dict(env)
    repo_root = _repo_root()
    existing_pythonpath = updated.get("PYTHONPATH", "")
    pythonpath_entries = [repo_root]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    updated["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return updated


def _link_or_copy_audio(src_path: str, dst_path: str) -> str:
    ensure_dir(os.path.dirname(dst_path))
    if os.path.lexists(dst_path):
        os.unlink(dst_path)
    src_abs = os.path.abspath(src_path)
    try:
        os.symlink(src_abs, dst_path)
    except OSError:
        shutil.copyfile(src_abs, dst_path)
    return dst_path


def build_batch_audio_inputs(
    items: tp.Sequence[dict],
    input_dir: str,
    audio_key: str,
) -> tp.Dict[str, str]:
    audio_inputs: tp.Dict[str, str] = {}
    ensure_dir(input_dir)
    for item in items:
        sample_id = get_item_id(item)
        source_audio = str(item[audio_key])
        suffix = os.path.splitext(source_audio)[1] or ".wav"
        batch_audio_path = os.path.join(input_dir, sample_id + suffix.lower())
        audio_inputs[sample_id] = _link_or_copy_audio(source_audio, batch_audio_path)
    return audio_inputs


def chunk_items(items: tp.Sequence[dict], max_files_per_batch: int) -> tp.Iterable[tp.Sequence[dict]]:
    if max_files_per_batch <= 0:
        raise ValueError("max_files_per_batch must be positive")
    for start in range(0, len(items), max_files_per_batch):
        yield items[start : start + max_files_per_batch]


def append_stage_error(report: dict, stage: str, error: tp.Union[str, Exception]) -> None:
    message = normalize_whitespace(error if isinstance(error, str) else describe_exception(error))
    report.setdefault("errors", []).append({"stage": stage, "error": message or f"{stage} failed"})


def update_report_status(record: dict, report: dict) -> None:
    report["status"] = compute_prepare_assets_status(record=record, report=report)


def initialize_prepare_assets_rows(items: tp.Sequence[dict]) -> tp.Tuple[tp.List[dict], tp.List[dict]]:
    output_rows: tp.List[dict] = []
    report_rows: tp.List[dict] = []
    for item in items:
        record = normalize_record_paths(item)
        report = {
            "id": get_item_id(record),
            "separator_status": "provided" if record.get("vocals_path") and record.get("no_vocals_path") else "skipped",
            "whisperx_status": "provided" if record.get("whisperx_json") else "skipped",
        }
        update_report_status(record, report)
        output_rows.append(record)
        report_rows.append(report)
    return output_rows, report_rows


def run_audio_separator_batch(batch_items: tp.Sequence[dict], args) -> tp.Tuple[tp.Dict[str, dict], tp.Dict[str, str]]:
    if not batch_items:
        return {}, {}

    batch_root = tempfile.mkdtemp(prefix="audio_separator_batch_", dir=ensure_dir(os.path.join(args.workspace_dir, "_separator_batch")))
    input_json = os.path.join(batch_root, "input.json")
    output_json = os.path.join(batch_root, "output.json")
    save_json(input_json, list(batch_items))

    python_launcher = resolve_python_launcher(args.separator_python, args.separator_cmd, "audio-separator")
    argv = shlex.split(python_launcher) + [
        "-m",
        "training.preprocess.audio_separator_batch_runner",
        "--input-json",
        input_json,
        "--output-json",
        output_json,
        "--model",
        resolve_separator_model(args.separator_model),
        "--output-format",
        args.separator_output_format,
        "--sample-rate",
        str(args.asset_sample_rate),
    ]
    if args.separator_model_file_dir:
        argv += ["--model-file-dir", args.separator_model_file_dir]

    env = _extend_pythonpath(os.environ.copy())
    env.setdefault("NUMBA_CACHE_DIR", get_separator_numba_cache_dir(args))
    ensure_dir(env["NUMBA_CACHE_DIR"])

    try:
        subprocess.run(
            argv,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=None if not args.separator_timeout_sec or args.separator_timeout_sec <= 0 else args.separator_timeout_sec,
        )
        rows = load_json(output_json)
    except Exception as exc:
        error_message = describe_exception(exc)
        shutil.rmtree(batch_root, ignore_errors=True)
        return {}, {get_item_id(item): error_message for item in batch_items}

    results: tp.Dict[str, dict] = {}
    errors: tp.Dict[str, str] = {}
    rows_by_id = {str(row["id"]): row for row in rows}
    for item in batch_items:
        sample_id = get_item_id(item)
        row = rows_by_id.get(sample_id)
        if row and row.get("status") == "ok":
            results[sample_id] = {
                "vocals_path": os.path.abspath(str(row["vocals_path"])),
                "no_vocals_path": os.path.abspath(str(row["no_vocals_path"])),
            }
        else:
            errors[sample_id] = normalize_whitespace(str((row or {}).get("error", "audio-separator batch runner returned no result")))

    shutil.rmtree(batch_root, ignore_errors=True)
    return results, errors


def prepare_separator_assets(output_rows: tp.List[dict], report_rows: tp.List[dict], args) -> None:
    if args.skip_separation:
        return

    report_by_id = {str(row["id"]): row for row in report_rows}
    pending: tp.List[dict] = []

    for record in output_rows:
        sample_id = get_item_id(record)
        report = report_by_id[sample_id]
        if record.get("vocals_path") and record.get("no_vocals_path"):
            report["separator_status"] = "provided"
            update_report_status(record, report)
            continue

        separator_dir = ensure_dir(os.path.join(args.assets_dir, sample_id, "separator"))
        if args.skip_existing:
            try:
                vocals_path, no_vocals_path = find_separator_outputs(separator_dir, args.separator_output_format)
                record["vocals_path"] = vocals_path
                record["no_vocals_path"] = no_vocals_path
                report["separator_status"] = "skipped_existing"
                update_report_status(record, report)
                continue
            except FileNotFoundError:
                pass

        workspace_sample_dir = ensure_dir(os.path.join(args.workspace_dir, sample_id))
        try:
            standardized_audio, _duration = convert_audio(
                input_path=get_audio_path(record),
                output_path=os.path.join(workspace_sample_dir, "separator_input.flac"),
                sample_rate=args.asset_sample_rate,
                mono=False,
            )
            expected_vocals, expected_no_vocals = expected_separator_outputs(separator_dir, args.separator_output_format)
            pending.append(
                {
                    "id": sample_id,
                    "audio_path": standardized_audio,
                    "vocals_output_path": os.path.abspath(expected_vocals),
                    "no_vocals_output_path": os.path.abspath(expected_no_vocals),
                }
            )
        except Exception as exc:
            report["separator_status"] = "error"
            append_stage_error(report, "separator", exc)
            update_report_status(record, report)

    if not pending:
        return

    total_batches = (len(pending) + args.separator_max_files_per_batch - 1) // args.separator_max_files_per_batch
    for batch_index, batch_items in enumerate(chunk_items(pending, args.separator_max_files_per_batch), start=1):
        log_progress(f"[prepare_assets] separator batch {batch_index}/{total_batches} size={len(batch_items)}")
        results, errors = run_audio_separator_batch(batch_items, args)
        for batch_item in batch_items:
            sample_id = get_item_id(batch_item)
            record = next(row for row in output_rows if get_item_id(row) == sample_id)
            report = report_by_id[sample_id]
            if sample_id in results:
                record["vocals_path"] = results[sample_id]["vocals_path"]
                record["no_vocals_path"] = results[sample_id]["no_vocals_path"]
                report["separator_status"] = "ok"
            else:
                report["separator_status"] = "error"
                append_stage_error(report, "separator", errors.get(sample_id, "audio-separator batch failed"))
            update_report_status(record, report)


def run_whisperx_batch_cli(
    audio_inputs: tp.Mapping[str, str],
    output_dir: str,
    whisperx_cmd: str,
    language: tp.Optional[str],
    model: str,
    device: str,
    compute_type: str,
    batch_size: int,
    timeout_sec: tp.Optional[float] = None,
) -> tp.Dict[str, str]:
    if not audio_inputs:
        return {}

    executable = shlex.split(whisperx_cmd)
    if not executable:
        raise ValueError("`whisperx_cmd` is empty.")
    argv = list(executable) + list(audio_inputs.values()) + [
        "--output_dir",
        output_dir,
        "--output_format",
        "json",
        "--model",
        model,
        "--device",
        device,
        "--compute_type",
        compute_type,
        "--batch_size",
        str(batch_size),
    ]
    if language:
        argv += ["--language", language]

    completed = subprocess.run(
        argv,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=None if not timeout_sec or timeout_sec <= 0 else timeout_sec,
    )
    if completed.stderr:
        print(completed.stderr.strip())

    resolved: tp.Dict[str, str] = {}
    missing: tp.List[str] = []
    for sample_id, audio_path in audio_inputs.items():
        expected_path = expected_whisperx_output(audio_path, output_dir)
        if os.path.exists(expected_path):
            resolved[sample_id] = expected_path
        else:
            missing.append(sample_id)
    if missing:
        raise FileNotFoundError("WhisperX batch outputs not found for " + ", ".join(missing))
    return resolved


def run_whisperx_batch_with_recovery(
    batch_items: tp.Sequence[dict],
    audio_inputs: tp.Mapping[str, str],
    batch_root: str,
    args,
    language: tp.Optional[str],
) -> tp.Tuple[tp.Dict[str, str], tp.Dict[str, str]]:
    def _run(items: tp.Sequence[dict]) -> tp.Tuple[tp.Dict[str, str], tp.Dict[str, str]]:
        if not items:
            return {}, {}

        call_output_dir = tempfile.mkdtemp(prefix="whisperx_call_", dir=batch_root)
        call_audio_inputs = {get_item_id(item): audio_inputs[get_item_id(item)] for item in items}
        try:
            outputs = run_whisperx_batch_cli(
                audio_inputs=call_audio_inputs,
                output_dir=call_output_dir,
                whisperx_cmd=args.whisperx_cmd,
                language=language,
                model=args.whisperx_model,
                device=args.whisperx_device,
                compute_type=args.whisperx_compute_type,
                batch_size=args.whisperx_batch_size,
                timeout_sec=args.whisperx_timeout_sec,
            )
            return outputs, {}
        except Exception as exc:
            shutil.rmtree(call_output_dir, ignore_errors=True)
            if len(items) == 1:
                return {}, {get_item_id(items[0]): describe_exception(exc)}

            midpoint = max(1, len(items) // 2)
            left_outputs, left_errors = _run(items[:midpoint])
            right_outputs, right_errors = _run(items[midpoint:])
            return (
                {**left_outputs, **right_outputs},
                {**left_errors, **right_errors},
            )

    return _run(batch_items)


def prepare_whisperx_assets(output_rows: tp.List[dict], report_rows: tp.List[dict], args) -> None:
    if args.skip_whisperx:
        return

    report_by_id = {str(row["id"]): row for row in report_rows}
    record_by_id = {get_item_id(row): row for row in output_rows}
    pending_by_language: tp.Dict[tp.Optional[str], tp.List[dict]] = {}

    for record in output_rows:
        sample_id = get_item_id(record)
        report = report_by_id[sample_id]
        if record.get("whisperx_json"):
            report["whisperx_status"] = "provided"
            update_report_status(record, report)
            continue

        asr_input = str(record.get("vocals_path") or get_audio_path(record))
        whisperx_dir = ensure_dir(os.path.join(args.assets_dir, sample_id, "whisperx"))
        existing_json = expected_whisperx_output(asr_input, whisperx_dir)
        if args.skip_existing and os.path.exists(existing_json):
            record["whisperx_json"] = existing_json
            report["whisperx_status"] = "skipped_existing"
            update_report_status(record, report)
            continue

        language = record.get("language", args.language)
        pending_by_language.setdefault(language, []).append(
            {
                "id": sample_id,
                "audio_path": asr_input,
                "whisperx_output_path": existing_json,
            }
        )

    batch_root_parent = ensure_dir(os.path.join(args.workspace_dir, "_whisperx_batch"))
    for language, language_items in pending_by_language.items():
        total_batches = (len(language_items) + args.whisperx_max_files_per_batch - 1) // args.whisperx_max_files_per_batch
        for batch_index, batch_items in enumerate(chunk_items(language_items, args.whisperx_max_files_per_batch), start=1):
            language_tag = language or "auto"
            log_progress(
                f"[prepare_assets] whisperx batch {batch_index}/{total_batches} language={language_tag} size={len(batch_items)}"
            )
            batch_root = tempfile.mkdtemp(prefix="whisperx_batch_", dir=batch_root_parent)
            try:
                audio_inputs = build_batch_audio_inputs(
                    items=batch_items,
                    input_dir=os.path.join(batch_root, "inputs"),
                    audio_key="audio_path",
                )
                results, errors = run_whisperx_batch_with_recovery(
                    batch_items=batch_items,
                    audio_inputs=audio_inputs,
                    batch_root=batch_root,
                    args=args,
                    language=language,
                )
                for batch_item in batch_items:
                    sample_id = get_item_id(batch_item)
                    record = record_by_id[sample_id]
                    report = report_by_id[sample_id]
                    if sample_id in results:
                        whisperx_json = os.path.abspath(batch_item["whisperx_output_path"])
                        ensure_dir(os.path.dirname(whisperx_json))
                        shutil.copyfile(results[sample_id], whisperx_json)
                        record["whisperx_json"] = whisperx_json
                        report["whisperx_status"] = "ok"
                    else:
                        report["whisperx_status"] = "error"
                        append_stage_error(report, "whisperx", errors.get(sample_id, "WhisperX batch failed"))
                    update_report_status(record, report)
            finally:
                shutil.rmtree(batch_root, ignore_errors=True)

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
        separator_exec = resolve_command_name(resolve_python_launcher(args.separator_python, args.separator_cmd, "audio-separator"))
        if not command_exists(separator_exec):
            issues.append(
                f"audio-separator python launcher {separator_exec!r} is not in PATH. "
                "Install it in a dedicated environment or pass --separator-python/--separator-cmd."
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

    output_rows, report_rows = initialize_prepare_assets_rows(items)

    separator_started_at = time.time()
    log_progress("[prepare_assets] separator stage start")
    prepare_separator_assets(output_rows, report_rows, args)
    write_jsonl(args.output_manifest, output_rows)
    write_jsonl(args.report_path, report_rows)
    log_progress(f"[prepare_assets] separator stage finished in {format_seconds(time.time() - separator_started_at)}")

    whisperx_started_at = time.time()
    log_progress("[prepare_assets] whisperx stage start")
    prepare_whisperx_assets(output_rows, report_rows, args)
    write_jsonl(args.output_manifest, output_rows)
    write_jsonl(args.report_path, report_rows)
    log_progress(f"[prepare_assets] whisperx stage finished in {format_seconds(time.time() - whisperx_started_at)}")

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

    if not args.keep_intermediate:
        for item in items:
            shutil.rmtree(os.path.join(args.workspace_dir, get_item_id(item)), ignore_errors=True)
        shutil.rmtree(os.path.join(args.workspace_dir, "_separator_batch"), ignore_errors=True)
        shutil.rmtree(os.path.join(args.workspace_dir, "_whisperx_batch"), ignore_errors=True)
        shutil.rmtree(os.path.join(args.workspace_dir, "_structure_batch"), ignore_errors=True)


if __name__ == "__main__":
    main()
