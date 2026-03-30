"""基于 SongFormer 的结构抽取与标签映射。

功能：
1. 读取预计算的 SongFormer JSON
2. 或通过本地 `third_party/SongFormer/src/SongFormer/infer/infer.py` 调用 SongFormer
3. 将结构标签映射到 SongBloom 的 `[intro]/[verse]/[chorus]/[bridge]/[inst]/[outro]`
4. 可选地利用 WhisperX vocal 活跃区间修正 vocal / non-vocal 段
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import typing as tp

from .common import (
    clip_segments,
    describe_subprocess_failure,
    ensure_dir,
    format_seconds,
    get_audio_path,
    get_item_id,
    log_progress,
    load_json,
    load_jsonl,
    make_temp_scp,
    merge_adjacent_segments,
    normalize_whitespace,
    save_json,
    vocal_ratio_for_segment,
    write_jsonl,
)


SONGFORMER_LABEL_MAP = {
    "intro": "[intro]",
    "verse": "[verse]",
    "chorus": "[chorus]",
    "bridge": "[bridge]",
    "inst": "[inst]",
    "outro": "[outro]",
    "pre-chorus": "[verse]",
}
VOCAL_LABELS = {"[verse]", "[chorus]", "[bridge]"}
NON_VOCAL_LABELS = {"[intro]", "[inst]", "[outro]"}


def _parse_major_version(version_text: str) -> tp.Optional[int]:
    head = str(version_text).strip().split(".", 1)[0]
    return int(head) if head.isdigit() else None


def collect_songformer_runtime_issues(
    python_exec: str,
    probe_timeout_sec: float = 30.0,
) -> tp.List[str]:
    python_cmd = shlex.split(python_exec)
    if not python_cmd:
        return ["SongFormer python launcher command is empty."]

    probe_code = "\n".join(
        [
            "import json",
            "info = {}",
            "try:",
            "    import torch",
            "    info['torch_version'] = getattr(torch, '__version__', '')",
            "    info['cuda_available'] = bool(torch.cuda.is_available())",
            "    if info['cuda_available']:",
            "        capability = torch.cuda.get_device_capability(0)",
            "        info['device_capability'] = f\"sm_{capability[0]}{capability[1]}\"",
            "        info['arch_list'] = list(torch.cuda.get_arch_list())",
            "except Exception as exc:",
            "    info['torch_error'] = repr(exc)",
            "try:",
            "    import numpy as np",
            "    info['numpy_version'] = getattr(np, '__version__', '')",
            "except Exception as exc:",
            "    info['numpy_error'] = repr(exc)",
            "print(json.dumps(info, ensure_ascii=False))",
        ]
    )

    try:
        completed = subprocess.run(
            python_cmd + ["-c", probe_code],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=probe_timeout_sec,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        details = describe_subprocess_failure(exc)
        if details:
            return [f"SongFormer runtime probe failed: {details}"]
        return ["SongFormer runtime probe failed."]

    stdout_lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    if not stdout_lines:
        return []

    try:
        info = json.loads(stdout_lines[-1])
    except json.JSONDecodeError:
        return []

    issues: tp.List[str] = []

    torch_error = info.get("torch_error")
    if torch_error:
        issues.append(f"SongFormer runtime probe failed to import torch: {torch_error}")

    numpy_error = info.get("numpy_error")
    if numpy_error:
        issues.append(f"SongFormer runtime probe failed to import NumPy: {numpy_error}")

    numpy_version = str(info.get("numpy_version") or "")
    numpy_major = _parse_major_version(numpy_version)
    if numpy_major is not None and numpy_major >= 2:
        issues.append(
            f"SongFormer environment has NumPy {numpy_version}, "
            "but SongFormer's jams/msaf stack requires NumPy < 2."
        )

    cuda_available = info.get("cuda_available")
    if cuda_available is False:
        issues.append(
            "SongFormer environment reports torch.cuda.is_available()=False, "
            "but SongFormer inference requires CUDA."
        )

    device_capability = str(info.get("device_capability") or "")
    arch_list = [str(item) for item in (info.get("arch_list") or []) if item]
    torch_version = str(info.get("torch_version") or "unknown")
    if device_capability and arch_list and device_capability not in arch_list:
        issues.append(
            f"SongFormer torch {torch_version} does not support GPU capability {device_capability}. "
            f"Supported capabilities: {', '.join(arch_list)}. "
            "Upgrade torch in the SongFormer environment."
        )

    return issues


def map_songformer_label(label: str, ignore_silence: bool = True) -> tp.Optional[str]:
    normalized = str(label).strip().lower()
    if normalized == "silence":
        return None if ignore_silence else "[silence]"
    return SONGFORMER_LABEL_MAP.get(normalized)


def parse_songformer_segments(payload: tp.Any, ignore_silence: bool = True) -> tp.List[dict]:
    if isinstance(payload, dict) and "segments" in payload:
        payload = payload["segments"]
    if not isinstance(payload, list):
        raise ValueError("SongFormer output must be a list of segments or {'segments': [...]} format.")

    results = []
    for segment in payload:
        mapped = map_songformer_label(segment.get("label", ""), ignore_silence=ignore_silence)
        if mapped is None:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if end <= start:
            continue
        results.append({"label": mapped, "start": start, "end": end})
    return results


def load_vocal_intervals(path: tp.Optional[str]) -> tp.List[tp.Tuple[float, float]]:
    if not path:
        return []
    payload = load_json(path)
    intervals = []
    for segment in payload.get("vocal_activity", []) or payload.get("transcript_segments", []):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if end > start:
            intervals.append((start, end))
    return intervals


def build_segments_from_vocal_intervals(
    vocal_intervals: tp.Sequence[tp.Tuple[float, float]],
    duration_hint: tp.Optional[float] = None,
    merge_gap: float = 0.5,
) -> tp.List[dict]:
    normalized: tp.List[tp.Tuple[float, float]] = []
    for start, end in sorted(vocal_intervals):
        start = max(0.0, float(start))
        end = max(start, float(end))
        if end <= start:
            continue
        if normalized and start <= normalized[-1][1] + merge_gap:
            normalized[-1] = (normalized[-1][0], max(normalized[-1][1], end))
        else:
            normalized.append((start, end))

    total_end = float(duration_hint) if duration_hint is not None else 0.0
    if normalized:
        total_end = max(total_end, normalized[-1][1])
    if total_end <= 0:
        return []

    segments: tp.List[dict] = []
    cursor = 0.0
    for start, end in normalized:
        if start > cursor + 1e-4:
            gap_label = "[intro]" if not segments else "[inst]"
            segments.append({"label": gap_label, "start": cursor, "end": start})
        segments.append({"label": "[verse]", "start": start, "end": end})
        cursor = end

    if total_end > cursor + 1e-4:
        tail_label = "[outro]" if segments else "[inst]"
        segments.append({"label": tail_label, "start": cursor, "end": total_end})

    if not segments:
        segments.append({"label": "[inst]", "start": 0.0, "end": total_end})
    return merge_adjacent_segments(segments)


def refine_with_vocal_activity(
    segments: tp.Sequence[dict],
    vocal_intervals: tp.Sequence[tp.Tuple[float, float]],
    vocal_threshold: float = 0.3,
    non_vocal_threshold: float = 0.05,
) -> tp.List[dict]:
    refined = []
    for segment in segments:
        label = segment["label"]
        start = float(segment["start"])
        end = float(segment["end"])
        ratio = vocal_ratio_for_segment(start, end, vocal_intervals)

        if label in NON_VOCAL_LABELS and ratio >= vocal_threshold:
            label = "[verse]"
        elif label in VOCAL_LABELS and ratio <= non_vocal_threshold:
            if label == "[verse]":
                label = "[inst]"
        refined.append({"label": label, "start": start, "end": end, "vocal_ratio": ratio})
    return refined


def absorb_tiny_segments(segments: tp.Sequence[dict], min_duration: float) -> tp.List[dict]:
    if not segments:
        return []
    segments = [dict(segment) for segment in segments]
    index = 0
    while index < len(segments):
        duration = segments[index]["end"] - segments[index]["start"]
        if duration >= min_duration or len(segments) == 1:
            index += 1
            continue

        if index == 0:
            segments[1]["start"] = segments[0]["start"]
            del segments[0]
            continue
        if index == len(segments) - 1:
            segments[-2]["end"] = segments[-1]["end"]
            del segments[-1]
            break

        left_gap = segments[index]["start"] - segments[index - 1]["start"]
        right_gap = segments[index + 1]["end"] - segments[index]["end"]
        if left_gap >= right_gap:
            segments[index - 1]["end"] = segments[index]["end"]
        else:
            segments[index + 1]["start"] = segments[index]["start"]
        del segments[index]
    return segments


def _list_songformer_json_candidates(output_dir: str) -> tp.List[str]:
    if not os.path.isdir(output_dir):
        return []
    candidates = []
    for name in os.listdir(output_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            candidates.append(path)
    return sorted(candidates, key=os.path.getmtime, reverse=True)


def _resolve_songformer_result(
    output_dir: str,
    expected_path: str,
    command_started_at: float,
    wait_sec: float = 2.0,
) -> tp.Tuple[tp.Optional[str], tp.List[str]]:
    deadline = time.time() + max(wait_sec, 0.0)
    candidates: tp.List[str] = []
    while True:
        if os.path.exists(expected_path):
            candidates = _list_songformer_json_candidates(output_dir)
            return expected_path, candidates

        candidates = _list_songformer_json_candidates(output_dir)
        fresh_candidates = [path for path in candidates if os.path.getmtime(path) >= command_started_at - 1.0]
        if len(fresh_candidates) == 1:
            return fresh_candidates[0], candidates
        if len(candidates) == 1:
            return candidates[0], candidates

        if time.time() >= deadline:
            return None, candidates
        time.sleep(0.2)


def _truncate_subprocess_log(text: str, max_len: int = 600) -> str:
    normalized = normalize_whitespace(text)
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3] + "..."


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


def build_songformer_batch_inputs(
    items: tp.Sequence[dict],
    input_dir: str,
) -> tp.Dict[str, str]:
    audio_inputs: tp.Dict[str, str] = {}
    ensure_dir(input_dir)
    for item in items:
        sample_id = get_item_id(item)
        source_audio = item.get("songformer_audio_path") or get_audio_path(item)
        suffix = os.path.splitext(str(source_audio))[1] or ".wav"
        batch_audio_path = os.path.join(input_dir, sample_id + suffix.lower())
        audio_inputs[sample_id] = _link_or_copy_audio(str(source_audio), batch_audio_path)
    return audio_inputs


def run_songformer_batch(
    audio_inputs: tp.Mapping[str, str],
    output_dir: str,
    songformer_root: str,
    python_exec: str,
    gpu_num: int,
    num_thread_per_gpu: int,
    model: str,
    checkpoint: str,
    config_path: str,
    no_rule_post_processing: bool,
    timeout_sec: tp.Optional[float] = None,
) -> tp.Dict[str, str]:
    if not audio_inputs:
        return {}

    infer_dir = os.path.join(songformer_root, "src", "SongFormer")
    if not os.path.exists(os.path.join(infer_dir, "infer", "infer.py")):
        raise FileNotFoundError(f"SongFormer infer script not found under {infer_dir}")

    python_cmd = shlex.split(python_exec)
    if not python_cmd:
        raise ValueError("`python_exec` is empty.")

    ensure_dir(output_dir)
    scp_tmp_dir = tempfile.mkdtemp(prefix="songformer_batch_scp_")
    scp_path = os.path.join(scp_tmp_dir, "input.scp")
    try:
        with open(scp_path, "w", encoding="utf-8") as handle:
            for audio_path in audio_inputs.values():
                handle.write(audio_path + "\n")

        env = os.environ.copy()
        infer_dir_abs = os.path.abspath(infer_dir)
        third_party_dir = os.path.abspath(os.path.normpath(os.path.join(infer_dir, "..", "third_party")))
        pythonpath_entries = [infer_dir_abs, third_party_dir]
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MPI_NUM_THREADS", "1")
        env.setdefault("NCCL_P2P_DISABLE", "1")
        env.setdefault("NCCL_IB_DISABLE", "1")

        argv = python_cmd + [
            os.path.join("infer", "infer.py"),
            "-i",
            scp_path,
            "-o",
            output_dir,
            "-gn",
            str(gpu_num),
            "-tn",
            str(num_thread_per_gpu),
            "--model",
            model,
            "--checkpoint",
            checkpoint,
            "--config_path",
            config_path,
        ]
        if no_rule_post_processing:
            argv.append("--no_rule_post_processing")

        started_at = time.time()
        log_progress(f"[songformer] batch start count={len(audio_inputs)}")
        completed = subprocess.run(
            argv,
            cwd=infer_dir,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=None if not timeout_sec or timeout_sec <= 0 else timeout_sec,
        )
        log_progress(
            f"[songformer] batch done count={len(audio_inputs)} in {format_seconds(time.time() - started_at)}"
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        details = describe_subprocess_failure(exc)
        if details:
            raise RuntimeError(f"SongFormer batch inference failed: {details}") from exc
        raise
    finally:
        shutil.rmtree(scp_tmp_dir, ignore_errors=True)

    resolved: tp.Dict[str, str] = {}
    missing: tp.List[str] = []
    for sample_id, audio_path in audio_inputs.items():
        expected_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}.json")
        if os.path.exists(expected_path):
            resolved[sample_id] = os.path.abspath(expected_path)
        else:
            missing.append(f"{sample_id}:{expected_path}")

    if not resolved:
        error_parts = ["SongFormer batch outputs not found"]
        if missing:
            error_parts.append(", ".join(missing[:10]))
        stderr_text = _truncate_subprocess_log(completed.stderr or "")
        stdout_text = _truncate_subprocess_log(completed.stdout or "")
        if stderr_text:
            error_parts.append(f"stderr={stderr_text}")
        elif stdout_text:
            error_parts.append(f"stdout={stdout_text}")
        raise FileNotFoundError(". ".join(error_parts))

    if missing:
        log_progress(
            "[songformer] batch missing outputs for "
            + ", ".join(item.split(":", 1)[0] for item in missing[:10])
        )
    return resolved


def prepare_structure_assets(
    items: tp.Sequence[dict],
    assets_dir: str,
    workspace_dir: str,
    songformer_root: str = "third_party/SongFormer",
    python_exec: str = sys.executable,
    gpu_num: int = 1,
    num_thread_per_gpu: int = 1,
    model: str = "SongFormer",
    checkpoint: str = "SongFormer.safetensors",
    config_path: str = "SongFormer.yaml",
    no_rule_post_processing: bool = False,
    skip_existing: bool = False,
    timeout_sec: tp.Optional[float] = None,
) -> tp.Tuple[tp.List[dict], tp.List[dict]]:
    updated_items = [dict(item) for item in items]
    report_rows: tp.List[dict] = []
    report_by_id: tp.Dict[str, dict] = {}
    pending_items: tp.List[dict] = []

    for item in updated_items:
        sample_id = get_item_id(item)
        report = {"id": sample_id}
        report_by_id[sample_id] = report
        report_rows.append(report)

        provided_path = item.get("structure_json")
        if provided_path and os.path.exists(str(provided_path)):
            item["structure_json"] = os.path.abspath(str(provided_path))
            report["structure_status"] = "provided"
            report["structure_json"] = item["structure_json"]
            continue

        cached_path = os.path.abspath(os.path.join(assets_dir, sample_id, "structure", sample_id + ".json"))
        if skip_existing and os.path.exists(cached_path):
            item["structure_json"] = cached_path
            report["structure_status"] = "skipped_existing"
            report["structure_json"] = cached_path
            continue

        pending_items.append(item)

    if not pending_items:
        return updated_items, report_rows

    batch_root = ensure_dir(os.path.join(workspace_dir, "_structure_batch"))
    batch_dir = tempfile.mkdtemp(prefix="songformer_batch_", dir=batch_root)
    try:
        batch_input_dir = os.path.join(batch_dir, "inputs")
        batch_output_dir = os.path.join(batch_dir, "outputs")
        audio_inputs = build_songformer_batch_inputs(pending_items, batch_input_dir)
        batch_outputs = run_songformer_batch(
            audio_inputs=audio_inputs,
            output_dir=batch_output_dir,
            songformer_root=songformer_root,
            python_exec=python_exec,
            gpu_num=gpu_num,
            num_thread_per_gpu=num_thread_per_gpu,
            model=model,
            checkpoint=checkpoint,
            config_path=config_path,
            no_rule_post_processing=no_rule_post_processing,
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        error_message = str(exc)
        for item in pending_items:
            report = report_by_id[get_item_id(item)]
            report["structure_status"] = "error"
            report["error"] = error_message
        return updated_items, report_rows

    try:
        for item in pending_items:
            sample_id = get_item_id(item)
            report = report_by_id[sample_id]
            batch_output_path = batch_outputs.get(sample_id)
            if not batch_output_path or not os.path.exists(batch_output_path):
                report["structure_status"] = "error"
                report["error"] = f"SongFormer batch output not found for {sample_id}"
                continue

            structure_dir = ensure_dir(os.path.join(assets_dir, sample_id, "structure"))
            structure_json = os.path.abspath(os.path.join(structure_dir, sample_id + ".json"))
            shutil.copyfile(batch_output_path, structure_json)
            item["structure_json"] = structure_json
            report["structure_status"] = "ok"
            report["structure_json"] = structure_json
    finally:
        shutil.rmtree(batch_dir, ignore_errors=True)

    return updated_items, report_rows


def run_songformer(
    audio_path: str,
    output_dir: str,
    songformer_root: str,
    python_exec: str,
    gpu_num: int,
    num_thread_per_gpu: int,
    model: str,
    checkpoint: str,
    config_path: str,
    no_rule_post_processing: bool,
    timeout_sec: tp.Optional[float] = None,
) -> str:
    infer_dir = os.path.join(songformer_root, "src", "SongFormer")
    if not os.path.exists(os.path.join(infer_dir, "infer", "infer.py")):
        raise FileNotFoundError(f"SongFormer infer script not found under {infer_dir}")

    temp_dir, scp_path = make_temp_scp(audio_path)
    try:
        python_cmd = shlex.split(python_exec)
        if not python_cmd:
            raise ValueError("`python_exec` is empty.")
        env = os.environ.copy()
        infer_dir_abs = os.path.abspath(infer_dir)
        third_party_dir = os.path.abspath(os.path.normpath(os.path.join(infer_dir, "..", "third_party")))
        pythonpath_entries = [infer_dir_abs, third_party_dir]
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MPI_NUM_THREADS", "1")
        env.setdefault("NCCL_P2P_DISABLE", "1")
        env.setdefault("NCCL_IB_DISABLE", "1")
        argv = python_cmd + [
            os.path.join("infer", "infer.py"),
            "-i",
            scp_path,
            "-o",
            output_dir,
            "-gn",
            str(gpu_num),
            "-tn",
            str(num_thread_per_gpu),
            "--model",
            model,
            "--checkpoint",
            checkpoint,
            "--config_path",
            config_path,
        ]
        if no_rule_post_processing:
            argv.append("--no_rule_post_processing")
        started_at = time.time()
        log_progress(f"[songformer] start {os.path.basename(audio_path)}")
        completed: tp.Optional[subprocess.CompletedProcess[str]] = None
        try:
            completed = subprocess.run(
                argv,
                cwd=infer_dir,
                env=env,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=None if not timeout_sec or timeout_sec <= 0 else timeout_sec,
            )
            log_progress(
                f"[songformer] done {os.path.basename(audio_path)} in {format_seconds(time.time() - started_at)}"
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            details = describe_subprocess_failure(exc)
            if details:
                raise RuntimeError(f"SongFormer inference failed: {details}") from exc
            raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    result_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}.json")
    resolved_path, candidates = _resolve_songformer_result(
        output_dir=output_dir,
        expected_path=result_path,
        command_started_at=started_at,
    )
    if resolved_path:
        if resolved_path != result_path:
            log_progress(
                f"[songformer] expected {os.path.basename(result_path)} missing; "
                f"fallback to {os.path.basename(resolved_path)}"
            )
        return resolved_path

    error_parts = [f"SongFormer output not found: {result_path}"]
    if candidates:
        error_parts.append(
            "json_candidates=" + ", ".join(os.path.basename(path) for path in candidates[:5])
        )
    if completed is not None:
        stderr_text = _truncate_subprocess_log(completed.stderr or "")
        stdout_text = _truncate_subprocess_log(completed.stdout or "")
        if stderr_text:
            error_parts.append(f"stderr={stderr_text}")
        elif stdout_text:
            error_parts.append(f"stdout={stdout_text}")
    raise FileNotFoundError(". ".join(error_parts))


def process_item(
    item: dict,
    output_dir: str,
    ignore_silence: bool = True,
    min_duration: float = 1.0,
    refine_vocals: bool = True,
    vocal_threshold: float = 0.3,
    non_vocal_threshold: float = 0.05,
    songformer_root: str = "third_party/SongFormer",
    python_exec: str = sys.executable,
    gpu_num: int = 1,
    num_thread_per_gpu: int = 1,
    model: str = "SongFormer",
    checkpoint: str = "SongFormer.safetensors",
    config_path: str = "SongFormer.yaml",
    no_rule_post_processing: bool = False,
    skip_existing: bool = False,
    timeout_sec: tp.Optional[float] = None,
) -> dict:
    sample_id = get_item_id(item)
    sample_dir = ensure_dir(os.path.join(output_dir, sample_id))
    output_path = os.path.join(sample_dir, "structure_segments.json")
    if skip_existing and os.path.exists(output_path):
        return load_json(output_path)

    vocal_intervals = load_vocal_intervals(item.get("lyrics_alignment_json"))
    songformer_error: tp.Optional[str] = None
    if item.get("structure_json"):
        songformer_json = str(item["structure_json"])
        segments = parse_songformer_segments(load_json(songformer_json), ignore_silence=ignore_silence)
    else:
        songformer_out_dir = ensure_dir(os.path.join(sample_dir, "songformer"))
        try:
            songformer_json = run_songformer(
                audio_path=item.get("songformer_audio_path") or get_audio_path(item),
                output_dir=songformer_out_dir,
                songformer_root=songformer_root,
                python_exec=python_exec,
                gpu_num=gpu_num,
                num_thread_per_gpu=num_thread_per_gpu,
                model=model,
                checkpoint=checkpoint,
                config_path=config_path,
                no_rule_post_processing=no_rule_post_processing,
                timeout_sec=timeout_sec,
            )
            segments = parse_songformer_segments(load_json(songformer_json), ignore_silence=ignore_silence)
        except FileNotFoundError as exc:
            songformer_json = ""
            songformer_error = str(exc)
            segments = build_segments_from_vocal_intervals(
                vocal_intervals=vocal_intervals,
                duration_hint=item.get("duration_hint"),
            )
            if segments:
                log_progress(
                    f"[songformer] fallback to vocal-activity structure for {sample_id}: "
                    f"{len(segments)} segments"
                )
            else:
                raise

    if refine_vocals:
        if vocal_intervals:
            segments = refine_with_vocal_activity(
                segments=segments,
                vocal_intervals=vocal_intervals,
                vocal_threshold=vocal_threshold,
                non_vocal_threshold=non_vocal_threshold,
            )

    segments = merge_adjacent_segments(segments)
    segments = absorb_tiny_segments(segments, min_duration=min_duration)
    segments = merge_adjacent_segments(segments)

    max_end = max((float(seg["end"]) for seg in segments), default=0.0)
    if item.get("duration_hint") is not None:
        max_end = min(max_end, float(item["duration_hint"]))
    segments = clip_segments(segments, min_start=0.0, max_end=max_end or None)

    payload = {
        "id": sample_id,
        "segments": segments,
        "source": "songformer" if songformer_json else "vocal_activity_fallback",
        "songformer_json": songformer_json,
    }
    if songformer_error:
        payload["songformer_error"] = songformer_error
    save_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ignore-silence", action="store_true")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--disable-vocal-refine", action="store_true")
    parser.add_argument("--vocal-threshold", type=float, default=0.3)
    parser.add_argument("--non-vocal-threshold", type=float, default=0.05)
    parser.add_argument("--songformer-root", type=str, default="third_party/SongFormer")
    parser.add_argument("--python-exec", type=str, default=sys.executable)
    parser.add_argument("--gpu-num", type=int, default=1)
    parser.add_argument("--num-thread-per-gpu", type=int, default=1)
    parser.add_argument("--model", type=str, default="SongFormer")
    parser.add_argument("--checkpoint", type=str, default="SongFormer.safetensors")
    parser.add_argument("--config-path", type=str, default="SongFormer.yaml")
    parser.add_argument("--no-rule-post-processing", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-jsonl", type=str, default="")
    args = parser.parse_args()

    rows = []
    for item in load_jsonl(args.input_jsonl):
        sample_id = get_item_id(item)
        try:
            payload = process_item(
                item=item,
                output_dir=args.output_dir,
                ignore_silence=args.ignore_silence,
                min_duration=args.min_duration,
                refine_vocals=not args.disable_vocal_refine,
                vocal_threshold=args.vocal_threshold,
                non_vocal_threshold=args.non_vocal_threshold,
                songformer_root=args.songformer_root,
                python_exec=args.python_exec,
                gpu_num=args.gpu_num,
                num_thread_per_gpu=args.num_thread_per_gpu,
                model=args.model,
                checkpoint=args.checkpoint,
                config_path=args.config_path,
                no_rule_post_processing=args.no_rule_post_processing,
                timeout_sec=args.timeout_sec,
                skip_existing=args.skip_existing,
            )
            rows.append(
                {
                    "id": sample_id,
                    "status": "ok",
                    "output_path": os.path.join(args.output_dir, sample_id, "structure_segments.json"),
                    "num_segments": len(payload["segments"]),
                }
            )
        except Exception as exc:
            rows.append({"id": sample_id, "status": "error", "error": str(exc)})

    if args.summary_jsonl:
        write_jsonl(args.summary_jsonl, rows)
    else:
        write_jsonl(os.path.join(args.output_dir, "structure_report.jsonl"), rows)


if __name__ == "__main__":
    main()
