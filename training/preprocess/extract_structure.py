"""基于 SongFormer 的结构抽取与标签映射。

功能：
1. 读取预计算的 SongFormer JSON
2. 或通过本地 `third_party/SongFormer/src/SongFormer/infer/infer.py` 调用 SongFormer
3. 将结构标签映射到 SongBloom 的 `[intro]/[verse]/[chorus]/[bridge]/[inst]/[outro]`
4. 可选地利用 WhisperX vocal 活跃区间修正 vocal / non-vocal 段
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
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
        try:
            subprocess.run(
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
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"SongFormer output not found: {result_path}")
    return result_path


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

    if item.get("structure_json"):
        songformer_json = str(item["structure_json"])
    else:
        songformer_out_dir = ensure_dir(os.path.join(sample_dir, "songformer"))
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
    if refine_vocals:
        vocal_intervals = load_vocal_intervals(item.get("lyrics_alignment_json"))
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
        "source": "songformer",
        "songformer_json": songformer_json,
    }
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
