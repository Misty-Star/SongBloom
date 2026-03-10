"""SongBloom 训练预处理通用工具。"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import typing as tp
from pathlib import Path


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
VOCAL_LABELS = {"[verse]", "[chorus]", "[bridge]"}
NON_VOCAL_LABELS = {"[intro]", "[inst]", "[outro]"}
ALL_STRUCTURE_LABELS = VOCAL_LABELS | NON_VOCAL_LABELS | {"[silence]"}


def load_json(path: str) -> tp.Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, payload: tp.Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_jsonl(path: str) -> tp.List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str, rows: tp.Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def run_command(
    argv: tp.Sequence[str],
    workdir: tp.Optional[str] = None,
    env: tp.Optional[dict] = None,
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    if verbose:
        print("[cmd]", " ".join(argv))
    return subprocess.run(
        list(argv),
        cwd=workdir,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_structure_tags(text: str) -> str:
    return normalize_whitespace(re.sub(r"\[[^\]]+\]", " ", text or ""))


def normalize_alignment_text(text: str) -> str:
    text = strip_structure_tags(text)
    text = text.lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    return normalize_whitespace(text)


def split_sentences(text: str) -> tp.List[str]:
    cleaned = strip_structure_tags(text)
    cleaned = re.sub(r"[，,]\s*", ". ", cleaned)
    units = re.split(r"[\.\!\?\n。！？；;]+", cleaned)
    results = [normalize_whitespace(unit) for unit in units]
    return [item for item in results if item]


def get_item_id(item: dict) -> str:
    sample_id = item.get("id") or item.get("idx")
    if not sample_id:
        raise KeyError("Each manifest item must contain `id` or `idx`.")
    return str(sample_id)


def get_audio_path(item: dict) -> str:
    for key in ("audio_path", "wav_path", "song_path", "path"):
        value = item.get(key)
        if value:
            return str(value)
    raise KeyError(f"Sample {get_item_id(item)} is missing `audio_path`.")


def get_raw_lyrics(item: dict) -> str:
    for key in ("lyrics_raw", "lyrics", "text"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def find_file_with_stem(directory: str, stem: str, suffixes: tp.Sequence[str]) -> tp.Optional[str]:
    base = Path(directory)
    for suffix in suffixes:
        candidate = base / f"{stem}{suffix}"
        if candidate.exists():
            return str(candidate)
    return None


def overlap_duration(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def intervals_total_length(intervals: tp.Sequence[tp.Tuple[float, float]]) -> float:
    return sum(max(0.0, end - start) for start, end in intervals)


def vocal_ratio_for_segment(
    start: float,
    end: float,
    intervals: tp.Sequence[tp.Tuple[float, float]],
) -> float:
    denom = max(end - start, 1e-6)
    vocal = sum(overlap_duration(start, end, iv_start, iv_end) for iv_start, iv_end in intervals)
    return max(0.0, min(1.0, vocal / denom))


def merge_adjacent_segments(
    segments: tp.Sequence[dict],
    min_duration: float = 0.0,
) -> tp.List[dict]:
    merged: tp.List[dict] = []
    for segment in sorted(segments, key=lambda item: (item["start"], item["end"])):
        start = float(segment["start"])
        end = float(segment["end"])
        label = str(segment["label"])
        if end <= start:
            continue
        if end - start < min_duration:
            continue
        if merged and merged[-1]["label"] == label and start <= merged[-1]["end"] + 1e-4:
            merged[-1]["end"] = max(merged[-1]["end"], end)
            continue
        merged.append({"label": label, "start": start, "end": end})
    return merged


def clip_segments(
    segments: tp.Sequence[dict],
    min_start: float = 0.0,
    max_end: tp.Optional[float] = None,
) -> tp.List[dict]:
    clipped: tp.List[dict] = []
    for segment in segments:
        start = max(float(segment["start"]), min_start)
        end = float(segment["end"])
        if max_end is not None:
            end = min(end, max_end)
        if end <= start:
            continue
        clipped.append({"label": str(segment["label"]), "start": start, "end": end})
    return clipped


def make_temp_scp(audio_path: str) -> tp.Tuple[str, str]:
    tmp_dir = tempfile.mkdtemp(prefix="songbloom_scp_")
    scp_path = os.path.join(tmp_dir, "input.scp")
    with open(scp_path, "w", encoding="utf-8") as handle:
        handle.write(audio_path + "\n")
    return tmp_dir, scp_path

