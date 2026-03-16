"""WhisperX 歌词对齐与清洗。

支持两种输入：
1. manifest 中直接提供 `whisperx_json`
2. 通过本地 WhisperX CLI 对音频执行转写/对齐

输出每首歌一个 `lyrics_alignment.json`，供后续结构修正与数据集构建使用。
"""

from __future__ import annotations

import argparse
import os
import shlex
import statistics
import typing as tp
from difflib import SequenceMatcher

from normalize_lyrics import clean_lyrics

from .common import (
    describe_subprocess_failure,
    ensure_dir,
    get_audio_path,
    get_item_id,
    get_raw_lyrics,
    load_json,
    load_jsonl,
    normalize_alignment_text,
    normalize_whitespace,
    save_json,
    strip_structure_tags,
    write_jsonl,
)


def parse_whisperx_segments(payload: dict) -> tp.List[dict]:
    results = []
    for segment in payload.get("segments", []):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = normalize_whitespace(segment.get("text", ""))
        words = []
        confidences = []
        for word in segment.get("words", []) or segment.get("word_segments", []) or []:
            if "start" not in word or "end" not in word:
                continue
            confidence = word.get("score", word.get("probability", word.get("confidence")))
            if confidence is not None:
                confidences.append(float(confidence))
            words.append(
                {
                    "word": normalize_whitespace(word.get("word", word.get("text", ""))),
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                    "confidence": None if confidence is None else float(confidence),
                }
            )
        results.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "words": words,
                "confidence": statistics.mean(confidences) if confidences else None,
            }
        )
    return [segment for segment in results if segment["end"] > segment["start"]]


def run_whisperx_cli(
    audio_path: str,
    output_dir: str,
    whisperx_cmd: str,
    language: tp.Optional[str],
    model: str,
    device: str,
    compute_type: str,
    timeout_sec: tp.Optional[float] = None,
) -> str:
    executable = shlex.split(whisperx_cmd)
    if not executable:
        raise ValueError("`whisperx_cmd` is empty.")
    argv = list(executable) + [
        audio_path,
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
    ]
    if language:
        argv += ["--language", language]
    try:
        completed = __import__("subprocess").run(
            argv,
            check=True,
            text=True,
            stdout=__import__("subprocess").PIPE,
            stderr=__import__("subprocess").PIPE,
            timeout=None if not timeout_sec or timeout_sec <= 0 else timeout_sec,
        )
    except (__import__("subprocess").CalledProcessError, __import__("subprocess").TimeoutExpired) as exc:
        raise RuntimeError(f"WhisperX failed: {describe_subprocess_failure(exc)}") from exc
    if completed.stderr:
        print(completed.stderr.strip())
    output_name = os.path.splitext(os.path.basename(audio_path))[0] + ".json"
    return os.path.join(output_dir, output_name)


def choose_lyrics_text(raw_lyrics: str, transcript_text: str, similarity_threshold: float) -> tp.Tuple[str, float]:
    raw_normalized = normalize_alignment_text(raw_lyrics)
    transcript_normalized = normalize_alignment_text(transcript_text)
    if raw_normalized and transcript_normalized:
        similarity = SequenceMatcher(None, raw_normalized, transcript_normalized).ratio()
    elif transcript_normalized:
        similarity = 1.0
    else:
        similarity = 0.0
    if raw_lyrics and (similarity >= similarity_threshold or not transcript_text):
        return strip_structure_tags(clean_lyrics(raw_lyrics)), similarity
    return normalize_whitespace(transcript_text), similarity


def build_alignment_record(
    item: dict,
    whisperx_payload: dict,
    similarity_threshold: float,
) -> dict:
    sample_id = get_item_id(item)
    raw_lyrics = get_raw_lyrics(item)
    segments = parse_whisperx_segments(whisperx_payload)
    transcript_text = normalize_whitespace(" ".join(segment["text"] for segment in segments))
    cleaned_lyrics, similarity = choose_lyrics_text(raw_lyrics, transcript_text, similarity_threshold)

    confidence_values = [seg["confidence"] for seg in segments if seg["confidence"] is not None]
    avg_confidence = statistics.mean(confidence_values) if confidence_values else None
    whisperx_quality = {
        "similarity": similarity,
        "avg_confidence": avg_confidence,
        "num_segments": len(segments),
        "has_word_timestamps": any(segment["words"] for segment in segments),
        "score": (similarity * 0.7) + ((avg_confidence or 0.0) * 0.3),
    }

    vocal_activity = [{"start": seg["start"], "end": seg["end"]} for seg in segments]
    return {
        "id": sample_id,
        "audio_path": item.get("vocals_path") or item.get("audio_path") or item.get("wav_path") or item.get("song_path"),
        "raw_lyrics": raw_lyrics,
        "cleaned_lyrics": cleaned_lyrics,
        "transcript_text": transcript_text,
        "transcript_segments": segments,
        "vocal_activity": vocal_activity,
        "whisperx_quality": whisperx_quality,
    }


def process_item(
    item: dict,
    output_dir: str,
    whisperx_cmd: str,
    language: tp.Optional[str],
    model: str,
    device: str,
    compute_type: str,
    similarity_threshold: float,
    skip_existing: bool,
    timeout_sec: tp.Optional[float] = None,
) -> dict:
    sample_id = get_item_id(item)
    sample_dir = ensure_dir(os.path.join(output_dir, sample_id))
    output_path = os.path.join(sample_dir, "lyrics_alignment.json")
    if skip_existing and os.path.exists(output_path):
        return load_json(output_path)

    whisperx_json = item.get("whisperx_json")
    if whisperx_json:
        payload = load_json(str(whisperx_json))
    else:
        asr_input = item.get("vocals_path") or get_audio_path(item)
        whisperx_out_dir = ensure_dir(os.path.join(sample_dir, "whisperx"))
        payload = load_json(
            run_whisperx_cli(
                audio_path=asr_input,
                output_dir=whisperx_out_dir,
                whisperx_cmd=whisperx_cmd,
                language=item.get("language", language),
                model=model,
                device=device,
                compute_type=compute_type,
                timeout_sec=timeout_sec,
            )
        )

    record = build_alignment_record(item, payload, similarity_threshold)
    save_json(output_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--whisperx-cmd", type=str, default="whisperx")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--model", type=str, default="large-v3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute-type", type=str, default="float16")
    parser.add_argument("--similarity-threshold", type=float, default=0.25)
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    items = load_jsonl(args.input_jsonl)
    summary = []
    for item in items:
        sample_id = get_item_id(item)
        try:
            record = process_item(
                item=item,
                output_dir=args.output_dir,
                whisperx_cmd=args.whisperx_cmd,
                language=args.language,
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                similarity_threshold=args.similarity_threshold,
                skip_existing=args.skip_existing,
                timeout_sec=args.timeout_sec,
            )
            summary.append(
                {
                    "id": sample_id,
                    "status": "ok",
                    "whisperx_score": record["whisperx_quality"]["score"],
                    "output_path": os.path.join(args.output_dir, sample_id, "lyrics_alignment.json"),
                }
            )
        except Exception as exc:
            summary.append({"id": sample_id, "status": "error", "error": str(exc)})

    write_jsonl(os.path.join(args.output_dir, "lyrics_alignment_report.jsonl"), summary)


if __name__ == "__main__":
    main()
