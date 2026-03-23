"""在单个 Python 进程内复用 audio-separator 模型处理多个音频。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import typing as tp

from audio_separator.separator.separator import Separator


def normalize_stem_name(name: str) -> str:
    normalized = str(name or "").strip().lower().replace("_", " ")
    return " ".join(normalized.split())


def resolve_output_path(output_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(output_dir, path))


def match_stem_output(
    output_paths: tp.Sequence[str],
    preferred_names: tp.Sequence[str],
) -> str:
    preferred = {normalize_stem_name(name) for name in preferred_names}
    for path in output_paths:
        base = normalize_stem_name(os.path.basename(path))
        if any(name in base for name in preferred):
            return path
    raise FileNotFoundError(f"Could not resolve stem output from {list(output_paths)} for {list(preferred_names)}")


def copy_output(src_path: str, dst_path: str) -> str:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copyfile(src_path, dst_path)
    return os.path.abspath(dst_path)


def cleanup_paths(paths: tp.Iterable[str]) -> None:
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)


def process_batch(
    items: tp.Sequence[dict],
    output_json: str,
    model: str,
    model_file_dir: str,
    output_format: str,
    sample_rate: int,
) -> None:
    batch_output_dir = os.path.join(os.path.dirname(os.path.abspath(output_json)), "separator_outputs")
    os.makedirs(batch_output_dir, exist_ok=True)

    separator = Separator(
        log_level=logging.INFO,
        model_file_dir=model_file_dir,
        output_dir=batch_output_dir,
        output_format=output_format,
        sample_rate=sample_rate,
    )
    separator.load_model(model)

    results: list[dict] = []
    for item in items:
        sample_id = str(item["id"])
        try:
            raw_outputs = separator.separate(str(item["audio_path"]))
            output_paths = [resolve_output_path(batch_output_dir, path) for path in raw_outputs]
            primary_name = normalize_stem_name(getattr(separator.model_instance, "primary_stem_name", ""))
            secondary_name = normalize_stem_name(getattr(separator.model_instance, "secondary_stem_name", ""))
            vocal_names = ["vocals"]
            instrumental_names = ["instrumental", "no vocals"]
            if primary_name == "vocals":
                vocal_names.append(primary_name)
            if secondary_name == "vocals":
                vocal_names.append(secondary_name)
            if primary_name in {"instrumental", "no vocals"}:
                instrumental_names.append(primary_name)
            if secondary_name in {"instrumental", "no vocals"}:
                instrumental_names.append(secondary_name)
            vocals_src = match_stem_output(
                output_paths,
                vocal_names,
            )
            no_vocals_src = match_stem_output(output_paths, instrumental_names)

            vocals_path = copy_output(vocals_src, str(item["vocals_output_path"]))
            no_vocals_path = copy_output(no_vocals_src, str(item["no_vocals_output_path"]))
            cleanup_paths(output_paths)
            results.append(
                {
                    "id": sample_id,
                    "status": "ok",
                    "vocals_path": vocals_path,
                    "no_vocals_path": no_vocals_path,
                }
            )
        except Exception as exc:  # pragma: no cover - exercised via parent integration
            results.append(
                {
                    "id": sample_id,
                    "status": "error",
                    "error": str(exc),
                }
            )

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-file-dir", type=str, default="/tmp/audio-separator-models")
    parser.add_argument("--output-format", type=str, default="FLAC")
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as handle:
        items = json.load(handle)
    process_batch(
        items=items,
        output_json=args.output_json,
        model=args.model,
        model_file_dir=args.model_file_dir,
        output_format=args.output_format,
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()
