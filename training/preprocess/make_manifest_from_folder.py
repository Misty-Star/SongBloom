"""从歌曲目录批量生成 SongBloom 训练预处理 manifest。

默认递归扫描源目录下所有 `.flac` 文件，并查找同名 `.lrc` 文件。
将结果追加写入目标 manifest（不存在则新建）。

最小输出字段：
    {
      "id": "...",
      "audio_path": "/abs/path/to/song.flac",
      "lyrics_raw": "原始歌词"
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import typing as tp
from pathlib import Path


def slugify(text: str, max_len: int = 48) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "song"
    return text[:max_len]


def build_sample_id(audio_path: str) -> str:
    basename = Path(audio_path).stem
    digest = hashlib.sha1(os.path.abspath(audio_path).encode("utf-8")).hexdigest()[:16]
    return f"{slugify(basename)}_{digest}"


def read_text_with_fallback(path: str) -> str:
    encodings = ["utf-8-sig", "utf-8", "utf-16", "gb18030", "big5", "latin-1"]
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as handle:
                return handle.read().replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return handle.read().replace("\r\n", "\n").replace("\r", "\n")


def load_existing_entries(manifest_path: str) -> tp.Tuple[set[str], set[str]]:
    existing_ids: set[str] = set()
    existing_audio_paths: set[str] = set()
    if not os.path.exists(manifest_path):
        return existing_ids, existing_audio_paths

    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("id")
            audio_path = row.get("audio_path")
            if sample_id:
                existing_ids.add(str(sample_id))
            if audio_path:
                existing_audio_paths.add(os.path.abspath(str(audio_path)))
    return existing_ids, existing_audio_paths


def iter_audio_files(source_dir: str, recursive: bool = True) -> tp.Iterable[str]:
    root = Path(source_dir)
    pattern = "**/*.flac" if recursive else "*.flac"
    for path in sorted(root.glob(pattern)):
        if path.is_file():
            yield str(path.resolve())


def make_record(audio_path: str, lrc_path: str) -> dict:
    return {
        "id": build_sample_id(audio_path),
        "audio_path": os.path.abspath(audio_path),
        "lyrics_raw": read_text_with_fallback(lrc_path),
    }


def append_records(manifest_path: str, records: tp.Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, required=True, help="源目录，包含 .flac 和同名 .lrc")
    parser.add_argument("--output-dir", type=str, required=True, help="manifest 输出目录")
    parser.add_argument("--manifest-name", type=str, default="raw_manifest.jsonl", help="manifest 文件名")
    parser.add_argument("--non-recursive", action="store_true", help="仅扫描源目录顶层")
    parser.add_argument("--strict", action="store_true", help="遇到缺失 .lrc 时直接报错")
    args = parser.parse_args()

    manifest_path = os.path.join(args.output_dir, args.manifest_name)
    existing_ids, existing_audio_paths = load_existing_entries(manifest_path)

    to_append: list[dict] = []
    skipped_missing_lrc = 0
    skipped_existing = 0

    for audio_path in iter_audio_files(args.source_dir, recursive=not args.non_recursive):
        lrc_path = os.path.splitext(audio_path)[0] + ".lrc"
        if not os.path.exists(lrc_path):
            if args.strict:
                raise FileNotFoundError(f"Missing paired LRC for {audio_path}")
            skipped_missing_lrc += 1
            continue

        record = make_record(audio_path, lrc_path)
        sample_id = record["id"]
        normalized_audio_path = record["audio_path"]

        if normalized_audio_path in existing_audio_paths or sample_id in existing_ids:
            skipped_existing += 1
            continue

        to_append.append(record)
        existing_ids.add(sample_id)
        existing_audio_paths.add(normalized_audio_path)

    append_records(manifest_path, to_append)

    print(json.dumps(
        {
            "manifest_path": manifest_path,
            "appended": len(to_append),
            "skipped_existing": skipped_existing,
            "skipped_missing_lrc": skipped_missing_lrc,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
