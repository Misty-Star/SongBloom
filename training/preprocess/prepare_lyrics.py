"""歌词预处理：G2P 转换 + 结构标注 + meta.json 生成。

用法:
    python -m training.preprocess.prepare_lyrics \
        --input-jsonl /path/to/songs.jsonl \
        --output-dir /path/to/output \
        --lyric-processor phoneme

输入 JSONL 格式（每行一个 JSON）:
    {
        "id": "song_001",
        "lyrics": "[verse]\n想要说些什么 . 却不知从何说起\n,\n[chorus]\n...",
        "duration": 120.0,
        "structure_duration": [["[intro]", 0.0, 8.0], ["[verse]", 8.0, 30.0], ...]
    }
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from SongBloom.g2p.lyric_common import process_lyric_preserve_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--lyric-processor", type=str, default="phoneme", choices=["pinyin", "phoneme"])
    args = parser.parse_args()

    with open(args.input_jsonl, "r") as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    for item in lines:
        song_id = item["id"]
        out_dir = os.path.join(args.output_dir, song_id)
        os.makedirs(out_dir, exist_ok=True)

        # G2P 转换
        raw_lyrics = item["lyrics"]
        processed = process_lyric_preserve_labels(raw_lyrics, args.lyric_processor)

        # 写入歌词
        with open(os.path.join(out_dir, "lyrics.txt"), "w") as f:
            f.write(processed)

        # 写入 meta.json
        meta = {"duration": item["duration"]}
        if "structure_duration" in item:
            meta["structure_duration"] = item["structure_duration"]

        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
