# SongBloom 训练预处理

## 输入 manifest

每行一个 JSON，至少包含：

```json
{
  "id": "song_001",
  "audio_path": "/path/to/full_song.wav",
  "lyrics_raw": "原始歌词"
}
```

可选字段：

- `whisperx_json`：预先导出的 WhisperX JSON
- `structure_json`：预先导出的 SongFormer JSON
- `vocals_path` / `no_vocals_path`：预先分离的人声/伴奏
- `language`：WhisperX 语言提示
- `prompt_start_sec` / `prompt_path`：保留给上游数据准备使用

## 主流程

```bash
python -m training.preprocess.build_dataset \
  --input-jsonl /path/to/raw_manifest.jsonl \
  --output-dir /path/to/processed_dataset \
  --vq-ckpt /path/to/vq_codebook.pt
```

如果 `SongFormer` 需要在单独的 `conda` 环境 `songformer` 中运行，推荐显式传入：

```bash
python -m training.preprocess.build_dataset \
  --input-jsonl /path/to/raw_manifest.jsonl \
  --output-dir /path/to/processed_dataset \
  --vq-ckpt /path/to/vq_codebook.pt \
  --songformer-python "conda run -n songformer python"
```

主流程会按顺序执行：

1. 标准化原始音频
2. `Demucs` 分离 `vocals` / `no_vocals`
3. `WhisperX` 对齐歌词并生成 `lyrics_alignment.json`
4. `third_party/SongFormer` 提取结构并生成 `structure_segments.json`
5. 生成 SongBloom 训练所需的结构化歌词与 `structure_duration`
6. 提取 `prompt_wav.flac`、`x_latent.pt`、`x_sketch.pt`
7. 写出最终 `meta.json`

其中歌词清洗阶段会优先过滤常见的：

- LRC 元数据标签，如 `[ar:...]`、`[ti:...]`、`[al:...]`、`[by:...]`
- 带时间戳的署名行，如 `[00:00.00]作词：...`
- 常见创作/制作署名，如 `作词/作曲/编曲/制作人/混音/母带/录音`
- 常见平台噪声文本，如“贡献歌词”“期待您的精彩评论”“该歌曲为纯音乐，请欣赏”

## 单阶段脚本

- `python -m training.preprocess.align_lyrics`
- `python -m training.preprocess.extract_structure`
- `python -m training.preprocess.extract_prompt`

如果单独运行 `extract_structure`，同样建议传入：

```bash
python -m training.preprocess.extract_structure \
  --input-jsonl /path/to/raw_manifest.jsonl \
  --output-dir /path/to/structure_out \
  --python-exec "conda run -n songformer python"
```

## 产物目录

每首歌一个子目录：

```text
sample_dir/
  lyrics.txt
  prompt_wav.flac
  x_latent.pt
  x_sketch.pt
  meta.json
```

其中 `meta.json` 至少包含：

- `duration`
- `structure_duration`
- `whisperx_quality`
- `songformer_segments`
