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

如果你的原始数据是“同一目录下成对出现的 `.flac` 和同名 `.lrc`”，可以先自动生成 manifest：

```bash
python -m training.preprocess.make_manifest_from_folder \
  --source-dir /path/to/song_folder \
  --output-dir /path/to/manifests
```

默认会：

- 递归扫描 `source-dir` 下所有 `.flac`
- 查找同名 `.lrc`
- 生成 `output-dir/raw_manifest.jsonl`
- 如果 manifest 已存在，则追加新样本并按 `audio_path`/`id` 去重

生成的每行至少包含：

```json
{
  "id": "song_name_ab12cd34ef56...",
  "audio_path": "/abs/path/to/song.flac",
  "lyrics_raw": "原始 lrc 内容"
}
```

其中 `id` 默认由“歌曲文件名 slug + 绝对路径 sha1 前 16 位”组成，适合多次从不同源目录追加数据时避免冲突。

如果你还没有官方提供的 `vq_codebook.pt`，当前仓库提供了一个**明确标识为 K-Means 替代方案**的脚本：

```bash
python -m training.preprocess.fit_vq_codebook_kmeans \
  --input-jsonl /path/to/raw_manifest.jsonl \
  --output-path /path/to/vq_codebook_kmeans.pt \
  --device cuda:0
```

这个脚本会：

- 使用 MuQ 提取 embedding
- 采样一部分 frame-level embedding
- 用 MiniBatch K-Means 拟合单层 codebook
- 输出与 `extract_sketch.py` / `build_dataset.py` 兼容的 `vq_codebook.pt`

之所以命名为 `fit_vq_codebook_kmeans.py`，是为了和后续可能加入的正式 VQ / RVQ 训练方案明确区分。

## 主流程

推荐把 `audio-separator` 和 `WhisperX` 放在独立环境中，先生成派生 manifest，再交给 `build_dataset`：

```bash
python -m training.preprocess.prepare_assets \
  --input-jsonl /path/to/raw_manifest.jsonl \
  --output-manifest /path/to/ready_manifest.jsonl \
  --assets-dir /path/to/prepared_assets \
  --separator-cmd "conda run -n audiosep audio-separator" \
  --whisperx-cmd "conda run -n whisperx whisperx"
```

如果你希望单命令串联资产准备和最终数据集构建，可以使用：

```bash
python -m training.preprocess.run_preprocess_pipeline \
  --input-jsonl /path/to/raw_manifest.jsonl \
  --prepared-manifest /path/to/ready_manifest.jsonl \
  --assets-dir /path/to/prepared_assets \
  --output-dir /path/to/processed_dataset \
  --vq-ckpt /path/to/vq_codebook.pt \
  --separator-cmd "conda run -n audiosep audio-separator" \
  --whisperx-cmd "conda run -n whisperx whisperx" \
  --songformer-python "conda run -n songformer python"
```

`prepare_assets` 会：

1. 为缺失 stems 的样本调用 `audio-separator`
2. 将输出写到 `prepared_assets/{sample_id}/separator/`
3. 为缺失 `whisperx_json` 的样本调用 `WhisperX`
4. 将输出写到 `prepared_assets/{sample_id}/whisperx/`
5. 生成一个派生 manifest（例如 `ready_manifest.jsonl`）

```bash
python -m training.preprocess.build_dataset \
  --input-jsonl /path/to/ready_manifest.jsonl \
  --output-dir /path/to/processed_dataset \
  --vq-ckpt /path/to/vq_codebook.pt
```

如果 `SongFormer` 需要在单独的 `conda` 环境 `songformer` 中运行，推荐显式传入：

```bash
python -m training.preprocess.build_dataset \
  --input-jsonl /path/to/ready_manifest.jsonl \
  --output-dir /path/to/processed_dataset \
  --vq-ckpt /path/to/vq_codebook.pt \
  --songformer-python "conda run -n songformer python"
```

主流程会按顺序执行：

1. 标准化原始音频
2. 读取 manifest 中已有的 `vocals_path` / `no_vocals_path`，缺失时回退到 `Demucs`
3. 读取 manifest 中已有的 `whisperx_json`，缺失时回退到在线 `WhisperX`
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
