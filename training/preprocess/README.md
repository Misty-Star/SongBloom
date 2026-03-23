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

当前推荐工作流是：

```text
raw_manifest.jsonl
  -> prepare_assets
  -> ready_manifest.jsonl
  -> build_dataset
```

推荐把 `audio-separator`、`WhisperX`、`SongFormer` 放在独立环境中，先生成派生 manifest，再交给 `build_dataset`：

```bash
python -m training.preprocess.prepare_assets \
  --input-jsonl /path/to/raw_manifest.jsonl \
  --output-manifest /path/to/ready_manifest.jsonl \
  --assets-dir /path/to/prepared_assets \
  --separator-cmd "conda run -n audiosep audio-separator" \
  --separator-python "conda run -n audiosep python" \
  --whisperx-cmd "conda run -n whisperx whisperx" \
  --whisperx-batch-size 8 \
  --songformer-python "conda run -n songformer python"
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
  --separator-python "conda run -n audiosep python" \
  --whisperx-cmd "conda run -n whisperx whisperx" \
  --whisperx-batch-size 8 \
  --songformer-python "conda run -n songformer python"
```

`prepare_assets` 会：

1. 为缺失 stems 的样本按批调用 `audio-separator`（通过单进程 batch runner 复用模型）
2. 将输出写到 `prepared_assets/{sample_id}/separator/`
3. 为缺失 `whisperx_json` 的样本按语言分组批量调用 `WhisperX`
4. 将输出写到 `prepared_assets/{sample_id}/whisperx/`
5. 为缺失 `structure_json` 的样本批量调用 `SongFormer`
6. 将结构输出写到 `prepared_assets/{sample_id}/structure/{sample_id}.json`
7. 生成一个派生 manifest（例如 `ready_manifest.jsonl`）
8. 写出 `prepare_assets_report.jsonl`，其中会记录 `separator_status`、`whisperx_status`、`structure_status`

补充说明：

- `audio-separator` CLI 本身只接受单个 `audio_file`，当前仓库会在 `--separator-python` 指向的环境里启动 `training.preprocess.audio_separator_batch_runner`，一次加载模型后串行处理当前批次的多首歌
- `WhisperX` CLI 原生支持 `audio [audio ...]` 多输入，当前仓库会把同语言样本放进同一批，并在整批失败时自动二分拆批，直到把报错样本隔离出来
- 对于纯音乐 / 无语音样本，WhisperX 批处理失败不会拖垮整批；最终只会把出错样本标成 `whisperx_status=error`，其他样本继续落盘

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

1. `prepare_assets` 优先补齐 `vocals_path` / `no_vocals_path`
2. `prepare_assets` 优先补齐 `whisperx_json`
3. `prepare_assets` 批量补齐 `structure_json`
4. `build_dataset` 标准化原始音频
5. `build_dataset` 根据 WhisperX + SongFormer 结果生成 SongBloom 所需的结构化歌词与 `structure_duration`
6. `build_dataset` 提取 `prompt_wav.flac`、`x_latent.pt`、`x_sketch.pt`
7. `build_dataset` 写出最终 `meta.json`

说明：

- `build_dataset` 仍然可以直接消费原始 manifest
- 如果直接传原始 manifest 且缺 `structure_json`，当前实现会在逐样本处理前先批量预取结构结果
- 但如果还缺 `vocals_path` / `no_vocals_path` 或 `whisperx_json`，这些阶段仍会落回样本循环，所以最佳吞吐依然是先跑 `prepare_assets`
- `build_dataset` 当前会复用同一份标准化 waveform 生成 `prompt_wav.flac`、`x_latent.pt`、`x_sketch.pt`，避免对同一首歌重复读盘和重复重采样

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

如果你走推荐工作流，额外还会看到：

```text
prepared_assets/
  <sample_id>/
    separator/
    whisperx/
    structure/
      <sample_id>.json
```
