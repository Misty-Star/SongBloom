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

如果你的目标是**先验证兼容性、尽快开始构建 `x_sketch.pt`**，优先用上面的 `fit_vq_codebook_kmeans.py`。

如果你的目标是**效果优先**，当前仓库额外提供了一套更完整的候选训练与筛选流程：

```bash
# 1) 训练单个 streaming K-Means codebook
python -m training.preprocess.fit_vq_codebook_streaming \
  --input-jsonl /path/to/raw_manifest.jsonl \
  --output-path /path/to/vq_codebook_streaming.pt \
  --device cuda:0 \
  --muq-cache-dir /path/to/muq_cache \
  --max-total-train-frames 200000 \
  --max-total-heldout-frames 50000

# 2) 在 heldout 帧上单独评估现有 codebook
python -m training.preprocess.evaluate_vq_codebook \
  --input-jsonl /path/to/raw_manifest.jsonl \
  --codebook-path /path/to/vq_codebook_streaming.pt \
  --report-path /path/to/vq_codebook_streaming.evaluation.report.json \
  --device cuda:0 \
  --muq-cache-dir /path/to/muq_cache

# 3) 按多 seed / 多 frame budget 批量搜索候选
python -m training.preprocess.search_vq_codebook_candidates \
  --input-jsonl /path/to/raw_manifest.jsonl \
  --output-dir /path/to/vq_candidates \
  --seeds 11 17 29 \
  --frame-budgets 200000 400000 \
  --device cuda:0 \
  --muq-cache-dir /path/to/muq_cache
```

其中 `--max-total-train-frames 0` 和 `--max-total-heldout-frames 0` 都表示对应 split 使用 full ceiling，不再按 frame budget 提前截断。

`fit_vq_codebook_streaming.py`、`evaluate_vq_codebook.py`、`search_vq_codebook_candidates.py` 与 `build_dataset.py` 现在都支持共享 `--muq-cache-dir`。同一首歌命中缓存后会直接复用已对齐到 `target_fps` 的 MuQ embedding，只在 cache miss 时重新计算。

这条“效果优先”工作流的设计目标是：

- 保持与论文和当前训练实现一致的 **MuQ + 单层 16384 code + 25fps**
- 保持输出仍然兼容 `extract_sketch.py` / `build_dataset.py`
- 通过 heldout `quantization_mse`、`dead_code_ratio`、`usage_entropy`、`top_1_usage_share` 先筛掉明显塌缩的候选
- 再用小规模 SongBloom proxy run 做最终决策，而不是只看聚类误差

推荐的候选筛选顺序：

1. 先剔除 `dead_code_ratio` 过高或 `top_1_usage_share` 过大的候选
2. 在剩余候选里优先看更低的 `quantization_mse`
3. 对前 1 到 2 个候选分别构建小规模 `processed_dataset`
4. 使用 `training/configs/songbloom_vq_proxy_eval.yaml` 跑 100-step proxy run，比对 `train/L_LM`

proxy run 命令示例：

```bash
python -m training.train --config training/configs/songbloom_vq_proxy_eval.yaml
```

其中 `fit_vq_codebook_streaming.py` 会额外写出：

- `vq_codebook_streaming.pt`
- `vq_codebook_streaming.meta.json`
- `vq_codebook_streaming.report.json`

而 `search_vq_codebook_candidates.py` 会在输出目录下汇总一个：

- `candidate_summary.json`

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
  --vq-ckpt /path/to/vq_codebook.pt \
  --muq-cache-dir /path/to/muq_cache
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
- `prompt_wav.flac` 默认优先从首个副歌起点开始截取最多 10 秒；若副歌接近结尾则只保留可用尾段，若无副歌则回退到旧的 vocal / 能量启发式

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
