# SongBloom 训练代码还原实施计划

## 需求重述

基于论文 arXiv:2506.07634 和现有推理代码，在 `training/` 目录中还原完整的模型训练流程。现有代码中 `MVSA_DiTAR.forward()` 已完整实现训练前向传播，`SongBloom_PL` 仅有模型构建，缺少 `training_step`、`configure_optimizers`、数据集加载等训练逻辑。

## 现有代码资产

| 已实现 | 位置 |
|--------|------|
| 训练前向传播 `MVSA_DiTAR.forward()` → `DiTAROutput` | `songbloom_mvsa.py:176-272` |
| VAE 冻结 + 模型构建 `SongBloom_PL.__init__()` | `songbloom_pl.py:25-55` |
| 条件组装参考 `_prepare_tokens_and_attributes()` | `songbloom_pl.py:163-230` |
| 全部条件器、VAE、Transformer 代码 | `SongBloom/models/` |

## 实施阶段

### Phase 1: SongBloom_PL 训练方法补全

**文件:** `SongBloom/models/songbloom/songbloom_pl.py`（修改现有文件）

添加以下方法到 `SongBloom_PL` 类：

1. **`training_step(self, batch, batch_idx)`**
   - 从 batch 解包: `x_sketch, x_latent, x_len, attributes`
   - CFG dropout: `attributes = [self.model.cfg_dropout(attr) for attr in attributes]`
   - 条件处理: `tokenized = self.model.condition_provider.tokenize(attributes)` → `condition_tensors = self.model.condition_provider(tokenized)`
   - 前向: `output = self.model(x_sketch, x_latent, x_len, condition_tensors)`
   - 损失计算:
     - `L_LM = F.cross_entropy(output.ar_logit.transpose(1,2), output.ar_target, ignore_index=self.model.special_token_id)`
     - `L_flow = F.mse_loss(output.nar_pred, output.nar_target)`
     - `loss = L_LM + 0.1 * L_flow`
   - 日志: `self.log_dict({"loss": loss, "L_LM": L_LM, "L_flow": L_flow})`

2. **`configure_optimizers(self)`**
   - AdamW, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1
   - CosineAnnealingLR with warmup 2000 步（使用 `get_cosine_schedule_with_warmup` 或手写 LambdaLR）
   - 排除 VAE 参数（已 frozen）

3. **`validation_step(self, batch, batch_idx)`**（可选，结构同 training_step，仅计算损失不反传）

### Phase 2: 训练数据集

**文件:** `training/dataset.py`（新建）

实现 `SongBloomDataset(torch.utils.data.Dataset)`：

- 数据源: 预处理后的目录结构，每个样本包含:
  - `x_sketch.pt` — `(T,)` LongTensor（MuQ+VQ 离线预计算）
  - `x_latent.pt` — `(64, T)` FloatTensor（VAE 离线预计算）
  - `lyrics.txt` — G2P 处理后的音素字符串
  - `prompt_wav.flac` — 参考音频（~10s）
  - `meta.json` — `{duration, structure_duration}`
- `__getitem__` 返回: `(x_sketch, x_latent, x_len, ConditioningAttributes)`
- 长度对齐: `x_len = floor(duration * 25 / block_size) * block_size`，截断/填充到 `x_len`
- collate_fn: 批内 padding 到最长样本

### Phase 3: DataModule

**文件:** `training/datamodule.py`（新建）

实现 `SongBloomDataModule(pl.LightningDataModule)`：
- 封装 train/val DataLoader
- 配置 batch_size、num_workers、sampler

### Phase 4: 训练入口脚本

**文件:** `training/train.py`（新建）

- 加载 YAML 配置（复用 `infer.py` 的 `load_config` 模式）
- 构建 `SongBloom_PL` + `SongBloomDataModule`
- 配置 Lightning Trainer:
  - `strategy="deepspeed_stage_2"`
  - `precision="bf16-mixed"`
  - `max_steps=150000`
  - `accumulate_grad_batches` 按需设置
  - `gradient_clip_val=1.0`
  - callbacks: ModelCheckpoint, LearningRateMonitor
- 支持 `--resume` 从 checkpoint 恢复

### Phase 5: 训练配置

**文件:** `training/configs/songbloom_full_240s_train.yaml`（新建）

基于 `pretrained/songbloom_full_240s.yaml` 扩展，添加:
- `training.lr`, `training.warmup_steps`, `training.max_steps`
- `training.batch_size`, `training.accumulate_grad_batches`
- `training.gradient_clip_val`
- `data.train_dir`, `data.val_dir`, `data.num_workers`

### Phase 6: 数据预处理脚本

**文件:** `training/preprocess/`（新建目录）

1. **`extract_sketch.py`** — MuQ + VQ 量化生成 x_sketch（需要 MuQ 模型权重）
2. **`extract_latent.py`** — StableVAE 批量编码生成 x_latent
3. **`prepare_lyrics.py`** — G2P 转换 + 结构标注
4. **`extract_prompt.py`** — 截取参考音频片段

## 风险评估

| 风险 | 级别 | 说明 |
|------|------|------|
| MuQ 模型不可用 | **高** | x_sketch 生成依赖外部 MuQ 模型，代码库中未包含。需要找到 MuQ 权重或用替代方案 |
| 显存不足 | 中 | 2B 参数模型 + VAE，单卡可能需要 DeepSpeed ZeRO-3 或梯度累积 |
| 数据格式假设 | 中 | 数据集格式基于论文推断，实际格式可能不同 |
| DPO 训练 | 低 | 论文未公开 DPO 细节，暂不实现 |

## 实施顺序

1. Phase 1（最关键，补全 training_step）→ 可立即验证前向传播和损失计算
2. Phase 2 + 3（数据加载）→ 需要有预处理好的数据才能端到端测试
3. Phase 4 + 5（训练脚本和配置）→ 整合为可运行的训练流程
4. Phase 6（预处理脚本）→ 从原始数据到训练数据的完整管线

## 文件变更清单

| 操作 | 文件 |
|------|------|
| **修改** | `SongBloom/models/songbloom/songbloom_pl.py` |
| 新建 | `training/__init__.py` |
| 新建 | `training/dataset.py` |
| 新建 | `training/datamodule.py` |
| 新建 | `training/train.py` |
| 新建 | `training/configs/songbloom_full_240s_train.yaml` |
| 新建 | `training/preprocess/extract_sketch.py` |
| 新建 | `training/preprocess/extract_latent.py` |
| 新建 | `training/preprocess/prepare_lyrics.py` |
| 新建 | `training/preprocess/extract_prompt.py` |
