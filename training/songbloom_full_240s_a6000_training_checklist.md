# SongBloom 240s 训练前检查清单

适用范围：

- 仓库：`/home/nicola/song/songbloom_dev`
- 模型配置：`training/configs/songbloom_full_240s_train.yaml`
- 当前机器：单卡 `NVIDIA RTX A6000`
- 当前已存在的数据目录：`/home/nicola/song/songbloom_dev/test_train/processed_dataset`

本文档基于当前仓库真实代码、真实配置和本地 smoke 结果整理，目标是给出一套可以直接落地的 240s 训练起步建议，而不是泛泛的理论说明。

## 1. 当前结论

如果你现在就是基于仓库里现成的 `test_train/processed_dataset` 开始训练，推荐先使用：

```yaml
training:
  lr: 1e-4
  warmup_steps: 2000
  max_steps: 150000
  flow_loss_weight: 0.1
  accumulate_grad_batches: 8
  gradient_clip_val: 1.0
  val_check_interval: 5000

data:
  train_dir: /home/nicola/song/songbloom_dev/test_train/processed_dataset
  val_dir: ""
  batch_size: 1
  num_workers: 0
```

核心判断：

- 当前这张 A6000 上，不建议直接使用 `batch_size: 4`
- 当前这份小数据集更适合先用 `batch_size: 1`
- 当前仓库没有现成验证集时，`val_dir: ""` 是合理配置
- 对当前 28 条可训练样本，`accumulate_grad_batches: 8` 比默认的 `32` 更适合作为工程调试起点

## 2. 当前仓库里的真实数据状态

当前仓库中能直接用于训练的候选目录只有：

```text
/home/nicola/song/songbloom_dev/test_train/processed_dataset
```

实际检查结果：

- `processed_dataset` 下共有 29 个一级子目录
- 其中真正可训练的样本目录有 28 个
- 额外的 1 个目录是 `_workspace`，不会进入训练

这是因为 `training/dataset.py` 只会收集带 `meta.json` 的样本目录。

## 3. 当前数据集的时长分布

基于现有 `meta.json` 统计，28 条可用样本的时长情况如下：

- 最短：`118.4s`
- 中位数：`197.44s`
- 最长：`319.36s`

其中有 7 条样本超过 `240s`。这些样本不会直接报错，而是会在训练时根据 `max_duration=240` 裁剪到 240s 后再进入模型。

示例长样本：

- `til_death_barcelona_bee09c3f96347198` -> `319.36s`
- `94_via_satellite_souls_of_mischief_the_funkee_ho_a05c7f66619f286d` -> `296.96s`
- `cello_song_nick_drake_1632723010113a16` -> `284.8s`

## 4. 已验证的 240s 训练边界

### 4.1 可行配置

我已经用临时 240s 配置做过真实 smoke，以下组合可以跑通：

```yaml
data:
  train_dir: /home/nicola/song/songbloom_dev/test_train/processed_dataset
  val_dir: ""
  batch_size: 1
  num_workers: 0

training:
  max_steps: 1
  accumulate_grad_batches: 1
```

运行结果：

- 成功完成 1 个优化 step
- Lightning 正常输出 `train/loss`、`train/L_LM`、`train/L_flow`
- 训练按 `max_steps=1` 正常退出
- 240s 条件器日志显示 `resolution = 0.24`

这说明当前 240s 主训练链路已经真实打通，而不是只在 60s mock 下可用。

### 4.2 不可行配置

我也实际测试了：

```yaml
data:
  batch_size: 2
  num_workers: 0

training:
  max_steps: 1
  accumulate_grad_batches: 1
```

在当前 A6000 上会在第一步反向传播时直接 OOM。

实际报错关键信息：

- `torch.cuda.OutOfMemoryError`
- 额外申请约 `10.40 GiB`

因此对当前这张卡，240s 训练不要再继续硬顶 `batch_size: 2`。

## 5. 为什么推荐 `accumulate_grad_batches: 8`

当前可训练样本数只有 28 条，而 `training/datamodule.py` 的 train loader 使用了：

- `shuffle=True`
- `drop_last=True`

所以每个 epoch 的 batch 数大致如下：

- `batch_size=1` -> 28 batches / epoch
- `batch_size=2` -> 14 batches / epoch
- `batch_size=4` -> 7 batches / epoch

如果在这 28 条小样本上仍使用默认：

```yaml
training:
  accumulate_grad_batches: 32
```

那么单次 optimizer step 会非常稀疏，不利于当前这份小数据做工程调试。

因此推荐：

- 工程调试起步：`accumulate_grad_batches: 8`
- 更稳一些：`16`
- 当前这份 28 条数据上，不建议一开始就用 `32`

## 6. 当前正式训练前的建议配置

如果你的目标是“先在当前仓库现成数据上稳定开跑 240s 训练”，建议直接从

`training/configs/songbloom_full_240s_train.yaml`

改成下面这组值：

```yaml
training:
  lr: 1e-4
  warmup_steps: 2000
  max_steps: 150000
  flow_loss_weight: 0.1
  accumulate_grad_batches: 8
  gradient_clip_val: 1.0
  val_check_interval: 5000

data:
  train_dir: /home/nicola/song/songbloom_dev/test_train/processed_dataset
  val_dir: ""
  batch_size: 1
  num_workers: 0
```

## 7. 如果以后换成更大的正式训练集

对更大的数据集，调参顺序建议如下：

1. 固定 `batch_size=1`
2. 先用 `accumulate_grad_batches=8` 跑通
3. 稳定后再试 `16`
4. 再根据吞吐和收敛情况决定是否上到 `32`

如果你的目标是尽量接近论文里的全局 batch size `128`，而仍保持单卡 `batch_size=1`，可按下面方式估算：

- 1 GPU -> `accumulate_grad_batches=128`
- 2 GPU -> `64`
- 4 GPU -> `32`
- 8 GPU -> `16`
- 16 GPU -> `8`

但这只是“有效 batch 近似对齐论文”，不代表对当前小数据集就是合理配置。

## 8. 当前训练前检查清单

### 数据路径

- 训练目录使用：`/home/nicola/song/songbloom_dev/test_train/processed_dataset`
- 不要把 `_workspace` 当成训练样本目录

### 验证集

- 当前仓库没有现成 `val_dir`
- 如果只是先把 240s 主链路跑起来，`val_dir: ""` 是合理的
- 当前 trainer 在 `val_dir` 为空时会自动关闭 validation 和 sanity check

### batch 与 accumulate

- 单卡 A6000：先锁定 `batch_size=1`
- 当前这 28 条数据：先用 `accumulate_grad_batches=8`

### DataLoader

- 已验证最稳的起步值是 `num_workers=0`
- 后续稳定后可以再尝试 `2` 或 `4`

### 配置文件选择

- 不要把 `training/configs/songbloom_full_240s_mock_train.yaml` 当正式训练配置
- 该文件默认 `max_steps=1`，只适合 smoke 测试

### 长样本处理

- 当前数据里有 7 条样本超过 `240s`
- 这些样本会被按 `max_duration=240` 截断，这是预期行为，不是预处理错误

## 9. 建议的验证命令

### 先跑 1-step smoke

```bash
python -m training.train --config training/configs/songbloom_full_240s_mock_train.yaml
```

### 再跑 240s 主配置的小步验证

建议先把正式配置改成：

- `train_dir=/home/nicola/song/songbloom_dev/test_train/processed_dataset`
- `val_dir=""`
- `batch_size=1`
- `num_workers=0`
- `accumulate_grad_batches=8`

然后短步数验证：

```bash
python -m training.train --config training/configs/songbloom_full_240s_train.yaml
```

如果只想先做一次短验证，可临时把：

- `training.max_steps` 改成 `1` 或 `10`

## 10. 停止规则

- 如果 `240s + batch_size=1` 都跑不通，再回头排查代码或环境
- 如果 `batch_size=1` 能跑通，但 `batch_size=2` OOM，就不要继续在当前 A6000 上硬顶 batch，应该转去调 `accumulate_grad_batches`

## 11. 本文档对应的关键事实

本清单基于以下事实整理：

- 当前仓库唯一现成训练目录是 `test_train/processed_dataset`
- 当前可训练样本数为 28
- 240s 配置在 A6000 上已真实完成单步训练
- 240s 配置在 `batch_size=2` 时会 OOM
- 当前仓库已经支持 `val_dir=""` 时跳过 validation / sanity check

如果后续你切换到新的正式数据集，这份文档里的核心原则仍然成立，但样本数、时长分布和最优 `accumulate_grad_batches` 需要重新按真实数据统计一遍。
