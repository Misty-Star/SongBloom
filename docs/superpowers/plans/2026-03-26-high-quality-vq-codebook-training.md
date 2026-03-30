# 高质量 VQ Codebook 训练管线 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `training/` 内落地一个“效果优先”的 `vq_codebook.pt` 训练与筛选流程：冻结 MuQ、保持单层 `16384` code、保持 `25fps` 对齐，并输出与 `training/preprocess/extract_sketch.py` / `training/preprocess/build_dataset.py` 完全兼容的 `{"codebook.weight": ...}`。

**Architecture:** 保持论文同构，不引入 RVQ、多层离散器或连续 sketch 分支。新增一条独立于当前 `training/preprocess/fit_vq_codebook_kmeans.py` 的高质量路径：`balanced frame sampling -> streaming k-means refinement -> heldout quantization evaluation -> SongBloom proxy selection`。现有 `fit_vq_codebook_kmeans.py` 保留为 bootstrap fallback，新路径只新增 `training/` 下的训练脚本、评估脚本、测试和一份 proxy 配置。

**Tech Stack:** Python 3.8, PyTorch, torchaudio, tqdm, unittest, existing MuQ loader, existing SongBloom training configs

---

## Scope And Guardrails

- 只改 `training/` 和本计划文件；不要修改 `SongBloom/models/` 主训练逻辑。
- 保持 `extract_sketch.py` / `build_dataset.py` 的输入输出契约不变：
  - codebook 仍然是单层最近邻量化
  - 输出 ckpt 仍然只有 `codebook.weight`
  - `num_codes` 仍然固定为 `16384`
  - `target_fps` 仍然固定为 `25`
- 保留 `training/preprocess/fit_vq_codebook_kmeans.py`，把它视为“快速 bootstrap / fallback”，不要把它改造成一锅端的大脚本。
- 候选 codebook 的最终选择必须走两段验证：
  - 先看 heldout quantization report
  - 再看小规模 SongBloom proxy run 的 `train/L_LM`

## File Map

- Create: `training/preprocess/vq_codebook_dataset.py`
  - 职责：读取 manifest / 音频目录、做确定性 train/heldout 切分、做 per-audio capped frame sampling、统一 `25fps` 对齐。
- Create: `training/preprocess/vq_codebook_trainer.py`
  - 职责：实现 streaming K-Means / online centroid update、dead-code reset、usage 统计、state 序列化。
- Create: `training/preprocess/fit_vq_codebook_streaming.py`
  - 职责：训练单个候选 codebook，写出 `.pt`、`.meta.json`、`.report.json`。
- Create: `training/preprocess/evaluate_vq_codebook.py`
  - 职责：在 heldout 帧上计算 quantization MSE、dead-code ratio、usage entropy、top-k coverage。
- Create: `training/preprocess/search_vq_codebook_candidates.py`
  - 职责：按多 seed / 多采样预算批量训练多个候选 codebook，并汇总排序。
- Create: `training/configs/songbloom_vq_proxy_eval.yaml`
  - 职责：给候选 codebook 做 100-step 左右的 SongBloom proxy run。
- Create: `training/tests/test_vq_codebook_dataset.py`
- Create: `training/tests/test_vq_codebook_trainer.py`
- Create: `training/tests/test_fit_vq_codebook_streaming.py`
- Create: `training/tests/test_evaluate_vq_codebook.py`
- Create: `training/tests/test_search_vq_codebook_candidates.py`
- Create: `training/tests/test_songbloom_vq_proxy_eval_config.py`
- Modify: `training/preprocess/README.md`

## Acceptance Criteria

- 训练脚本输出的 `vq_codebook.pt` 可直接被 `training/preprocess/extract_sketch.py` 加载，无需改任何调用方。
- 候选搜索输出至少包含：
  - `quantization_mse`
  - `dead_code_ratio`
  - `usage_entropy`
  - `top_1_usage_share`
  - `num_train_frames`
  - `num_heldout_frames`
- 候选筛选规则明确、可重复：
  - 先剔除明显塌缩的 codebook（如 `dead_code_ratio` 过高或 `top_1_usage_share` 过大）
  - 再按 heldout `quantization_mse` 排序
  - 最后用 proxy run 的 `train/L_LM` 选最终版本
- README 明确区分：
  - `fit_vq_codebook_kmeans.py` = 快速兼容替代
  - `fit_vq_codebook_streaming.py` + `evaluate_vq_codebook.py` + `search_vq_codebook_candidates.py` = 效果优先推荐流程

### Task 1: 确定性采样与 heldout 切分

**Files:**
- Create: `training/preprocess/vq_codebook_dataset.py`
- Test: `training/tests/test_vq_codebook_dataset.py`

- [ ] **Step 1: 写失败测试，锁定切分和采样契约**

```python
import unittest

from training.preprocess.vq_codebook_dataset import (
    sample_frames_from_embedding,
    split_audio_paths,
)


class VQCodebookDatasetTest(unittest.TestCase):
    def test_split_audio_paths_is_deterministic_and_disjoint(self):
        audio_paths = [f"/tmp/song_{i}.flac" for i in range(10)]
        train_a, heldout_a = split_audio_paths(audio_paths, heldout_ratio=0.2, seed=7)
        train_b, heldout_b = split_audio_paths(audio_paths, heldout_ratio=0.2, seed=7)
        self.assertEqual(train_a, train_b)
        self.assertEqual(heldout_a, heldout_b)
        self.assertTrue(set(train_a).isdisjoint(heldout_a))

    def test_sample_frames_from_embedding_respects_per_audio_cap(self):
        embedding = torch.arange(0, 120, dtype=torch.float32).view(1, 15, 8)
        sampled = sample_frames_from_embedding(embedding, max_frames=4, rng=random.Random(3))
        self.assertEqual(tuple(sampled.shape), (4, 8))
```

- [ ] **Step 2: 运行测试，确认当前缺少实现**

Run: `python -m unittest training.tests.test_vq_codebook_dataset -v`  
Expected: FAIL，报 `ModuleNotFoundError` 或 `ImportError`

- [ ] **Step 3: 实现最小可用的数据模块**

```python
def split_audio_paths(audio_paths, heldout_ratio, seed):
    ordered = sorted(audio_paths)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    heldout_count = max(1, int(len(ordered) * heldout_ratio))
    return ordered[heldout_count:], ordered[:heldout_count]


def sample_frames_from_embedding(embedding, max_frames, rng):
    frames = embedding.squeeze(0).detach().cpu().float()
    if frames.shape[0] <= max_frames:
        return frames
    indices = sorted(rng.sample(range(frames.shape[0]), max_frames))
    return frames[indices]
```

- [ ] **Step 4: 补上 train/heldout 元数据和 `25fps` 对齐测试后重新跑**

Run: `python -m unittest training.tests.test_vq_codebook_dataset -v`  
Expected: PASS，覆盖 deterministic split、per-audio cap、target fps 对齐

- [ ] **Step 5: 提交这一层基础设施**

```bash
git add training/preprocess/vq_codebook_dataset.py training/tests/test_vq_codebook_dataset.py
git commit -m "feat: add deterministic vq codebook frame sampling"
```

### Task 2: Streaming K-Means 核心与 dead-code reset

**Files:**
- Create: `training/preprocess/vq_codebook_trainer.py`
- Test: `training/tests/test_vq_codebook_trainer.py`

- [ ] **Step 1: 写失败测试，锁定 centroid update 和 dead-code reset**

```python
import unittest

from training.preprocess.vq_codebook_trainer import StreamingKMeans


class StreamingKMeansTest(unittest.TestCase):
    def test_partial_fit_updates_cluster_counts(self):
        trainer = StreamingKMeans(num_codes=4, embed_dim=2, device="cpu", seed=0)
        trainer.initialize(torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]))
        trainer.partial_fit(torch.tensor([[0.1, 0.0], [9.9, 10.1]]))
        self.assertEqual(int(trainer.counts.sum().item()), 2)

    def test_refresh_dead_codes_refills_empty_centroids(self):
        trainer = StreamingKMeans(num_codes=3, embed_dim=2, device="cpu", seed=0)
        trainer.centers = torch.zeros(3, 2)
        trainer.counts = torch.tensor([10.0, 0.0, 0.0])
        trainer.refresh_dead_codes(torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
        self.assertTrue(torch.all(trainer.counts > 0))
```

- [ ] **Step 2: 跑测试，确认 trainer 还不存在**

Run: `python -m unittest training.tests.test_vq_codebook_trainer -v`  
Expected: FAIL，报 `ModuleNotFoundError`

- [ ] **Step 3: 实现最小 trainer，不引入新依赖**

```python
class StreamingKMeans:
    def __init__(self, num_codes, embed_dim, device, seed):
        self.num_codes = num_codes
        self.embed_dim = embed_dim
        self.device = device
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.centers = None
        self.counts = None

    def partial_fit(self, batch):
        distances = torch.cdist(batch, self.centers)
        assignments = distances.argmin(dim=-1)
        # update centers and counts

    def refresh_dead_codes(self, reserve):
        dead = self.counts <= 0
        self.centers[dead] = reserve[: int(dead.sum().item())]
        self.counts[dead] = 1.0
```

- [ ] **Step 4: 扩展测试，覆盖 usage stats、state_dict、load_state_dict**

Run: `python -m unittest training.tests.test_vq_codebook_trainer -v`  
Expected: PASS，覆盖 partial fit、dead-code reset、save/load round-trip

- [ ] **Step 5: 提交聚类核心**

```bash
git add training/preprocess/vq_codebook_trainer.py training/tests/test_vq_codebook_trainer.py
git commit -m "feat: add streaming kmeans trainer for vq codebook"
```

### Task 3: 单候选 codebook 训练 CLI

**Files:**
- Create: `training/preprocess/fit_vq_codebook_streaming.py`
- Test: `training/tests/test_fit_vq_codebook_streaming.py`

- [ ] **Step 1: 写失败测试，锁定 CLI 输出契约**

```python
import json
import tempfile
import unittest
from unittest import mock


class FitVQCodebookStreamingCLITest(unittest.TestCase):
    def test_main_writes_codebook_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = f"{temp_dir}/vq_codebook.pt"
            argv = [
                "prog",
                "--input-jsonl", "/tmp/raw_manifest.jsonl",
                "--output-path", output_path,
                "--device", "cpu",
            ]
            with mock.patch("sys.argv", argv):
                from training.preprocess.fit_vq_codebook_streaming import main
                main()
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(os.path.exists(output_path.replace(".pt", ".meta.json")))
            self.assertTrue(os.path.exists(output_path.replace(".pt", ".report.json")))
```

- [ ] **Step 2: 跑 CLI 测试，确认入口脚本还不存在**

Run: `python -m unittest training.tests.test_fit_vq_codebook_streaming -v`  
Expected: FAIL，报 `ModuleNotFoundError`

- [ ] **Step 3: 实现 CLI，复用现有 MuQ loader 和新 trainer**

```python
def main():
    args = parse_args()
    train_audio, heldout_audio = split_audio_paths(...)
    train_frames = collect_embedding_samples(...)
    heldout_frames = collect_embedding_samples(...)
    trainer = StreamingKMeans(...)
    trainer.fit(train_frames)
    save_codebook(args.output_path, trainer.centers, metadata)
    save_report(args.output_path.replace(".pt", ".report.json"), report)
```

- [ ] **Step 4: 增加失败保护测试，再重新跑**

Run: `python -m unittest training.tests.test_fit_vq_codebook_streaming -v`  
Expected: PASS，覆盖以下场景：
- 缺少音频时报错
- `num_codes > sampled_frames` 时报错
- 正常路径会写出 `.pt`、`.meta.json`、`.report.json`

- [ ] **Step 5: 提交单候选训练入口**

```bash
git add training/preprocess/fit_vq_codebook_streaming.py training/tests/test_fit_vq_codebook_streaming.py
git commit -m "feat: add high-quality vq codebook training cli"
```

### Task 4: Heldout evaluator

**Files:**
- Create: `training/preprocess/evaluate_vq_codebook.py`
- Test: `training/tests/test_evaluate_vq_codebook.py`

- [ ] **Step 1: 写失败测试，锁定评估指标**

```python
import unittest

from training.preprocess.evaluate_vq_codebook import summarize_assignments


class EvaluateVQCodebookTest(unittest.TestCase):
    def test_summarize_assignments_reports_dead_code_ratio_and_entropy(self):
        assignments = torch.tensor([0, 0, 1, 2, 2, 2])
        summary = summarize_assignments(assignments, num_codes=4)
        self.assertAlmostEqual(summary["dead_code_ratio"], 0.25)
        self.assertIn("usage_entropy", summary)
        self.assertIn("top_1_usage_share", summary)
```

- [ ] **Step 2: 跑测试，确认 evaluator 还不存在**

Run: `python -m unittest training.tests.test_evaluate_vq_codebook -v`  
Expected: FAIL，报 `ModuleNotFoundError`

- [ ] **Step 3: 实现最小 evaluator**

```python
def summarize_assignments(assignments, num_codes):
    counts = torch.bincount(assignments, minlength=num_codes).float()
    probs = counts / counts.sum().clamp_min(1.0)
    return {
        "dead_code_ratio": float((counts == 0).float().mean().item()),
        "top_1_usage_share": float(probs.max().item()),
        "usage_entropy": float((-(probs[probs > 0] * probs[probs > 0].log())).sum().item()),
    }
```

- [ ] **Step 4: 扩展到完整 CLI，并跑测试**

Run: `python -m unittest training.tests.test_evaluate_vq_codebook -v`  
Expected: PASS，覆盖：
- 从 heldout 音频提取帧
- 加载 `vq_codebook.pt`
- 计算 `quantization_mse`
- 写出 `evaluation.report.json`

- [ ] **Step 5: 提交评估脚本**

```bash
git add training/preprocess/evaluate_vq_codebook.py training/tests/test_evaluate_vq_codebook.py
git commit -m "feat: add heldout vq codebook evaluation"
```

### Task 5: 候选搜索与排序

**Files:**
- Create: `training/preprocess/search_vq_codebook_candidates.py`
- Test: `training/tests/test_search_vq_codebook_candidates.py`

- [ ] **Step 1: 写失败测试，锁定候选汇总输出**

```python
import unittest

from training.preprocess.search_vq_codebook_candidates import rank_candidates


class SearchVQCodebookCandidatesTest(unittest.TestCase):
    def test_rank_candidates_filters_collapsed_codebooks_first(self):
        rows = [
            {"name": "good", "dead_code_ratio": 0.01, "top_1_usage_share": 0.01, "quantization_mse": 0.5},
            {"name": "collapsed", "dead_code_ratio": 0.40, "top_1_usage_share": 0.30, "quantization_mse": 0.1},
        ]
        ranked = rank_candidates(rows, max_dead_code_ratio=0.05, max_top1_share=0.05)
        self.assertEqual(ranked[0]["name"], "good")
```

- [ ] **Step 2: 跑测试，确认搜索脚本还不存在**

Run: `python -m unittest training.tests.test_search_vq_codebook_candidates -v`  
Expected: FAIL，报 `ModuleNotFoundError`

- [ ] **Step 3: 实现最小候选搜索器**

```python
def rank_candidates(rows, max_dead_code_ratio, max_top1_share):
    valid = [
        row for row in rows
        if row["dead_code_ratio"] <= max_dead_code_ratio
        and row["top_1_usage_share"] <= max_top1_share
    ]
    return sorted(valid, key=lambda row: row["quantization_mse"])
```

- [ ] **Step 4: 扩展 CLI，支持多 seed / 多预算训练并汇总报告**

Run: `python -m unittest training.tests.test_search_vq_codebook_candidates -v`  
Expected: PASS，覆盖：
- 生成多个候选命名目录
- 汇总 `candidate_summary.json`
- 排出 `best_candidate`

- [ ] **Step 5: 提交候选搜索器**

```bash
git add training/preprocess/search_vq_codebook_candidates.py training/tests/test_search_vq_codebook_candidates.py
git commit -m "feat: add vq codebook candidate search workflow"
```

### Task 6: SongBloom proxy run 配置

**Files:**
- Create: `training/configs/songbloom_vq_proxy_eval.yaml`
- Test: `training/tests/test_songbloom_vq_proxy_eval_config.py`

- [ ] **Step 1: 写失败测试，锁定 proxy config 的关键值**

```python
import unittest

from training.train import load_config


class SongBloomVQProxyEvalConfigTest(unittest.TestCase):
    def test_proxy_eval_config_keeps_num_pitch_and_uses_longer_smoke(self):
        cfg = load_config("training/configs/songbloom_vq_proxy_eval.yaml")
        self.assertEqual(cfg["model"]["num_pitch"], 16384)
        self.assertEqual(cfg["training"]["max_steps"], 100)
        self.assertEqual(cfg["data"]["batch_size"], 1)
```

- [ ] **Step 2: 跑测试，确认配置文件还不存在**

Run: `python -m unittest training.tests.test_songbloom_vq_proxy_eval_config -v`  
Expected: FAIL，报找不到配置文件

- [ ] **Step 3: 基于现有 mock config 创建 proxy config**

```yaml
precision: 'bf16-mixed'
min_dur: 80
max_dur: 60
training:
  max_steps: 100
data:
  train_dir: /tmp/replace_me
  val_dir: ""
  batch_size: 1
  num_workers: 0
model:
  num_pitch: 16384
```

- [ ] **Step 4: 跑配置测试，并补一条执行说明**

Run: `python -m unittest training.tests.test_songbloom_vq_proxy_eval_config -v`  
Expected: PASS

Run: `python -m training.train --config training/configs/songbloom_vq_proxy_eval.yaml`  
Expected: 能启动训练并打印 `train/L_LM`

- [ ] **Step 5: 提交 proxy 配置**

```bash
git add training/configs/songbloom_vq_proxy_eval.yaml training/tests/test_songbloom_vq_proxy_eval_config.py
git commit -m "feat: add proxy config for vq codebook selection"
```

### Task 7: README 与操作手册

**Files:**
- Modify: `training/preprocess/README.md`

- [ ] **Step 1: 写 README 失败检查清单**

```text
- README 必须解释为什么 streaming 方案是效果优先路径
- README 必须保留 kmeans fallback 说明
- README 必须给出 train -> evaluate -> search -> proxy run 的命令顺序
```

- [ ] **Step 2: 先只更新目录和命令块，不改解释段落**

Run: `rg -n "fit_vq_codebook_kmeans|build_dataset|vq_codebook" training/preprocess/README.md`  
Expected: 能定位旧说明位置

- [ ] **Step 3: 补充推荐工作流和筛选规则**

```markdown
python -m training.preprocess.fit_vq_codebook_streaming ...
python -m training.preprocess.evaluate_vq_codebook ...
python -m training.preprocess.search_vq_codebook_candidates ...
python -m training.train --config training/configs/songbloom_vq_proxy_eval.yaml
```

- [ ] **Step 4: 跑一次针对性测试回归**

Run: `python -m unittest training.tests.test_vq_codebook_dataset training.tests.test_vq_codebook_trainer training.tests.test_fit_vq_codebook_streaming training.tests.test_evaluate_vq_codebook training.tests.test_search_vq_codebook_candidates training.tests.test_songbloom_vq_proxy_eval_config -v`  
Expected: PASS

- [ ] **Step 5: 提交 README 更新**

```bash
git add training/preprocess/README.md
git commit -m "docs: document high-quality vq codebook workflow"
```

## Candidate Selection Runbook

1. 先训练 3 到 5 个候选 codebook：

```bash
python -m training.preprocess.search_vq_codebook_candidates \
  --input-jsonl /path/to/raw_manifest.jsonl \
  --output-dir /path/to/vq_candidates \
  --seeds 11 17 29 \
  --frame-budgets 200000 400000 \
  --device cuda:0
```

2. 用 heldout 报告先做第一轮筛选：
   - `dead_code_ratio <= 0.05`
   - `top_1_usage_share <= 0.05`
   - 在通过门槛的候选里优先选择更低的 `quantization_mse`

3. 对前 2 个候选各自构建一份小型 `processed_dataset`：

```bash
python -m training.preprocess.build_dataset \
  --input-jsonl /path/to/tiny_ready_manifest.jsonl \
  --output-dir /path/to/proxy_dataset_candidate_a \
  --vq-ckpt /path/to/candidate_a.pt
```

4. 分别运行 100-step proxy run：

```bash
python -m training.train --config training/configs/songbloom_vq_proxy_eval.yaml
```

5. 最终选择规则：
   - 如果某个候选在 proxy run 中明显更快收敛、`train/L_LM` 更低，选它
   - 如果 proxy run 差异很小，选 heldout `quantization_mse` 更低且 `dead_code_ratio` 更稳的版本

## Final Verification Checklist

- `python -m unittest training.tests.test_vq_codebook_dataset -v`
- `python -m unittest training.tests.test_vq_codebook_trainer -v`
- `python -m unittest training.tests.test_fit_vq_codebook_streaming -v`
- `python -m unittest training.tests.test_evaluate_vq_codebook -v`
- `python -m unittest training.tests.test_search_vq_codebook_candidates -v`
- `python -m unittest training.tests.test_songbloom_vq_proxy_eval_config -v`
- `python -m training.preprocess.fit_vq_codebook_streaming --help`
- `python -m training.preprocess.evaluate_vq_codebook --help`
- `python -m training.preprocess.search_vq_codebook_candidates --help`

## Stop Rule

- 完成代码、测试和 README 后先停下，不要自动更新 `llmdoc/`。
- 按仓库约束，最后一步必须让用户决定是否执行“使用 recorder agent 更新项目文档”。
- 只有用户明确同意后，才进入文档更新动作。

Plan complete and saved to `docs/superpowers/plans/2026-03-26-high-quality-vq-codebook-training.md`. Ready to execute?
