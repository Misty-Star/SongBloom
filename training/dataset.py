"""SongBloom 训练数据集。

数据目录结构（每个样本一个子目录）：
    sample_dir/
        x_sketch.pt      — (T,) LongTensor, MuQ+VQ 离线预计算
        x_latent.pt      — (64, T) FloatTensor, VAE 离线预计算
        lyrics.txt        — G2P 处理后的音素字符串
        prompt_wav.flac   — 参考音频 (~10s)
        meta.json         — {"duration": float, "structure_duration": [...]}
"""

import json
import os
import typing as tp

import torch
import torchaudio
from torch.utils.data import Dataset

from SongBloom.models.musicgen.conditioners import (
    ConditioningAttributes,
    WavCondition,
)
from training.dataset_utils import compute_effective_frame_length


class SongBloomDataset(Dataset):
    """加载预处理后的 SongBloom 训练数据。"""

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 48000,
        block_size: int = 16,
        max_duration: float = 240.0,
        prompt_len: float = 10.0,
    ):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.max_duration = max_duration
        self.prompt_samples = int(prompt_len * sample_rate)

        # 收集所有包含 meta.json 的子目录
        self.samples: tp.List[str] = []
        for name in sorted(os.listdir(data_dir)):
            sample_path = os.path.join(data_dir, name)
            if os.path.isdir(sample_path) and os.path.exists(os.path.join(sample_path, "meta.json")):
                self.samples.append(sample_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]

        # 加载预计算数据
        x_sketch = torch.load(os.path.join(path, "x_sketch.pt"), map_location="cpu")
        x_latent = torch.load(os.path.join(path, "x_latent.pt"), map_location="cpu")

        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        # 计算有效帧数（对齐到 block_size）
        x_len = compute_effective_frame_length(
            sketch_frames=x_sketch.shape[0],
            latent_frames=x_latent.shape[-1],
            duration_seconds=meta.get("duration"),
            block_size=self.block_size,
            max_duration=self.max_duration,
        )

        # 截断到有效长度
        x_sketch = x_sketch[:x_len]
        x_latent = x_latent[:, :x_len]

        # 加载歌词
        with open(os.path.join(path, "lyrics.txt"), "r") as f:
            lyrics = f.read().strip()

        # 加载参考音频
        prompt_path = os.path.join(path, "prompt_wav.flac")
        prompt_wav, sr = torchaudio.load(prompt_path)
        if sr != self.sample_rate:
            prompt_wav = torchaudio.functional.resample(prompt_wav, sr, self.sample_rate)
        if prompt_wav.shape[0] > 1:
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)
        prompt_wav = prompt_wav[:, :self.prompt_samples]

        # 构造 ConditioningAttributes
        attr = ConditioningAttributes()
        attr.text["lyrics"] = lyrics
        attr.wav["prompt_wav"] = WavCondition(
            wav=prompt_wav.unsqueeze(0),  # (1, C, T)
            length=torch.tensor([prompt_wav.shape[-1]]),
            sample_rate=[self.sample_rate],
            path=[None],
            seek_time=[0.0],
        )

        # structure_duration 存入 attr.text 供条件器时长插值使用
        if "structure_duration" in meta:
            attr.text["structure_duration"] = meta["structure_duration"]

        return x_sketch, x_latent, torch.tensor(x_len, dtype=torch.long), attr


def collate_fn(batch):
    """批内 padding 到最长样本。"""
    sketches, latents, lengths, attrs = zip(*batch)

    max_len = max(s.shape[0] for s in sketches)
    B = len(sketches)
    latent_dim = latents[0].shape[0]

    # Pad x_sketch with EOS token (16384)
    eos_id = 16384
    x_sketch = torch.full((B, max_len), eos_id, dtype=torch.long)
    x_latent = torch.zeros(B, latent_dim, max_len)
    x_len = torch.stack(list(lengths))

    for i, (s, l) in enumerate(zip(sketches, latents)):
        T = s.shape[0]
        x_sketch[i, :T] = s
        x_latent[i, :, :T] = l

    return x_sketch, x_latent, x_len, list(attrs)
