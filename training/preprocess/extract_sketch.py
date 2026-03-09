"""提取 sketch tokens（x_sketch）：MuQ 嵌入 + VQ 量化。

依赖:
    pip install muq

MuQ 模型: https://huggingface.co/OpenMuQ/MuQ-large-msd-iter

注意: VQ 量化层（codebook 16384, 单层）需要单独训练或从作者处获取权重。
      本脚本提供 MuQ 嵌入提取 + VQ 量化的完整流程框架。

用法:
    python -m training.preprocess.extract_sketch \
        --audio-dir /path/to/audio \
        --output-dir /path/to/output \
        --vq-ckpt /path/to/vq_codebook.pt
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


class SingleLayerVQ(nn.Module):
    """单层向量量化，codebook 大小 16384。"""

    def __init__(self, dim: int, codebook_size: int = 16384):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → indices: (B, T)"""
        dists = torch.cdist(x, self.codebook.weight.unsqueeze(0))
        return dists.argmin(dim=-1)

    @classmethod
    def from_pretrained(cls, path: str):
        state = torch.load(path, map_location="cpu")
        dim = state["codebook.weight"].shape[1]
        codebook_size = state["codebook.weight"].shape[0]
        model = cls(dim, codebook_size)
        model.load_state_dict(state)
        return model


def load_muq_model(model_name="OpenMuQ/MuQ-large-msd-iter", device="cuda:0"):
    """加载 MuQ 模型。需要 pip install muq。"""
    from muq import MuQ
    model = MuQ.from_pretrained(model_name)
    return model.eval().to(device)


def extract_muq_embeddings(model, wav: torch.Tensor, sr: int = 48000) -> torch.Tensor:
    """提取 MuQ 嵌入。

    Args:
        model: MuQ 模型
        wav: (C, T) 音频波形
        sr: 采样率

    Returns:
        embeddings: (1, T_frames, D) 语义嵌入
    """
    # MuQ 输入要求: 单声道, 16kHz
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    with torch.no_grad():
        embeddings = model(wav.unsqueeze(0).to(next(model.parameters()).device))

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--vq-ckpt", type=str, required=True, help="VQ codebook 权重路径")
    parser.add_argument("--muq-model", type=str, default="OpenMuQ/MuQ-large-msd-iter")
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--target-fps", type=int, default=25, help="目标帧率，需与 VAE latent 对齐")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    muq = load_muq_model(args.muq_model, args.device)
    vq = SingleLayerVQ.from_pretrained(args.vq_ckpt).eval().to(args.device)

    audio_files = [f for f in os.listdir(args.audio_dir) if f.endswith((".wav", ".flac", ".mp3"))]

    for fname in tqdm(audio_files, desc="Extracting sketch"):
        stem = os.path.splitext(fname)[0]
        out_dir = os.path.join(args.output_dir, stem)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, "x_sketch.pt")
        if os.path.exists(out_path):
            continue

        wav, sr = torchaudio.load(os.path.join(args.audio_dir, fname))
        if sr != args.sr:
            wav = torchaudio.functional.resample(wav, sr, args.sr)

        # MuQ 嵌入提取
        embeddings = extract_muq_embeddings(muq, wav, args.sr)  # (1, T_muq, D)

        # 重采样到目标帧率 (25fps) — MuQ 输出帧率可能不同
        # 使用 nearest 插值对齐到 VAE latent 帧数
        target_frames = int(wav.shape[-1] / args.sr * args.target_fps)
        if embeddings.shape[1] != target_frames:
            embeddings = F.interpolate(
                embeddings.transpose(1, 2), size=target_frames, mode="nearest"
            ).transpose(1, 2)

        # VQ 量化
        with torch.no_grad():
            sketch_tokens = vq.encode(embeddings)  # (1, T)

        torch.save(sketch_tokens.squeeze(0).cpu(), out_path)


if __name__ == "__main__":
    main()
