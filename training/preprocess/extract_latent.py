"""提取 VAE latent（x_latent）。

用法:
    python -m training.preprocess.extract_latent \
        --audio-dir /path/to/audio \
        --output-dir /path/to/output \
        --vae-cfg pretrained/stable_audio_1920_vae.json \
        --vae-ckpt pretrained/autoencoder_music_dsp1920.ckpt
"""

import argparse
import os

import torch
import torchaudio
from tqdm import tqdm

from SongBloom.models.vae_frontend import StableVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--vae-cfg", type=str, default="pretrained/stable_audio_1920_vae.json")
    parser.add_argument("--vae-ckpt", type=str, default="pretrained/autoencoder_music_dsp1920.ckpt")
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    vae = StableVAE(vae_cfg=args.vae_cfg, vae_ckpt=args.vae_ckpt, sr=args.sr).eval().to(args.device)

    audio_files = [f for f in os.listdir(args.audio_dir) if f.endswith((".wav", ".flac", ".mp3"))]

    for fname in tqdm(audio_files, desc="Extracting latent"):
        stem = os.path.splitext(fname)[0]
        out_dir = os.path.join(args.output_dir, stem)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, "x_latent.pt")
        if os.path.exists(out_path):
            continue

        wav, sr = torchaudio.load(os.path.join(args.audio_dir, fname))
        if sr != args.sr:
            wav = torchaudio.functional.resample(wav, sr, args.sr)

        with torch.no_grad():
            latent = vae.encode(wav.unsqueeze(0).to(args.device))  # (1, 64, T)

        torch.save(latent.squeeze(0).cpu(), out_path)


if __name__ == "__main__":
    main()
